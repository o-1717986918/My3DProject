#!/usr/bin/env python3
"""Screen a retained finite-motion actor against the exact RCSS soccer ball."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from my3d_rl.apollo_walk_cpu import ApolloWalkCpu
from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.reference_dynamics import configure_pd_actuators
from my3d_rl.soccer_motion_ball import (
    classify_ball_contacts,
    deterministic_ball_placement_perturbation,
    place_reference_ball_xy,
    select_reference_strike,
)
from my3d_rl.soccer_motion_corpus import SoccerMotionCorpus, load_soccer_motion_corpus
from my3d_rl.soccer_motion_policy import (
    SoccerMotionPolicy,
    load_soccer_motion_policy,
    soccer_motion_actor_observation,
)
from my3d_rl.soccer_motion_reset import (
    derive_case_seed,
    deterministic_reset_perturbation,
    yaw_quaternion_rotate,
    yaw_vector_rotate,
)
from my3d_rl.t1_control import APOLLO_DEFAULT_POSE, apollo_joint_gains


REPOSITORY_ROOT = Path(__file__).parents[2]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_tree_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _reference_foot_centers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    corpus: SoccerMotionCorpus,
    *,
    motion: int,
    root_qpos: int,
    joint_qpos: np.ndarray,
    foot_geom: int,
) -> np.ndarray:
    length = int(corpus.lengths[motion])
    result = np.empty((length, 3), dtype=np.float64)
    for frame in range(length):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[root_qpos : root_qpos + 3] = corpus.root_position[motion, frame]
        data.qpos[root_qpos + 3 : root_qpos + 7] = (
            corpus.root_quaternion_wxyz[motion, frame]
        )
        data.qpos[joint_qpos] = corpus.joint_position[motion, frame]
        mujoco.mj_forward(model, data)
        result[frame] = data.geom_xpos[foot_geom]
    return result


def _evaluate_case(
    *,
    model: mujoco.MjModel,
    corpus: SoccerMotionCorpus,
    policy: SoccerMotionPolicy,
    contract: Any,
    motion: int,
    start_frame: int,
    replicate: int,
    strike_frame: int,
    reference_strike_distance_m: float,
    reference_strike_speed_mps: float,
    ball_radius_offset_m: float,
    ball_arc_angle_rad: float,
    target_angle_rad: float,
    post_motion_frames: int,
    post_contact_controller: str,
    recovery_blend_frames: int,
    apollo_walk: ApolloWalkCpu | None,
    perturbation_seed: int | None,
    reset_joint_noise: float,
    reset_root_velocity_noise: float,
    reset_yaw_range: float,
    ball_radius_noise_m: float,
    ball_arc_noise_rad: float,
    arrays: dict[str, Any],
) -> dict[str, Any]:
    length = int(corpus.lengths[motion])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    root_qpos = arrays["root_qpos"]
    root_dof = arrays["root_dof"]
    joint_qpos = arrays["joint_qpos"]
    joint_dof = arrays["joint_dof"]
    model_root_xy = arrays["model_root_xy"]

    reset = (
        deterministic_reset_perturbation(
            base_seed=perturbation_seed,
            motion=motion,
            start_frame=start_frame,
            action_size=contract.action_size,
            joint_noise=reset_joint_noise,
            root_velocity_noise=reset_root_velocity_noise,
            yaw_range=reset_yaw_range,
        )
        if perturbation_seed is not None
        else None
    )
    ball_perturbation = (
        deterministic_ball_placement_perturbation(
            base_seed=perturbation_seed,
            motion=motion,
            start_frame=start_frame,
            radius_noise_m=ball_radius_noise_m,
            arc_noise_rad=ball_arc_noise_rad,
        )
        if perturbation_seed is not None
        else None
    )
    reset_yaw = reset.yaw if reset is not None else 0.0
    joint_noise = (
        reset.joint_position_noise
        if reset is not None
        else np.zeros(contract.action_size, dtype=np.float64)
    )
    root_velocity_noise = (
        reset.root_velocity_noise
        if reset is not None
        else np.zeros(6, dtype=np.float64)
    )
    sampled_radius_offset = ball_radius_offset_m + (
        ball_perturbation.radius_offset_m if ball_perturbation is not None else 0.0
    )
    sampled_arc_angle = ball_arc_angle_rad + (
        ball_perturbation.arc_angle_rad if ball_perturbation is not None else 0.0
    )

    data.qpos[root_qpos : root_qpos + 2] = model_root_xy
    data.qpos[root_qpos + 2] = corpus.root_position[motion, start_frame, 2]
    data.qpos[root_qpos + 3 : root_qpos + 7] = yaw_quaternion_rotate(
        corpus.root_quaternion_wxyz[motion, start_frame], reset_yaw
    )
    data.qpos[joint_qpos] = np.clip(
        corpus.joint_position[motion, start_frame] + joint_noise,
        arrays["lower"],
        arrays["upper"],
    )
    data.qvel[root_dof : root_dof + 3] = yaw_vector_rotate(
        corpus.root_linear_velocity[motion, start_frame], reset_yaw
    ) + root_velocity_noise[:3]
    data.qvel[root_dof + 3 : root_dof + 6] = yaw_vector_rotate(
        corpus.root_angular_velocity[motion, start_frame], reset_yaw
    ) + root_velocity_noise[3:]
    data.qvel[joint_dof] = corpus.joint_velocity[motion, start_frame]

    ball_xy = place_reference_ball_xy(
        corpus.root_position[motion, :length, :2],
        start_frame=start_frame,
        model_root_xy=model_root_xy,
        yaw_rad=reset_yaw,
        radius_offset_m=sampled_radius_offset,
        arc_angle_rad=sampled_arc_angle,
    )
    ball_qpos = arrays["ball_qpos"]
    ball_dof = arrays["ball_dof"]
    data.qpos[ball_qpos : ball_qpos + 3] = np.array(
        [ball_xy[0], ball_xy[1], arrays["ball_radius"]]
    )
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[ball_dof : ball_dof + 6] = 0.0
    data.ctrl[arrays["tau_actuator"]] = 0.0
    data.ctrl[arrays["vel_actuator"]] = 0.0
    data.ctrl[arrays["pos_actuator"]] = corpus.joint_position[motion, start_frame]
    mujoco.mj_forward(model, data)

    initial_ball = data.xpos[arrays["ball_body"]].copy()
    initial_pairs = np.asarray(data.contact.geom[: data.ncon], dtype=np.int32)
    initial_contact = classify_ball_contacts(
        initial_pairs,
        ball_geom=arrays["ball_geom"],
        left_foot_geom=arrays["left_foot_geom"],
        right_foot_geom=arrays["right_foot_geom"],
        robot_geoms=arrays["robot_geoms"],
    )
    if initial_contact.any_robot:
        return {
            "motion": motion,
            "relative_path": corpus.relative_paths[motion],
            "start_frame": start_frame,
            "strike_frame": strike_frame,
            "lead_frames": strike_frame - start_frame,
            "initial_overlap": True,
            "completed_reference": False,
            "fell": False,
            "termination_reason": "initial_ball_robot_overlap",
            "any_robot_contact": True,
            "correct_foot_contact": False,
            "wrong_foot_contact": False,
            "contact_screening_passed": False,
            "maximum_progress_m": 0.0,
            "maximum_ball_speed_mps": 0.0,
        }

    target_direction = np.array(
        [
            np.cos(target_angle_rad + reset_yaw),
            np.sin(target_angle_rad + reset_yaw),
        ],
        dtype=np.float64,
    )
    lateral_direction = np.array(
        [-target_direction[1], target_direction[0]], dtype=np.float64
    )
    previous_action = np.zeros(contract.action_size, dtype=np.float64)
    correct_is_left = bool(corpus.kick_leg_one_hot[motion, 0] > 0.5)
    any_robot_contact = False
    correct_foot_contact = False
    wrong_foot_contact = False
    first_correct_contact_sim_step: int | None = None
    maximum_progress = 0.0
    maximum_lateral = 0.0
    maximum_ball_speed = 0.0
    maximum_directional_speed = 0.0
    minimum_torso_height = float("inf")
    minimum_upright = float("inf")
    fell = False
    invalid = False
    reached_reference_end = False
    recovery_start_target: np.ndarray | None = None
    recovery_control_frames = 0
    recovery_walk_action = np.zeros(contract.action_size, dtype=np.float64)
    sim_step = 0
    terminal_frame = start_frame
    total_control_frames = length - 1 - start_frame + post_motion_frames

    for offset in range(1, total_control_frames + 1):
        frame = min(start_frame + offset, length - 1)
        current = max(start_frame, frame - 1)
        if post_contact_controller != "reference" and recovery_start_target is not None:
            recovery_control_frames += 1
            fraction = min(recovery_control_frames / recovery_blend_frames, 1.0)
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            if post_contact_controller == "apollo_stand":
                if apollo_walk is None:
                    raise RuntimeError("Apollo recovery controller was not loaded")
                recovery_target, recovery_walk_action = apollo_walk.target(
                    data,
                    recovery_walk_action,
                    np.zeros(3, dtype=np.float64),
                )
            else:
                recovery_target = APOLLO_DEFAULT_POSE
            target = np.clip(
                (1.0 - fraction) * recovery_start_target
                + fraction * recovery_target,
                arrays["lower"],
                arrays["upper"],
            )
        else:
            observation = soccer_motion_actor_observation(
                data,
                joint_qpos=joint_qpos,
                joint_dof=joint_dof,
                gyro_slice=arrays["gyro_slice"],
                torso_site=arrays["torso_site"],
                reference_joint_position=corpus.joint_position[motion, current],
                reference_joint_velocity=corpus.joint_velocity[motion, current],
                reference_root_linear_velocity=(
                    corpus.root_linear_velocity[motion, current]
                ),
                reference_root_angular_velocity=(
                    corpus.root_angular_velocity[motion, current]
                ),
                reference_contact=corpus.foot_contact[motion, current],
                previous_action=previous_action,
                progress=current / max(length - 1, 1),
                kick_leg_one_hot=corpus.kick_leg_one_hot[motion],
            )
            action = np.clip(policy(observation), *contract.action_clip)
            target = np.clip(
                corpus.joint_position[motion, frame]
                + contract.action_scale * action,
                arrays["lower"],
                arrays["upper"],
            )
            previous_action = action
        data.ctrl[arrays["pos_actuator"]] = target
        terminal_frame = frame

        for _ in range(arrays["substeps"]):
            mujoco.mj_step(model, data)
            sim_step += 1
            pairs = np.asarray(data.contact.geom[: data.ncon], dtype=np.int32)
            contact = classify_ball_contacts(
                pairs,
                ball_geom=arrays["ball_geom"],
                left_foot_geom=arrays["left_foot_geom"],
                right_foot_geom=arrays["right_foot_geom"],
                robot_geoms=arrays["robot_geoms"],
            )
            any_robot_contact |= contact.any_robot
            correct_now = contact.left_foot if correct_is_left else contact.right_foot
            wrong_now = contact.right_foot if correct_is_left else contact.left_foot
            if correct_now and first_correct_contact_sim_step is None:
                first_correct_contact_sim_step = sim_step
                recovery_start_target = target.copy()
            correct_foot_contact |= correct_now
            wrong_foot_contact |= wrong_now

            displacement = data.xpos[arrays["ball_body"], :2] - initial_ball[:2]
            velocity = data.qvel[ball_dof : ball_dof + 2]
            maximum_progress = max(
                maximum_progress, float(np.dot(displacement, target_direction))
            )
            maximum_lateral = max(
                maximum_lateral, float(abs(np.dot(displacement, lateral_direction)))
            )
            maximum_ball_speed = max(maximum_ball_speed, float(np.linalg.norm(velocity)))
            maximum_directional_speed = max(
                maximum_directional_speed, float(np.dot(velocity, target_direction))
            )

        rotation = data.site_xmat[arrays["torso_site"]].reshape(3, 3)
        torso_height = float(data.xpos[arrays["torso_body"], 2])
        upright = float(rotation[2, 2])
        minimum_torso_height = min(minimum_torso_height, torso_height)
        minimum_upright = min(minimum_upright, upright)
        invalid = not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
        fell = torso_height < 0.35 or upright < 0.20
        reached_reference_end |= frame >= length - 1
        if invalid or fell:
            break

    final_displacement = data.xpos[arrays["ball_body"], :2] - initial_ball[:2]
    final_progress = float(np.dot(final_displacement, target_direction))
    final_lateral = float(abs(np.dot(final_displacement, lateral_direction)))
    contact_screening_passed = (
        correct_foot_contact and maximum_progress >= 0.08 and not fell and not invalid
    )
    return {
        "motion": motion,
        "relative_path": corpus.relative_paths[motion],
        "kick_leg": "left" if correct_is_left else "right",
        "start_frame": start_frame,
        "replicate": replicate,
        "strike_frame": strike_frame,
        "lead_frames": strike_frame - start_frame,
        "reference_strike_center_distance_m": reference_strike_distance_m,
        "reference_strike_planar_speed_mps": reference_strike_speed_mps,
        "reset_perturbation_seed": reset.case_seed if reset is not None else None,
        "ball_perturbation_seed": (
            ball_perturbation.case_seed if ball_perturbation is not None else None
        ),
        "reset_yaw_rad": reset_yaw,
        "ball_radius_offset_m": sampled_radius_offset,
        "ball_arc_angle_rad": sampled_arc_angle,
        "initial_ball_xy_m": initial_ball[:2].tolist(),
        "initial_overlap": False,
        "completed_reference": reached_reference_end and not fell and not invalid,
        "fell": fell,
        "invalid": invalid,
        "termination_reason": (
            "non_finite_state" if invalid else "fall" if fell else "completed"
        ),
        "terminal_frame": terminal_frame,
        "any_robot_contact": any_robot_contact,
        "correct_foot_contact": correct_foot_contact,
        "wrong_foot_contact": wrong_foot_contact,
        "first_correct_contact_sim_step": first_correct_contact_sim_step,
        "post_contact_controller": post_contact_controller,
        "recovery_blend_frames": recovery_blend_frames,
        "recovery_control_frames": recovery_control_frames,
        "contact_screening_passed": contact_screening_passed,
        "maximum_progress_m": maximum_progress,
        "final_progress_m": final_progress,
        "maximum_lateral_error_m": maximum_lateral,
        "final_lateral_error_m": final_lateral,
        "maximum_ball_speed_mps": maximum_ball_speed,
        "maximum_directional_speed_mps": maximum_directional_speed,
        "minimum_torso_height_m": minimum_torso_height,
        "minimum_upright": minimum_upright,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY_ROOT / "training/contracts/soccer_motion_policy_v2.yaml",
    )
    parser.add_argument(
        "--lead-frames", type=int, nargs="+", default=[4, 8, 16, 32, 64]
    )
    parser.add_argument("--include-start-zero", action="store_true")
    parser.add_argument("--motions", type=int, nargs="+")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--ball-radius-offset-m", type=float, default=0.0)
    parser.add_argument("--ball-arc-angle-deg", type=float, default=0.0)
    parser.add_argument("--target-angle-deg", type=float, default=0.0)
    parser.add_argument("--post-motion-frames", type=int, default=25)
    parser.add_argument("--perturbation-seed", type=int)
    parser.add_argument("--reset-joint-noise", type=float, default=0.0)
    parser.add_argument("--reset-root-velocity-noise", type=float, default=0.0)
    parser.add_argument("--reset-yaw-range", type=float, default=0.0)
    parser.add_argument("--ball-radius-noise-m", type=float, default=0.0)
    parser.add_argument("--ball-arc-noise-deg", type=float, default=0.0)
    parser.add_argument(
        "--post-contact-controller",
        choices=("reference", "blend_default_pose", "apollo_stand"),
        default="reference",
    )
    parser.add_argument("--recovery-blend-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.zero_policy == (args.checkpoint is not None):
        raise ValueError("select exactly one of --zero-policy or --checkpoint")
    if (
        not args.lead_frames
        or any(value < 1 for value in args.lead_frames)
        or args.replicates < 1
        or args.post_motion_frames < 0
        or args.recovery_blend_frames < 1
        or min(
            args.reset_joint_noise,
            args.reset_root_velocity_noise,
            args.reset_yaw_range,
            args.ball_radius_noise_m,
            args.ball_arc_noise_deg,
        )
        < 0.0
        or not np.isfinite(
            [
                args.ball_radius_offset_m,
                args.ball_arc_angle_deg,
                args.target_angle_deg,
            ]
        ).all()
    ):
        raise ValueError("invalid motion-guided ball screening grid")
    if args.perturbation_seed is not None and args.perturbation_seed < 0:
        raise ValueError("perturbation seed must be non-negative")
    if args.perturbation_seed is None and any(
        value > 0.0
        for value in (
            args.reset_joint_noise,
            args.reset_root_velocity_noise,
            args.reset_yaw_range,
            args.ball_radius_noise_m,
            args.ball_arc_noise_deg,
        )
    ):
        raise ValueError("non-zero perturbations require --perturbation-seed")
    if args.replicates > 1 and args.perturbation_seed is None:
        raise ValueError("multiple replicates require --perturbation-seed")

    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    corpus = load_soccer_motion_corpus(args.corpus_root)
    motions = list(range(corpus.motion_count)) if args.motions is None else args.motions
    if not motions or len(set(motions)) != len(motions):
        raise ValueError("motion selection must be non-empty and unique")
    if any(value < 0 or value >= corpus.motion_count for value in motions):
        raise ValueError("motion index is outside the corpus")
    policy = load_soccer_motion_policy(
        zero_policy=args.zero_policy,
        checkpoint=args.checkpoint,
        profile_name=args.profile,
        policy_contract_name=contract.policy_name,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )

    prefix = "soccer_ball_cpu_"
    model = build_single_t1_soccer_model(prefix=prefix, robot_x=-10.0, robot_y=0.0)
    model.opt.timestep = 0.005
    gains = np.asarray(
        [apollo_joint_gains(name) for name in contract.joint_order], dtype=np.float64
    )
    tau_actuator, pos_actuator, vel_actuator = configure_pd_actuators(
        model, contract, kp=gains[:, 0], kd=gains[:, 1], prefix=prefix
    )
    joint_qpos = np.asarray(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    joint_dof = np.asarray(
        [model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
    )
    root = model.joint(prefix + "root")
    root_qpos = int(root.qposadr[0])
    root_dof = int(root.dofadr[0])
    torso_body = model.body(prefix + "torso").id
    torso_site = model.site(prefix + "torso").id
    gyro = model.sensor(prefix + "torso_gyro")
    ball_joint = model.joint("ball-root")
    ball_geom = model.geom("ball")
    left_foot_geom = model.geom(prefix + "left_foot").id
    right_foot_geom = model.geom(prefix + "right_foot").id
    robot_geoms = frozenset(
        index
        for index in range(model.ngeom)
        if (model.geom(index).name or "").startswith(prefix)
    )
    lower = model.jnt_range[model.dof_jntid[joint_dof], 0]
    upper = model.jnt_range[model.dof_jntid[joint_dof], 1]
    arrays = {
        "tau_actuator": tau_actuator,
        "pos_actuator": pos_actuator,
        "vel_actuator": vel_actuator,
        "joint_qpos": joint_qpos,
        "joint_dof": joint_dof,
        "root_qpos": root_qpos,
        "root_dof": root_dof,
        "model_root_xy": model.qpos0[root_qpos : root_qpos + 2].copy(),
        "torso_body": torso_body,
        "torso_site": torso_site,
        "gyro_slice": slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0]),
        "ball_body": model.body("ball").id,
        "ball_qpos": int(ball_joint.qposadr[0]),
        "ball_dof": int(ball_joint.dofadr[0]),
        "ball_geom": ball_geom.id,
        "ball_radius": float(ball_geom.size[0]),
        "left_foot_geom": left_foot_geom,
        "right_foot_geom": right_foot_geom,
        "robot_geoms": robot_geoms,
        "lower": lower,
        "upper": upper,
        "substeps": round((1.0 / contract.frequency_hz) / model.opt.timestep),
    }
    apollo_walk = (
        ApolloWalkCpu(model, contract, prefix=prefix)
        if args.post_contact_controller == "apollo_stand"
        else None
    )

    kinematic_data = mujoco.MjData(model)
    records: list[dict[str, Any]] = []
    for motion in motions:
        length = int(corpus.lengths[motion])
        foot_geom = (
            left_foot_geom
            if corpus.kick_leg_one_hot[motion, 0] > 0.5
            else right_foot_geom
        )
        foot_centers = _reference_foot_centers(
            model,
            kinematic_data,
            corpus,
            motion=motion,
            root_qpos=root_qpos,
            joint_qpos=joint_qpos,
            foot_geom=foot_geom,
        )
        reference_ball = np.array(
            [
                corpus.root_position[motion, length - 1, 0],
                corpus.root_position[motion, length - 1, 1],
                arrays["ball_radius"],
            ],
            dtype=np.float64,
        )
        strike = select_reference_strike(
            foot_centers, reference_ball, frequency_hz=contract.frequency_hz
        )
        starts = {max(0, strike.frame - lead) for lead in args.lead_frames}
        if args.include_start_zero:
            starts.add(0)
        starts = {value for value in starts if value < length - 1}
        for start_frame in sorted(starts):
            for replicate in range(args.replicates):
                case_perturbation_seed = (
                    derive_case_seed(
                        args.perturbation_seed,
                        motion,
                        start_frame,
                        replicate,
                    )
                    if args.perturbation_seed is not None
                    else None
                )
                records.append(
                    _evaluate_case(
                        model=model,
                        corpus=corpus,
                        policy=policy,
                        contract=contract,
                        motion=motion,
                        start_frame=start_frame,
                        replicate=replicate,
                        strike_frame=strike.frame,
                        reference_strike_distance_m=strike.center_distance_m,
                        reference_strike_speed_mps=strike.planar_speed_mps,
                        ball_radius_offset_m=args.ball_radius_offset_m,
                        ball_arc_angle_rad=np.deg2rad(args.ball_arc_angle_deg),
                        target_angle_rad=np.deg2rad(args.target_angle_deg),
                        post_motion_frames=args.post_motion_frames,
                        post_contact_controller=args.post_contact_controller,
                        recovery_blend_frames=args.recovery_blend_frames,
                        apollo_walk=apollo_walk,
                        perturbation_seed=case_perturbation_seed,
                        reset_joint_noise=args.reset_joint_noise,
                        reset_root_velocity_noise=args.reset_root_velocity_noise,
                        reset_yaw_range=args.reset_yaw_range,
                        ball_radius_noise_m=args.ball_radius_noise_m,
                        ball_arc_noise_rad=np.deg2rad(args.ball_arc_noise_deg),
                        arrays=arrays,
                    )
                )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["motion"])].append(record)
    per_motion = []
    for motion, items in sorted(grouped.items()):
        per_motion.append(
            {
                "motion": motion,
                "relative_path": corpus.relative_paths[motion],
                "trials": len(items),
                "correct_foot_contacts": sum(
                    bool(item["correct_foot_contact"]) for item in items
                ),
                "screening_passes": sum(
                    bool(item["contact_screening_passed"]) for item in items
                ),
                "falls": sum(bool(item["fell"]) for item in items),
                "initial_overlaps": sum(
                    bool(item["initial_overlap"]) for item in items
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "purpose": "k2_exact_cpu_motion_guided_ball_contact_screening",
        "promotion_authorized": False,
        "promotion_blocker": (
            "screening only; requires a frozen target-conditioned K2 policy, "
            "three untouched seeds, arrival-speed gates and server replay"
        ),
        "engine": f"MuJoCo {mujoco.__version__}",
        "policy": "zero_residual" if args.zero_policy else "checkpoint",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_tree_sha256": (
            _checkpoint_tree_sha256(args.checkpoint) if args.checkpoint else None
        ),
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256_file(args.contract),
        "corpus_root": str(args.corpus_root.resolve()),
        "corpus_hashes": list(corpus.sha256),
        "protocol": {
            "ball_placement": "remaining_reference_anchor_endpoint_arc",
            "lead_frames": args.lead_frames,
            "replicates": args.replicates,
            "include_start_zero": args.include_start_zero,
            "ball_radius_offset_m": args.ball_radius_offset_m,
            "ball_arc_angle_deg": args.ball_arc_angle_deg,
            "target_angle_deg": args.target_angle_deg,
            "post_motion_frames": args.post_motion_frames,
            "post_contact_controller": args.post_contact_controller,
            "recovery_blend_frames": args.recovery_blend_frames,
            "perturbation_seed": args.perturbation_seed,
            "reset_joint_noise": args.reset_joint_noise,
            "reset_root_velocity_noise": args.reset_root_velocity_noise,
            "reset_yaw_range": args.reset_yaw_range,
            "ball_radius_noise_m": args.ball_radius_noise_m,
            "ball_arc_noise_deg": args.ball_arc_noise_deg,
            "perturbation_generator": "numpy_seedsequence_uint63_v2",
            "contact_definition": "exact_ball_geom_to_labelled_foot_geom_pair",
            "screening_definition": (
                "correct_foot_contact_and_progress_ge_0.08m_and_no_fall_or_invalid"
            ),
        },
        "trial_count": len(records),
        "correct_foot_contacts": sum(
            bool(record["correct_foot_contact"]) for record in records
        ),
        "wrong_foot_contacts": sum(
            bool(record["wrong_foot_contact"]) for record in records
        ),
        "screening_passes": sum(
            bool(record["contact_screening_passed"]) for record in records
        ),
        "falls": sum(bool(record["fell"]) for record in records),
        "initial_overlaps": sum(
            bool(record["initial_overlap"]) for record in records
        ),
        "per_motion": per_motion,
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
