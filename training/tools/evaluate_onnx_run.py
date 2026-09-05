#!/usr/bin/env python3
"""CPU MuJoCo acceptance for deployed 78/80 -> 23 locomotion policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.motion_reference import validate_motion_reference
from my3d_rl.policy_symmetry import (
    mirror_run_action,
    mirror_run_observation,
    training_mirror_map,
)
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.reference_dynamics import circular_interpolate
from my3d_rl.run_env import NOMINAL_TRAINING_POSE, TRAIN_TO_SERVER_SIGN


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v1.yaml"
PHASE_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v2.yaml"
DEFAULT_MODEL = REPOSITORY_ROOT / "mujococodebase" / "skills" / "walk" / "walk.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _reference_quaternion(
    reference_quaternion_wxyz: np.ndarray, phase: float
) -> np.ndarray:
    """Match DirectionalRun's shortest-path normalized linear interpolation."""
    frame = (phase % 1.0) * reference_quaternion_wxyz.shape[0]
    lower = int(np.floor(frame)) % reference_quaternion_wxyz.shape[0]
    upper = (lower + 1) % reference_quaternion_wxyz.shape[0]
    fraction = frame - np.floor(frame)
    upper_value = reference_quaternion_wxyz[upper]
    if np.dot(reference_quaternion_wxyz[lower], upper_value) < 0.0:
        upper_value = -upper_value
    result = (1.0 - fraction) * reference_quaternion_wxyz[
        lower
    ] + fraction * upper_value
    return result / max(float(np.linalg.norm(result)), 1.0e-8)


def _yaw_rotate(vector: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]]
    )


def _has_contact(data: mujoco.MjData, geom_a: int, geom_b: int) -> bool:
    pairs = data.contact.geom[: data.ncon]
    return bool(
        np.any(
            ((pairs[:, 0] == geom_a) & (pairs[:, 1] == geom_b))
            | ((pairs[:, 0] == geom_b) & (pairs[:, 1] == geom_a))
        )
    )


def _max_consecutive(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _box_geometric_contact(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    geom_id: int,
    *,
    pitch_height: float,
    tolerance: float = 0.01,
) -> bool:
    """Mirror the MJX-Warp contact proxy using an oriented box lower bound."""
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    lowest_z = data.geom_xpos[geom_id, 2] - np.sum(
        np.abs(rotation[2, :]) * model.geom_size[geom_id]
    )
    return bool(lowest_z <= pitch_height + tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--motion-reference",
        type=Path,
        help="required by motion-reference residual contracts",
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--action-scale",
        type=float,
        help="explicit stress-test override; defaults to the policy contract",
    )
    parser.add_argument("--gait-frequency", type=float, default=1.75)
    parser.add_argument(
        "--contact-proxy-tolerance",
        type=float,
        default=0.01,
        help="training contact-proxy tolerance used only for parity diagnostics",
    )
    symmetry_group = parser.add_mutually_exclusive_group()
    symmetry_group.add_argument("--symmetry-ensemble", action="store_true")
    symmetry_group.add_argument(
        "--mirror-policy",
        action="store_true",
        help="run only the reflected observation/action branch",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.action_scale is not None and not 0.0 < args.action_scale <= 1.0:
        raise ValueError("action-scale must be in (0, 1]")
    if args.contact_proxy_tolerance < 0.0:
        raise ValueError("contact-proxy-tolerance must be non-negative")

    model = build_single_t1_soccer_model(prefix="accept_", robot_x=-10.0)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    actor_size = input_meta.shape[-1]
    if actor_size not in (78, 80) or output_meta.shape[-1] != 23:
        raise ValueError(
            f"unexpected ONNX boundary {input_meta.shape} -> {output_meta.shape}"
        )
    contract_path = args.contract or (
        PHASE_CONTRACT if actor_size == 80 else DEFAULT_CONTRACT
    )
    contract = load_policy_contract(contract_path)
    reference_centered = (
        contract.control_mode == "motion_reference_residual_joint_position"
    )
    if reference_centered and args.motion_reference is None:
        raise ValueError("reference-centred policy requires --motion-reference")
    if not reference_centered and args.motion_reference is not None:
        raise ValueError(
            "--motion-reference is only valid for reference-centred policy"
        )
    reference_validation = None
    reference: dict[str, np.ndarray] | None = None
    reference_nominal_frequency = None
    reference_forward_speed = None
    gait_frequency = args.gait_frequency
    reference_velocity_scale = 1.0
    if args.motion_reference is not None:
        reference_validation = validate_motion_reference(args.motion_reference)
        if not reference_validation["passed"]:
            raise ValueError(
                "motion reference validation failed: "
                + "; ".join(reference_validation["errors"])
            )
        if (
            contract.reference_sha256 is not None
            and reference_validation["sha256"] != contract.reference_sha256
        ):
            raise ValueError("motion reference SHA-256 differs from policy contract")
        with np.load(args.motion_reference, allow_pickle=False) as archive:
            reference = {
                name: np.asarray(archive[name])
                for name in (
                    "root_position",
                    "root_quaternion_xyzw",
                    "root_linear_velocity",
                    "root_angular_velocity",
                    "joint_position",
                    "joint_velocity",
                    "foot_contact",
                )
            }
        frame_count = reference["joint_position"].shape[0]
        reference_nominal_frequency = contract.frequency_hz / frame_count
        reference_forward_speed = float(
            np.mean(reference["root_linear_velocity"][:, 0])
        )
        if reference_forward_speed <= 0.0:
            raise ValueError("reference must have positive mean forward velocity")
        gait_frequency = (
            reference_nominal_frequency * abs(args.vx) / reference_forward_speed
            if np.linalg.norm([args.vx, args.vy]) > 1.0e-4
            else 0.0
        )
        reference_velocity_scale = gait_frequency / reference_nominal_frequency
    mirror_source, mirror_factor = training_mirror_map(
        contract.joint_order, TRAIN_TO_SERVER_SIGN
    )
    if contract.action_scale is None and args.action_scale is None:
        raise ValueError("run policy contract must declare action_scale")
    action_scale = (
        contract.action_scale if args.action_scale is None else args.action_scale
    )
    if contract.observation_size != actor_size:
        raise ValueError(
            f"contract observation size {contract.observation_size} does not match "
            f"ONNX input {actor_size}"
        )

    prefix = "accept_"
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    joint_dof = np.array(
        [model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
    )
    pos_actuator = np.array(
        [model.actuator(prefix + name + "_pos").id for name in contract.effector_order]
    )
    vel_actuator = np.array(
        [model.actuator(prefix + name + "_vel").id for name in contract.effector_order]
    )
    root_joint = model.joint(prefix + "root")
    root_qpos = root_joint.qposadr[0]
    root_dof = root_joint.dofadr[0]
    torso_body = model.body(prefix + "torso").id
    torso_site = model.site(prefix + "torso").id
    gyro = model.sensor(prefix + "torso_gyro")
    gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
    pitch_geom = model.geom("pitch").id
    left_foot_geom = model.geom(prefix + "left_foot").id
    right_foot_geom = model.geom(prefix + "right_foot").id
    pitch_height = float(model.geom_pos[pitch_geom, 2])
    ball_joint = model.joint("ball-root")

    lowers = np.array(
        [model.joint(prefix + name).range[0] for name in contract.joint_order]
    )
    uppers = np.array(
        [model.joint(prefix + name).range[1] for name in contract.joint_order]
    )
    nominal = NOMINAL_TRAINING_POSE.astype(np.float64)
    sign = TRAIN_TO_SERVER_SIGN.astype(np.float64)
    nominal_physical = np.clip(nominal * sign, lowers, uppers)
    if contract.kp is None or contract.kd is None:
        raise ValueError("run policy contract must declare PD gains")
    model.actuator_gainprm[pos_actuator, 0] = contract.kp
    model.actuator_biasprm[pos_actuator, 1] = -contract.kp
    model.actuator_gainprm[vel_actuator, 0] = contract.kd
    model.actuator_biasprm[vel_actuator, 2] = -contract.kd

    rng = np.random.default_rng(args.seed)
    command = np.array([args.vx, args.vy, args.yaw_rate], dtype=np.float32)
    completed: list[bool] = []
    mean_forward_speeds: list[float] = []
    mean_lateral_speeds: list[float] = []
    mean_yaw_rates: list[float] = []
    forward_rmses: list[float] = []
    lateral_rmses: list[float] = []
    yaw_rate_rmses: list[float] = []
    planar_velocity_rmses: list[float] = []
    lateral_drifts: list[float] = []
    flight_phase: list[bool] = []
    flight_phase_anytime: list[bool] = []
    max_flight_steps: list[int] = []
    survived_control_steps: list[int] = []
    invalid_episodes = 0
    proxy_true_positive = 0
    proxy_true_negative = 0
    proxy_false_positive = 0
    proxy_false_negative = 0
    proxy_flight_frames = 0
    actual_flight_frames = 0
    episode_steps = round(10.0 / 0.02)
    warmup_steps = round(2.0 / 0.02)

    for _ in range(args.episodes):
        mujoco.mj_resetData(model, data)
        gait_phase = float(rng.uniform())
        joint_noise_limit = 0.005 if reference_centered else 0.02
        root_velocity_noise_limit = 0.01 if reference_centered else 0.03
        yaw_limit = 0.02 if reference_centered else 0.05
        joint_noise = rng.uniform(
            -joint_noise_limit, joint_noise_limit, contract.action_size
        )
        yaw = rng.uniform(-yaw_limit, yaw_limit)
        if reference_centered:
            assert reference is not None
            reference_position_physical = circular_interpolate(
                reference["joint_position"], gait_phase
            )
            reference_velocity_physical = (
                reference_velocity_scale
                * circular_interpolate(reference["joint_velocity"], gait_phase)
            )
            reference_root_position = circular_interpolate(
                reference["root_position"], gait_phase
            )
            reference_root_quaternion = _reference_quaternion(
                reference["root_quaternion_xyzw"][:, [3, 0, 1, 2]], gait_phase
            )
            yaw_cos, yaw_sin = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
            reference_w, reference_x, reference_y, reference_z = (
                reference_root_quaternion
            )
            data.qpos[root_qpos + 2] = reference_root_position[2]
            data.qpos[root_qpos + 3 : root_qpos + 7] = [
                yaw_cos * reference_w - yaw_sin * reference_z,
                yaw_cos * reference_x - yaw_sin * reference_y,
                yaw_cos * reference_y + yaw_sin * reference_x,
                yaw_cos * reference_z + yaw_sin * reference_w,
            ]
            root_linear_velocity = reference_velocity_scale * circular_interpolate(
                reference["root_linear_velocity"], gait_phase
            )
            root_angular_velocity = reference_velocity_scale * circular_interpolate(
                reference["root_angular_velocity"], gait_phase
            )
            data.qvel[root_dof : root_dof + 3] = _yaw_rotate(root_linear_velocity, yaw)
            data.qvel[root_dof + 3 : root_dof + 6] = _yaw_rotate(
                root_angular_velocity, yaw
            )
            data.qvel[joint_dof] = reference_velocity_physical
            initial_physical = reference_position_physical
        else:
            data.qpos[root_qpos + 3 : root_qpos + 7] = [
                np.cos(0.5 * yaw),
                0.0,
                0.0,
                np.sin(0.5 * yaw),
            ]
            initial_physical = nominal_physical
        data.qpos[joint_qpos] = np.clip(initial_physical + joint_noise, lowers, uppers)
        data.qvel[root_dof : root_dof + 6] += rng.uniform(
            -root_velocity_noise_limit, root_velocity_noise_limit, 6
        )
        data.qpos[ball_joint.qposadr[0] : ball_joint.qposadr[0] + 3] = [
            10.0,
            8.0,
            0.11,
        ]
        data.ctrl[pos_actuator] = initial_physical
        mujoco.mj_forward(model, data)

        initial_xy = data.qpos[root_qpos : root_qpos + 2].copy()
        previous_action = np.zeros(contract.action_size, dtype=np.float32)
        local_velocity_samples: list[np.ndarray] = []
        yaw_rate_samples: list[float] = []
        airborne_substeps: list[bool] = []
        fell = False
        invalid = False

        for step in range(episode_steps):
            joint_position_training = data.qpos[joint_qpos] * sign
            joint_velocity_training = data.qvel[joint_dof] * sign
            if reference_centered:
                assert reference is not None
                reference_position_training = (
                    circular_interpolate(reference["joint_position"], gait_phase) * sign
                )
                reference_velocity_training = (
                    reference_velocity_scale
                    * circular_interpolate(reference["joint_velocity"], gait_phase)
                    * sign
                )
            else:
                reference_position_training = nominal
                reference_velocity_training = np.zeros(contract.action_size)
            joint_triplets = np.stack(
                [
                    (joint_position_training - reference_position_training) / 4.6,
                    (joint_velocity_training - reference_velocity_training) / 110.0,
                    previous_action / 10.0,
                ],
                axis=1,
            ).reshape(-1)
            projected_gravity = data.site_xmat[torso_site].reshape(3, 3).T @ np.array(
                [0.0, 0.0, -1.0]
            )
            observation = np.concatenate(
                [
                    joint_triplets,
                    data.sensordata[gyro_slice] / 50.0,
                    command,
                    projected_gravity,
                ]
            ).astype(np.float32)
            if actor_size == 80:
                moving = float(gait_frequency > 1.0e-8)
                angle = 2.0 * np.pi * gait_phase
                observation = np.concatenate(
                    [
                        observation,
                        moving
                        * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32),
                    ]
                )
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
            observation = np.clip(observation, -10.0, 10.0)
            action = session.run(None, {input_meta.name: observation[None]})[0][0]
            if args.symmetry_ensemble or args.mirror_policy:
                mirrored_observation = mirror_run_observation(
                    observation, mirror_source, mirror_factor
                )
                mirrored_action = session.run(
                    None, {input_meta.name: mirrored_observation[None]}
                )[0][0]
                reflected_action = mirror_run_action(
                    mirrored_action, mirror_source, mirror_factor
                )
                action = (
                    0.5 * (action + reflected_action)
                    if args.symmetry_ensemble
                    else reflected_action
                )
            action = np.clip(
                np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0),
                contract.action_clip[0],
                contract.action_clip[1],
            )
            targets = np.clip(
                (reference_position_training + action_scale * action) * sign,
                lowers,
                uppers,
            )
            data.ctrl[pos_actuator] = targets
            previous_action = action.astype(np.float32)
            gait_phase = (gait_phase + 0.02 * gait_frequency) % 1.0

            for _ in range(4):
                mujoco.mj_step(model, data)
                left_contact = _has_contact(data, pitch_geom, left_foot_geom)
                right_contact = _has_contact(data, pitch_geom, right_foot_geom)
                airborne_substeps.append(not left_contact and not right_contact)
                proxy_left = _box_geometric_contact(
                    data,
                    model,
                    left_foot_geom,
                    pitch_height=pitch_height,
                    tolerance=args.contact_proxy_tolerance,
                )
                proxy_right = _box_geometric_contact(
                    data,
                    model,
                    right_foot_geom,
                    pitch_height=pitch_height,
                    tolerance=args.contact_proxy_tolerance,
                )
                for actual, proxy in (
                    (left_contact, proxy_left),
                    (right_contact, proxy_right),
                ):
                    proxy_true_positive += int(actual and proxy)
                    proxy_true_negative += int(not actual and not proxy)
                    proxy_false_positive += int(not actual and proxy)
                    proxy_false_negative += int(actual and not proxy)
                proxy_flight_frames += int(not proxy_left and not proxy_right)
                actual_flight_frames += int(not left_contact and not right_contact)

            torso_xmat = data.site_xmat[torso_site].reshape(3, 3)
            body_yaw = np.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
            c, s = np.cos(body_yaw), np.sin(body_yaw)
            world_vx, world_vy = data.qvel[root_dof : root_dof + 2]
            local_velocity_samples.append(
                np.array(
                    [c * world_vx + s * world_vy, -s * world_vx + c * world_vy],
                    dtype=np.float64,
                )
            )
            yaw_rate_samples.append(float(data.sensordata[gyro_slice][2]))
            upright = torso_xmat[2, 2]
            height = data.xpos[torso_body, 2]
            invalid = not (
                np.isfinite(data.qpos).all()
                and np.isfinite(data.qvel).all()
                and np.isfinite(action).all()
            )
            fell = height < 0.35 or upright < 0.20
            if invalid or fell:
                break

        local_velocities = np.asarray(local_velocity_samples, dtype=np.float64)
        yaw_rates = np.asarray(yaw_rate_samples, dtype=np.float64)
        warmup_index = min(warmup_steps, local_velocities.shape[0])
        after_warmup = local_velocities[warmup_index:]
        yaw_after_warmup = yaw_rates[warmup_index:]
        if after_warmup.shape[0] == 0:
            after_warmup = local_velocities
            yaw_after_warmup = yaw_rates
        velocity_error = after_warmup - command[None, :2]
        yaw_error = yaw_after_warmup - command[2]
        delta_world = data.qpos[root_qpos : root_qpos + 2] - initial_xy
        c0, s0 = np.cos(yaw), np.sin(yaw)
        lateral = -s0 * delta_world[0] + c0 * delta_world[1]
        airborne_array = np.asarray(airborne_substeps[warmup_steps * 4 :], dtype=bool)
        longest_flight = _max_consecutive(airborne_array)
        longest_flight_anytime = _max_consecutive(np.asarray(airborne_substeps))

        completed.append(
            not fell and not invalid and len(local_velocity_samples) == episode_steps
        )
        mean_forward_speeds.append(float(np.mean(after_warmup[:, 0])))
        mean_lateral_speeds.append(float(np.mean(after_warmup[:, 1])))
        mean_yaw_rates.append(float(np.mean(yaw_after_warmup)))
        forward_rmses.append(
            float(np.sqrt(np.mean(np.square(velocity_error[:, 0]))))
        )
        lateral_rmses.append(
            float(np.sqrt(np.mean(np.square(velocity_error[:, 1]))))
        )
        yaw_rate_rmses.append(float(np.sqrt(np.mean(np.square(yaw_error)))))
        planar_velocity_rmses.append(
            float(np.sqrt(np.mean(np.sum(np.square(velocity_error), axis=1))))
        )
        lateral_drifts.append(abs(float(lateral)))
        # Two consecutive 5 ms physics frames reject one-frame contact noise.
        flight_phase.append(longest_flight >= 2)
        flight_phase_anytime.append(longest_flight_anytime >= 2)
        max_flight_steps.append(longest_flight)
        survived_control_steps.append(len(local_velocity_samples))
        invalid_episodes += int(invalid)

    speeds = np.asarray(mean_forward_speeds)
    lateral_speeds = np.asarray(mean_lateral_speeds)
    yaw_rates = np.asarray(mean_yaw_rates)
    rmses = np.asarray(forward_rmses)
    lateral_error_rmses = np.asarray(lateral_rmses)
    yaw_error_rmses = np.asarray(yaw_rate_rmses)
    planar_error_rmses = np.asarray(planar_velocity_rmses)
    drifts = np.asarray(lateral_drifts)
    completion_rate = float(np.mean(completed))
    flight_rate = float(np.mean(flight_phase))
    payload = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "contract": str(contract_path.resolve()),
        "engine": "MuJoCo CPU",
        "episodes": args.episodes,
        "seed": args.seed,
        "command": command.tolist(),
        "action_scale": action_scale,
        "action_scale_source": (
            "policy_contract" if args.action_scale is None else "cli_override"
        ),
        "pd_gains": {"kp": contract.kp, "kd": contract.kd},
        "gait_frequency_hz": (gait_frequency if actor_size == 80 else None),
        "gait_frequency_source": (
            "speed_scaled_motion_reference" if reference_centered else "cli"
        ),
        "motion_reference": reference_validation,
        "reference_nominal_frequency_hz": reference_nominal_frequency,
        "reference_forward_speed_m_s": reference_forward_speed,
        "reference_velocity_scale": (
            reference_velocity_scale if reference_centered else None
        ),
        "symmetry_ensemble": args.symmetry_ensemble,
        "mirror_policy": args.mirror_policy,
        "duration_seconds": 10.0,
        "warmup_seconds": 2.0,
        "upright_completion_rate": completion_rate,
        "invalid_episode_count": invalid_episodes,
        "forward_speed": {
            "median_m_s": _percentile(speeds, 50),
            "p10_m_s": _percentile(speeds, 10),
            "p90_m_s": _percentile(speeds, 90),
        },
        "lateral_speed": {
            "median_m_s": _percentile(lateral_speeds, 50),
            "p10_m_s": _percentile(lateral_speeds, 10),
            "p90_m_s": _percentile(lateral_speeds, 90),
        },
        "yaw_rate": {
            "median_rad_s": _percentile(yaw_rates, 50),
            "p10_rad_s": _percentile(yaw_rates, 10),
            "p90_rad_s": _percentile(yaw_rates, 90),
        },
        "forward_tracking_rmse": {
            "median_m_s": _percentile(rmses, 50),
            "p90_m_s": _percentile(rmses, 90),
        },
        "lateral_tracking_rmse": {
            "median_m_s": _percentile(lateral_error_rmses, 50),
            "p90_m_s": _percentile(lateral_error_rmses, 90),
        },
        "planar_velocity_tracking_rmse": {
            "median_m_s": _percentile(planar_error_rmses, 50),
            "p90_m_s": _percentile(planar_error_rmses, 90),
        },
        "yaw_rate_tracking_rmse": {
            "median_rad_s": _percentile(yaw_error_rmses, 50),
            "p90_rad_s": _percentile(yaw_error_rmses, 90),
        },
        "absolute_lateral_drift": {
            "median_m": _percentile(drifts, 50),
            "p90_m": _percentile(drifts, 90),
        },
        "flight_phase_episode_rate": flight_rate,
        "flight_phase_anytime_episode_rate": float(np.mean(flight_phase_anytime)),
        "survival": {
            "median_control_steps": _percentile(np.asarray(survived_control_steps), 50),
            "p10_control_steps": _percentile(np.asarray(survived_control_steps), 10),
            "p90_control_steps": _percentile(np.asarray(survived_control_steps), 90),
            "maximum_control_steps": int(max(survived_control_steps)),
            "median_seconds": 0.02
            * _percentile(np.asarray(survived_control_steps), 50),
        },
        "max_flight_duration": {
            "median_seconds": _percentile(np.asarray(max_flight_steps) * 0.005, 50),
            "p90_seconds": _percentile(np.asarray(max_flight_steps) * 0.005, 90),
        },
        "training_contact_proxy": {
            "method": "oriented_box_lowest_point_lte_pitch_plus_tolerance",
            "tolerance_m": args.contact_proxy_tolerance,
            "true_positive_frames": proxy_true_positive,
            "true_negative_frames": proxy_true_negative,
            "false_positive_frames": proxy_false_positive,
            "false_negative_frames": proxy_false_negative,
            "proxy_flight_frames": proxy_flight_frames,
            "actual_flight_frames": actual_flight_frames,
        },
    }
    gates = {
        "upright_completion_rate_gte_0_95": completion_rate >= 0.95,
        "median_forward_speed_gte_1_2": payload["forward_speed"]["median_m_s"] >= 1.2,
        "median_forward_rmse_lte_0_35": payload["forward_tracking_rmse"]["median_m_s"]
        <= 0.35,
        "median_lateral_drift_lte_0_25": payload["absolute_lateral_drift"]["median_m"]
        <= 0.25,
        "flight_phase_episode_rate_gte_0_80": flight_rate >= 0.80,
        "all_values_finite": invalid_episodes == 0,
    }
    payload["gates"] = gates
    payload["candidate_gate_passed"] = all(gates.values())
    soccer_command_gates = {
        "upright_completion_rate_gte_0_95": completion_rate >= 0.95,
        "median_planar_velocity_rmse_lte_0_45": payload[
            "planar_velocity_tracking_rmse"
        ]["median_m_s"] <= 0.45,
        "median_yaw_rate_rmse_lte_0_35": payload["yaw_rate_tracking_rmse"][
            "median_rad_s"
        ] <= 0.35,
        "all_values_finite": invalid_episodes == 0,
    }
    payload["soccer_command_gates"] = soccer_command_gates
    payload["soccer_command_gate_passed"] = all(soccer_command_gates.values())

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
