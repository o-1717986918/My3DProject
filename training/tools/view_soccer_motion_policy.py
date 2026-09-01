#!/usr/bin/env python3
"""Replay one finite soccer-motion policy rollout in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.reference_dynamics import configure_pd_actuators
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.soccer_motion_policy import (
    load_soccer_motion_policy,
    soccer_motion_actor_observation,
)
from my3d_rl.t1_control import apollo_joint_gains


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "training/contracts/soccer_motion_policy_v2.yaml"
)


def select_motion(
    relative_paths: tuple[str, ...], *, motion_index: int, motion_path: str | None
) -> int:
    if motion_path is not None:
        try:
            return relative_paths.index(motion_path)
        except ValueError as exc:
            raise ValueError(
                f"motion path is unavailable: {motion_path}; "
                f"choose one of {list(relative_paths)}"
            ) from exc
    if not 0 <= motion_index < len(relative_paths):
        raise ValueError(
            f"motion index {motion_index} is outside [0, {len(relative_paths) - 1}]"
        )
    return motion_index


def _foot_contacts(
    data: mujoco.MjData, pitch_geom: int, foot_geoms: tuple[int, int]
) -> np.ndarray:
    pairs = np.asarray(data.contact.geom[: data.ncon], dtype=np.int32)
    return np.asarray(
        [
            np.any(
                ((pairs[:, 0] == pitch_geom) & (pairs[:, 1] == foot))
                | ((pairs[:, 1] == pitch_geom) & (pairs[:, 0] == foot))
            )
            for foot in foot_geoms
        ],
        dtype=bool,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--zero-policy", action="store_true")
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-index", type=int, default=0)
    motion.add_argument("--motion-path")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0.0 or args.loops < 1 or args.start_frame < 0:
        raise ValueError("speed/loops/start-frame settings are invalid")

    contract = load_policy_contract(args.contract)
    corpus = load_soccer_motion_corpus(args.corpus_root)
    motion_index = select_motion(
        corpus.relative_paths,
        motion_index=args.motion_index,
        motion_path=args.motion_path,
    )
    length = int(corpus.lengths[motion_index])
    if args.start_frame >= length - 1:
        raise ValueError(
            f"start frame {args.start_frame} must be below {length - 1}"
        )
    policy = load_soccer_motion_policy(
        zero_policy=args.zero_policy,
        checkpoint=args.checkpoint,
        profile_name=args.profile,
        policy_contract_name=contract.policy_name,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )

    prefix = "soccer_policy_"
    model = build_single_t1_soccer_model(prefix=prefix, robot_x=-10.0, robot_y=0.0)
    model.opt.timestep = 0.005
    gains = np.asarray(
        [apollo_joint_gains(name) for name in contract.joint_order],
        dtype=np.float64,
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
    gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
    pitch_geom = model.geom("pitch").id
    foot_geoms = (
        model.geom(prefix + "left_foot").id,
        model.geom(prefix + "right_foot").id,
    )
    lower = model.jnt_range[model.dof_jntid[joint_dof], 0]
    upper = model.jnt_range[model.dof_jntid[joint_dof], 1]
    substeps = round((1.0 / contract.frequency_hz) / model.opt.timestep)
    frame_period = (1.0 / contract.frequency_hz) / args.speed
    model_root_xy = model.qpos0[root_qpos : root_qpos + 2].copy()

    completed_loops = 0
    termination_reason = "viewer_closed"
    terminal_frame: int | None = None
    contact_matches = 0
    contact_count = 0
    squared_joint_error: list[float] = []
    print(
        json.dumps(
            {
                "policy": (
                    "zero_residual"
                    if args.zero_policy
                    else str(args.checkpoint.resolve())
                ),
                "motion": corpus.relative_paths[motion_index],
                "start_frame": args.start_frame,
                "frames": length - args.start_frame,
                "speed": args.speed,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for unused_loop in range(args.loops):
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0
            data.qpos[root_qpos : root_qpos + 2] = model_root_xy
            data.qpos[root_qpos + 2] = corpus.root_position[
                motion_index, args.start_frame, 2
            ]
            data.qpos[root_qpos + 3 : root_qpos + 7] = (
                corpus.root_quaternion_wxyz[motion_index, args.start_frame]
            )
            data.qpos[joint_qpos] = corpus.joint_position[
                motion_index, args.start_frame
            ]
            data.qvel[root_dof : root_dof + 3] = corpus.root_linear_velocity[
                motion_index, args.start_frame
            ]
            data.qvel[root_dof + 3 : root_dof + 6] = (
                corpus.root_angular_velocity[motion_index, args.start_frame]
            )
            data.qvel[joint_dof] = corpus.joint_velocity[
                motion_index, args.start_frame
            ]
            data.ctrl[tau_actuator] = 0.0
            data.ctrl[vel_actuator] = 0.0
            data.ctrl[pos_actuator] = corpus.joint_position[
                motion_index, args.start_frame
            ]
            mujoco.mj_forward(model, data)
            previous_action = np.zeros(contract.action_size, dtype=np.float64)
            terminal_frame = None
            termination_reason = "completed"
            for frame in range(args.start_frame + 1, length):
                if not viewer.is_running():
                    termination_reason = "viewer_closed"
                    break
                started = time.monotonic()
                current = frame - 1
                observation = soccer_motion_actor_observation(
                    data,
                    joint_qpos=joint_qpos,
                    joint_dof=joint_dof,
                    gyro_slice=gyro_slice,
                    torso_site=torso_site,
                    reference_joint_position=corpus.joint_position[
                        motion_index, current
                    ],
                    reference_joint_velocity=corpus.joint_velocity[
                        motion_index, current
                    ],
                    reference_root_linear_velocity=corpus.root_linear_velocity[
                        motion_index, current
                    ],
                    reference_root_angular_velocity=corpus.root_angular_velocity[
                        motion_index, current
                    ],
                    reference_contact=corpus.foot_contact[motion_index, current],
                    previous_action=previous_action,
                    progress=current / max(length - 1, 1),
                    kick_leg_one_hot=corpus.kick_leg_one_hot[motion_index],
                )
                action = np.clip(policy(observation), *contract.action_clip)
                target = np.clip(
                    corpus.joint_position[motion_index, frame]
                    + contract.action_scale * action,
                    lower,
                    upper,
                )
                data.ctrl[pos_actuator] = target
                for unused_substep in range(substeps):
                    mujoco.mj_step(model, data)
                previous_action = action
                error = (
                    data.qpos[joint_qpos]
                    - corpus.joint_position[motion_index, frame]
                )
                squared_joint_error.extend(np.square(error).tolist())
                actual_contact = _foot_contacts(data, pitch_geom, foot_geoms)
                contact_matches += int(
                    np.sum(actual_contact == corpus.foot_contact[motion_index, frame])
                )
                contact_count += 2
                rotation = data.site_xmat[torso_site].reshape(3, 3)
                torso_height = float(data.xpos[torso_body, 2])
                viewer.sync()
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    terminal_frame = frame
                    termination_reason = "non_finite_state"
                    break
                if torso_height < 0.35 or rotation[2, 2] < 0.20:
                    terminal_frame = frame
                    termination_reason = "fall"
                    break
                remaining = frame_period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
            if termination_reason == "viewer_closed":
                break
            completed_loops += int(terminal_frame is None)
        print(
            json.dumps(
                {
                    "completed_loops": completed_loops,
                    "requested_loops": args.loops,
                    "termination_reason": termination_reason,
                    "terminal_frame": terminal_frame,
                    "joint_tracking_rmse_rad": (
                        float(np.sqrt(np.mean(squared_joint_error)))
                        if squared_joint_error
                        else None
                    ),
                    "foot_contact_agreement": (
                        contact_matches / contact_count if contact_count else None
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while args.hold and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
