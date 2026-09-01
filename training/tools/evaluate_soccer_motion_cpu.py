#!/usr/bin/env python3
"""Exact CPU MuJoCo evaluation on a fixed motion-by-phase grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable

from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.reference_dynamics import configure_pd_actuators
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.t1_control import apollo_joint_gains


REPOSITORY_ROOT = Path(__file__).parents[2]


def _start_frames(length: int, samples: int, minimum_remaining: int) -> list[int]:
    final_start = max(0, length - minimum_remaining)
    return sorted(
        set(
            int(value)
            for value in np.linspace(0, final_start, samples, dtype=np.int64)
            if value < length - 1
        )
    )


def _foot_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pitch_geom: int,
    foot_geoms: tuple[int, int],
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


def _actor_observation(
    data: mujoco.MjData,
    *,
    joint_qpos: np.ndarray,
    joint_dof: np.ndarray,
    gyro_slice: slice,
    torso_site: int,
    reference_joint_position: np.ndarray,
    reference_joint_velocity: np.ndarray,
    reference_root_linear_velocity: np.ndarray,
    reference_root_angular_velocity: np.ndarray,
    reference_contact: np.ndarray,
    previous_action: np.ndarray,
    progress: float,
    kick_leg_one_hot: np.ndarray,
) -> np.ndarray:
    triplets = np.stack(
        [
            (data.qpos[joint_qpos] - reference_joint_position) / 4.6,
            (data.qvel[joint_dof] - reference_joint_velocity) / 110.0,
            previous_action / 10.0,
        ],
        axis=1,
    ).reshape(-1)
    angular_velocity = data.sensordata[gyro_slice] / 50.0
    gravity = data.site_xmat[torso_site].reshape(3, 3).T @ np.array(
        [0.0, 0.0, -1.0]
    )
    angle = 2.0 * np.pi * progress
    actor = np.concatenate(
        [
            triplets,
            angular_velocity,
            gravity,
            reference_joint_position / 4.6,
            reference_root_linear_velocity / 5.0,
            reference_root_angular_velocity / 10.0,
            reference_contact.astype(np.float64),
            np.array([np.cos(angle), np.sin(angle)]),
            kick_leg_one_hot,
        ]
    )
    if actor.shape != (110,):
        raise ValueError(f"CPU actor observation shape {actor.shape} != (110,)")
    return np.clip(np.nan_to_num(actor), -10.0, 10.0).astype(np.float32)


def _make_policy(
    *,
    zero_policy: bool,
    checkpoint: Path | None,
    profile_name: str,
    observation_size: int,
    action_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    if zero_policy:
        return lambda unused: np.zeros(action_size, dtype=np.float32)
    profile = get_ppo_profile(profile_name)
    preprocess = (
        running_statistics.normalize
        if profile.normalize_observations
        else brax_types.identity_observation_preprocessor
    )
    networks = profile.network_factory()(
        {"state": observation_size, "privileged_state": 118},
        action_size,
        preprocess_observations_fn=preprocess,
    )
    params = ppo_checkpoint.load(checkpoint)
    inference = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    @jax.jit
    def infer(actor: jax.Array) -> jax.Array:
        return inference(
            {"state": actor}, jax.random.PRNGKey(0)
        )[0]

    return lambda actor: np.asarray(infer(jp.asarray(actor)), dtype=np.float64)


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
    parser.add_argument("--phase-samples", type=int, default=8)
    parser.add_argument("--minimum-remaining-frames", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.zero_policy == (args.checkpoint is not None):
        raise ValueError("select exactly one of --zero-policy or --checkpoint")
    if args.phase_samples < 1 or args.minimum_remaining_frames < 2:
        raise ValueError("invalid fixed phase grid")
    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    corpus = load_soccer_motion_corpus(args.corpus_root)
    policy = _make_policy(
        zero_policy=args.zero_policy,
        checkpoint=args.checkpoint,
        profile_name=args.profile,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )

    prefix = "soccer_cpu_"
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
    model_root_xy = model.qpos0[root_qpos : root_qpos + 2].copy()
    records: list[dict[str, Any]] = []

    for motion, relative_path in enumerate(corpus.relative_paths):
        length = int(corpus.lengths[motion])
        for start in _start_frames(
            length, args.phase_samples, args.minimum_remaining_frames
        ):
            data = mujoco.MjData(model)
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0
            data.qpos[root_qpos : root_qpos + 2] = model_root_xy
            data.qpos[root_qpos + 2] = corpus.root_position[motion, start, 2]
            data.qpos[root_qpos + 3 : root_qpos + 7] = (
                corpus.root_quaternion_wxyz[motion, start]
            )
            data.qpos[joint_qpos] = corpus.joint_position[motion, start]
            data.qvel[root_dof : root_dof + 3] = (
                corpus.root_linear_velocity[motion, start]
            )
            data.qvel[root_dof + 3 : root_dof + 6] = (
                corpus.root_angular_velocity[motion, start]
            )
            data.qvel[joint_dof] = corpus.joint_velocity[motion, start]
            data.ctrl[tau_actuator] = 0.0
            data.ctrl[vel_actuator] = 0.0
            data.ctrl[pos_actuator] = corpus.joint_position[motion, start]
            mujoco.mj_forward(model, data)
            previous_action = np.zeros(contract.action_size, dtype=np.float64)
            squared_joint_error: list[float] = []
            contact_match = 0
            contact_count = 0
            mean_action: list[float] = []
            maximum_action = 0.0
            terminal_frame: int | None = None
            reason = "completed"

            for frame in range(start + 1, length):
                current = frame - 1
                observation = _actor_observation(
                    data,
                    joint_qpos=joint_qpos,
                    joint_dof=joint_dof,
                    gyro_slice=gyro_slice,
                    torso_site=torso_site,
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
                raw_action = policy(observation)
                action = np.clip(raw_action, *contract.action_clip)
                target = np.clip(
                    corpus.joint_position[motion, frame]
                    + contract.action_scale * action,
                    lower,
                    upper,
                )
                data.ctrl[pos_actuator] = target
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                previous_action = action
                mean_action.append(float(np.mean(np.abs(action))))
                maximum_action = max(maximum_action, float(np.max(np.abs(action))))
                error = data.qpos[joint_qpos] - corpus.joint_position[motion, frame]
                squared_joint_error.extend(np.square(error).tolist())
                actual_contact = _foot_contacts(
                    model, data, pitch_geom, foot_geoms
                )
                contact_match += int(
                    np.sum(actual_contact == corpus.foot_contact[motion, frame])
                )
                contact_count += 2
                rotation = data.site_xmat[torso_site].reshape(3, 3)
                torso_height = float(data.xpos[torso_body, 2])
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    terminal_frame = frame
                    reason = "non_finite_state"
                    break
                if torso_height < 0.35 or rotation[2, 2] < 0.20:
                    terminal_frame = frame
                    reason = "fall"
                    break
            completed = terminal_frame is None
            records.append(
                {
                    "motion": motion,
                    "relative_path": relative_path,
                    "start_frame": start,
                    "length": length,
                    "completed": completed,
                    "terminal_frame": terminal_frame,
                    "termination_reason": reason,
                    "survival_fraction": (
                        1.0
                        if completed
                        else (terminal_frame - start) / (length - 1 - start)
                    ),
                    "joint_tracking_rmse_rad": float(
                        np.sqrt(np.mean(squared_joint_error))
                    ),
                    "foot_contact_agreement": contact_match / max(contact_count, 1),
                    "mean_abs_action": float(np.mean(mean_action)),
                    "maximum_abs_action": maximum_action,
                }
            )

    per_motion: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["motion"]].append(record)
    for motion, items in sorted(grouped.items()):
        per_motion.append(
            {
                "motion": motion,
                "relative_path": corpus.relative_paths[motion],
                "episodes": len(items),
                "completion_rate": float(
                    np.mean([item["completed"] for item in items])
                ),
                "mean_survival_fraction": float(
                    np.mean([item["survival_fraction"] for item in items])
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "k1_exact_cpu_fixed_motion_phase_grid",
        "engine": f"MuJoCo {mujoco.__version__}",
        "policy": "zero_residual" if args.zero_policy else "checkpoint",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "profile": args.profile,
        "contract": str(args.contract.resolve()),
        "corpus_root": str(args.corpus_root.resolve()),
        "phase_samples": args.phase_samples,
        "minimum_remaining_frames": args.minimum_remaining_frames,
        "episode_count": len(records),
        "completion_rate": float(np.mean([item["completed"] for item in records])),
        "mean_survival_fraction": float(
            np.mean([item["survival_fraction"] for item in records])
        ),
        "mean_joint_tracking_rmse_rad": float(
            np.mean([item["joint_tracking_rmse_rad"] for item in records])
        ),
        "mean_foot_contact_agreement": float(
            np.mean([item["foot_contact_agreement"] for item in records])
        ),
        "mean_abs_action": float(np.mean([item["mean_abs_action"] for item in records])),
        "maximum_abs_action": float(
            np.max([item["maximum_abs_action"] for item in records])
        ),
        "per_motion": per_motion,
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
