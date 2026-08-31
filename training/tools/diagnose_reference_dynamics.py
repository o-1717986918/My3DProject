#!/usr/bin/env python3
"""Measure PD lag and inverse-dynamics feasibility of a running reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.motion_reference import sha256, validate_motion_reference
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.reference_dynamics import (
    circular_interpolate,
    compute_inverse_dynamics_reference,
    configure_pd_actuators,
)
from my3d_rl.run_env import TRAIN_TO_SERVER_SIGN


REPOSITORY_ROOT = Path(__file__).parents[2]


def _parse_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return result


def _percentiles(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "mean_abs": float(np.mean(absolute)),
        "p90_abs": float(np.percentile(absolute, 90)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def _reference_quaternion(reference: dict[str, np.ndarray], phase: float) -> np.ndarray:
    values = reference["root_quaternion_xyzw"][:, [3, 0, 1, 2]]
    frame = (phase % 1.0) * values.shape[0]
    lower = int(np.floor(frame)) % values.shape[0]
    upper = (lower + 1) % values.shape[0]
    fraction = frame - np.floor(frame)
    upper_value = values[upper]
    if np.dot(values[lower], upper_value) < 0.0:
        upper_value = -upper_value
    result = (1.0 - fraction) * values[lower] + fraction * upper_value
    return result / np.linalg.norm(result)


def _actual_foot_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pitch_geom: int,
    foot_geoms: tuple[int, int],
) -> tuple[bool, bool]:
    pairs = data.contact.geom[: data.ncon]
    return tuple(
        bool(
            np.any(
                ((pairs[:, 0] == pitch_geom) & (pairs[:, 1] == foot))
                | ((pairs[:, 0] == foot) & (pairs[:, 1] == pitch_geom))
            )
        )
        for foot in foot_geoms
    )


def _evaluate_targets(
    model: mujoco.MjModel,
    contract,
    reference: dict[str, np.ndarray],
    target_position_physical: np.ndarray,
    *,
    episodes: int,
    maximum_steps: int,
    speed_scale: float,
    phase_lead_frames: float,
    kp: float,
    kd: float,
    prefix: str,
) -> dict[str, Any]:
    tau_actuator, pos_actuator, vel_actuator = configure_pd_actuators(
        model, contract, kp=kp, kd=kd, prefix=prefix
    )
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    joint_dof = np.array(
        [model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
    )
    root = model.joint(prefix + "root")
    root_qpos = int(root.qposadr[0])
    root_dof = int(root.dofadr[0])
    torso_body = model.body(prefix + "torso").id
    torso_site = model.site(prefix + "torso").id
    pitch_geom = model.geom("pitch").id
    foot_geoms = (
        model.geom(prefix + "left_foot").id,
        model.geom(prefix + "right_foot").id,
    )
    frame_count = reference["joint_position"].shape[0]
    # Motion-reference NPZ files are written in exact RCSS/MuJoCo physical
    # coordinates.  DirectionalRun applies the sign table only when converting
    # them into its historical policy-training coordinate frame.
    physical_joint_position = reference["joint_position"]
    physical_joint_velocity = reference["joint_velocity"]
    reference_xy_origin = reference["root_position"][0, :2]
    model_xy_origin = model.qpos0[root_qpos : root_qpos + 2]
    episode_length: list[int] = []
    forward_velocity: list[float] = []
    joint_error: list[float] = []
    flight_by_episode: list[bool] = []

    for episode in range(episodes):
        phase = episode / episodes
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        root_position = circular_interpolate(reference["root_position"], phase)
        data.qpos[root_qpos : root_qpos + 2] = (
            model_xy_origin + root_position[:2] - reference_xy_origin
        )
        data.qpos[root_qpos + 2] = root_position[2]
        data.qpos[root_qpos + 3 : root_qpos + 7] = _reference_quaternion(
            reference, phase
        )
        data.qpos[joint_qpos] = circular_interpolate(physical_joint_position, phase)
        data.qvel[root_dof : root_dof + 3] = speed_scale * circular_interpolate(
            reference["root_linear_velocity"], phase
        )
        data.qvel[root_dof + 3 : root_dof + 6] = speed_scale * circular_interpolate(
            reference["root_angular_velocity"], phase
        )
        data.qvel[joint_dof] = speed_scale * circular_interpolate(
            physical_joint_velocity, phase
        )
        data.ctrl[tau_actuator] = 0.0
        data.ctrl[vel_actuator] = 0.0
        mujoco.mj_forward(model, data)
        had_flight = False
        alive_steps = maximum_steps

        for step in range(maximum_steps):
            target_phase = phase + phase_lead_frames / frame_count
            data.ctrl[pos_actuator] = circular_interpolate(
                target_position_physical, target_phase
            )
            for _ in range(round(0.02 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            phase = (phase + speed_scale / frame_count) % 1.0
            rotation = data.site_xmat[torso_site].reshape(3, 3)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            forward_velocity.append(
                float(
                    np.cos(yaw) * data.qvel[root_dof]
                    + np.sin(yaw) * data.qvel[root_dof + 1]
                )
            )
            expected = circular_interpolate(physical_joint_position, phase)
            joint_error.append(
                float(np.sqrt(np.mean(np.square(data.qpos[joint_qpos] - expected))))
            )
            contacts = _actual_foot_contacts(
                model, data, pitch_geom, foot_geoms
            )
            had_flight |= not any(contacts)
            upright = float(rotation[2, 2])
            if data.xpos[torso_body, 2] < 0.35 or upright < 0.20:
                alive_steps = step + 1
                break
        episode_length.append(alive_steps)
        flight_by_episode.append(had_flight)

    lengths = np.asarray(episode_length)
    return {
        "mean_episode_length_steps": float(np.mean(lengths)),
        "median_episode_length_steps": float(np.median(lengths)),
        "minimum_episode_length_steps": int(np.min(lengths)),
        "maximum_episode_length_steps": int(np.max(lengths)),
        "full_episode_rate": float(np.mean(lengths == maximum_steps)),
        "alive_weighted_forward_speed_m_s": float(np.mean(forward_velocity)),
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error)))),
        "actual_flight_episode_rate": float(np.mean(flight_by_episode)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v3.yaml",
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--kd", type=float, default=0.6)
    parser.add_argument("--smoothing-passes", type=int, default=2)
    parser.add_argument("--maximum-residual", type=float, default=0.15)
    parser.add_argument("--phase-leads", type=_parse_floats, default=[-2, -1, 0, 1, 2, 3])
    parser.add_argument("--inverse-blends", type=_parse_floats, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-target", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.speed_scale <= 0.0:
        raise ValueError("episodes, steps and speed scale must be positive")

    validation = validate_motion_reference(args.reference)
    if not validation["passed"]:
        raise ValueError("motion reference failed validation")
    contract = load_policy_contract(args.contract)
    reference_sha = sha256(args.reference)
    if contract.reference_sha256 and contract.reference_sha256 != reference_sha:
        raise ValueError("motion reference SHA-256 differs from policy contract")
    with np.load(args.reference, allow_pickle=False) as archive:
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

    model = build_single_t1_soccer_model(prefix="train_", robot_x=-10.0)
    model.opt.timestep = 0.005
    inverse = compute_inverse_dynamics_reference(
        model,
        contract,
        root_position=reference["root_position"],
        root_quaternion_xyzw=reference["root_quaternion_xyzw"],
        root_linear_velocity=reference["root_linear_velocity"],
        root_angular_velocity=reference["root_angular_velocity"],
        joint_position_physical=reference["joint_position"],
        joint_velocity_physical=reference["joint_velocity"],
        frequency_hz=50.0,
        kp=args.kp,
        kd=args.kd,
        smoothing_passes=args.smoothing_passes,
        maximum_residual_rad=args.maximum_residual,
    )
    physical_reference = reference["joint_position"]
    experiments: dict[str, Any] = {}
    for lead in args.phase_leads:
        experiments[f"reference_phase_lead_{lead:g}_frames"] = _evaluate_targets(
            model,
            contract,
            reference,
            physical_reference,
            episodes=args.episodes,
            maximum_steps=args.steps,
            speed_scale=args.speed_scale,
            phase_lead_frames=lead,
            kp=args.kp,
            kd=args.kd,
            prefix="train_",
        )
    for blend in args.inverse_blends:
        if not 0.0 <= blend <= 1.0:
            raise ValueError("inverse blends must lie in [0, 1]")
        targets = physical_reference + blend * inverse.joint_target_residual
        experiments[f"inverse_dynamics_blend_{blend:g}"] = _evaluate_targets(
            model,
            contract,
            reference,
            targets,
            episodes=args.episodes,
            maximum_steps=args.steps,
            speed_scale=args.speed_scale,
            phase_lead_frames=0.0,
            kp=args.kp,
            kd=args.kd,
            prefix="train_",
        )

    payload = {
        "schema_version": 1,
        "purpose": "reference_pd_lag_and_inverse_dynamics_feasibility",
        "reference": str(args.reference.resolve()),
        "reference_sha256": reference_sha,
        "contract": str(args.contract.resolve()),
        "cpu_engine": f"MuJoCo {mujoco.__version__}",
        "episodes": args.episodes,
        "maximum_steps": args.steps,
        "speed_scale": args.speed_scale,
        "kp": args.kp,
        "kd": args.kd,
        "inverse_dynamics": {
            "smoothing_passes": args.smoothing_passes,
            "maximum_residual_rad": args.maximum_residual,
            "joint_torque_nm": _percentiles(inverse.joint_torque),
            "joint_target_residual_rad": _percentiles(
                inverse.joint_target_residual
            ),
            "residual_saturation_rate": float(
                np.mean(
                    np.isclose(
                        np.abs(inverse.joint_target_residual),
                        args.maximum_residual,
                        atol=1.0e-6,
                    )
                )
            ),
            "floating_base_force": _percentiles(
                inverse.root_generalized_force[:, :3]
            ),
            "floating_base_torque": _percentiles(
                inverse.root_generalized_force[:, 3:]
            ),
        },
        "experiments": experiments,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.output_target:
        args.output_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_target,
            joint_target_position=(
                inverse.joint_target_position * TRAIN_TO_SERVER_SIGN
            ).astype(np.float32),
            joint_target_residual=(
                inverse.joint_target_residual * TRAIN_TO_SERVER_SIGN
            ).astype(np.float32),
            joint_inverse_torque=(
                inverse.joint_torque * TRAIN_TO_SERVER_SIGN
            ).astype(np.float32),
            root_generalized_force=inverse.root_generalized_force.astype(np.float32),
            reference_sha256=np.array(reference_sha),
            metadata_json=np.array(
                json.dumps(
                    {
                        "schema_version": 1,
                        "purpose": "local_only_inverse_dynamics_target_diagnostic",
                        "source_reference": str(args.reference.resolve()),
                        "source_reference_sha256": reference_sha,
                        "kp": args.kp,
                        "kd": args.kd,
                        "maximum_residual_rad": args.maximum_residual,
                        "smoothing_passes": args.smoothing_passes,
                    },
                    sort_keys=True,
                )
            ),
        )
    print(rendered, end="")


if __name__ == "__main__":
    main()
