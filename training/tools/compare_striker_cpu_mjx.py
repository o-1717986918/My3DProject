#!/usr/bin/env python3
"""Replay identical striker controls in CPU MuJoCo and MJX/Warp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.sim_parity import generate_action_sequence, quaternion_angle_error
from my3d_rl.striker_env import DEFAULT_CONTRACT, LongHorizonStriker
from tools.train_striker_teacher import _load_kick_prior_bank


def _snapshot(data: Any, env: LongHorizonStriker, target: np.ndarray) -> dict[str, Any]:
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    root_qpos = env._mj_model.joint(env.prefix + "root").qposadr[0]
    return {
        "joint_target_rad": np.asarray(target, dtype=np.float64).tolist(),
        "joint_position_rad": qpos[env._joint_qpos].tolist(),
        "joint_velocity_rad_s": qvel[env._joint_dof].tolist(),
        "root_position_m": qpos[root_qpos : root_qpos + 3].tolist(),
        "torso_quaternion_wxyz": np.asarray(data.xquat, dtype=np.float64)[
            env._torso_body
        ].tolist(),
        "torso_height_m": float(np.asarray(data.xpos)[env._torso_body, 2]),
        "ball_position_m": np.asarray(data.xpos, dtype=np.float64)[
            env._ball_body
        ].tolist(),
        "ball_velocity_mps": qvel[env._ball_dof : env._ball_dof + 3].tolist(),
    }


def _errors(cpu: dict[str, Any], accelerated: dict[str, Any]) -> dict[str, float]:
    def maximum(name: str) -> float:
        return float(
            np.max(
                np.abs(
                    np.asarray(cpu[name], dtype=np.float64)
                    - np.asarray(accelerated[name], dtype=np.float64)
                )
            )
        )

    def norm(name: str) -> float:
        return float(
            np.linalg.norm(
                np.asarray(cpu[name], dtype=np.float64)
                - np.asarray(accelerated[name], dtype=np.float64)
            )
        )

    return {
        "target_max_abs_rad": maximum("joint_target_rad"),
        "joint_position_max_abs_rad": maximum("joint_position_rad"),
        "joint_velocity_max_abs_rad_s": maximum("joint_velocity_rad_s"),
        "root_position_norm_m": norm("root_position_m"),
        "torso_orientation_rad": quaternion_angle_error(
            cpu["torso_quaternion_wxyz"], accelerated["torso_quaternion_wxyz"]
        ),
        "torso_height_abs_m": abs(
            float(cpu["torso_height_m"]) - float(accelerated["torso_height_m"])
        ),
        "ball_position_norm_m": norm("ball_position_m"),
        "ball_velocity_norm_mps": norm("ball_velocity_mps"),
    }


def _summarize(
    trace: list[dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, Any]:
    first = {name: None for name in thresholds}
    maxima = {name: 0.0 for name in thresholds}
    finite = True
    for row in trace:
        for name, threshold in thresholds.items():
            value = float(row["errors"][name])
            finite &= bool(np.isfinite(value))
            maxima[name] = max(maxima[name], value)
            if value > threshold and first[name] is None:
                first[name] = int(row["step"])
    return {
        "finite": finite,
        "max_errors": maxima,
        "first_divergence_step": first,
        "gates": {name: step is None for name, step in first.items()},
        "parity_gate_passed": finite and all(step is None for step in first.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kick_prior_manifest", type=Path)
    parser.add_argument("--kick-prior-condition-index", type=int, default=1)
    parser.add_argument(
        "--kick-prior-bank-manifest", action="append", type=Path, default=[]
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--implementation", choices=("jax", "warp"), default="warp")
    parser.add_argument("--pattern", choices=("zero", "sine", "random"), default="sine")
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12_101)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 250:
        raise ValueError("steps must be in [1, 250]")
    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be absolute and outside the repository")

    contract = load_policy_contract(args.contract)
    prior, prior_distances, prior_metadata = _load_kick_prior_bank(
        args.kick_prior_manifest,
        args.kick_prior_bank_manifest,
        contract,
        primary_condition_index=args.kick_prior_condition_index,
    )
    env = LongHorizonStriker(
        config_overrides={
            "impl": args.implementation,
            "episode_length": max(args.steps + 1, 120),
            "robot_distance_range": [0.31, 0.31],
            "robot_lateral_range": [0.04, 0.04],
            "robot_yaw_noise_range": [0.0, 0.0],
            "target_angle_range": [0.0, 0.0],
            "target_distance_range": [2.0, 2.0],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "fixed_action_mode": 0,
            "fixed_desired_arrival_speed": 0.8,
            "fixed_kick_prior_index": 0,
            "naconmax": 2048,
        },
        contract=contract,
        kick_prior_joint_residuals=prior,
        kick_prior_target_distances=prior_distances,
    )
    actions = generate_action_sequence(
        pattern=args.pattern,
        steps=args.steps,
        action_size=env.action_size,
        amplitude=args.amplitude,
        seed=args.seed,
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    accelerated = reset(jax.random.PRNGKey(args.seed))
    accelerated.data.qpos.block_until_ready()
    cpu = mujoco.MjData(env.mj_model)
    cpu.qpos[:] = np.asarray(accelerated.data.qpos, dtype=np.float64)
    cpu.qvel[:] = np.asarray(accelerated.data.qvel, dtype=np.float64)
    cpu.ctrl[:] = np.asarray(accelerated.data.ctrl, dtype=np.float64)
    mujoco.mj_forward(env.mj_model, cpu)

    initial_target = np.asarray(accelerated.data.ctrl)[env._pos_actuator]
    cpu_initial = _snapshot(cpu, env, initial_target)
    accelerated_initial = _snapshot(accelerated.data, env, initial_target)
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "time_s": 0.0,
            "action": np.zeros(env.action_size).tolist(),
            "cpu": cpu_initial,
            args.implementation: accelerated_initial,
            "errors": _errors(cpu_initial, accelerated_initial),
        }
    ]
    for control_step, action in enumerate(actions, start=1):
        accelerated = step(accelerated, jp.asarray(action))
        accelerated.data.qpos.block_until_ready()
        target = np.asarray(accelerated.data.ctrl)[env._pos_actuator]
        cpu.ctrl[env._pos_actuator] = target
        for _ in range(env.n_substeps):
            mujoco.mj_step(env.mj_model, cpu)
        cpu_snapshot = _snapshot(cpu, env, target)
        accelerated_snapshot = _snapshot(accelerated.data, env, target)
        trace.append(
            {
                "step": control_step,
                "time_s": control_step * env.dt,
                "action": np.asarray(action, dtype=np.float64).tolist(),
                "cpu": cpu_snapshot,
                args.implementation: accelerated_snapshot,
                "errors": _errors(cpu_snapshot, accelerated_snapshot),
            }
        )

    thresholds = {
        "target_max_abs_rad": 1.0e-6,
        "joint_position_max_abs_rad": 0.05,
        "joint_velocity_max_abs_rad_s": 1.0,
        "root_position_norm_m": 0.03,
        "torso_orientation_rad": 0.08,
        "torso_height_abs_m": 0.03,
        "ball_position_norm_m": 0.05,
        "ball_velocity_norm_mps": 0.50,
    }
    summary = _summarize(trace, thresholds)
    payload = {
        "schema_version": 1,
        "purpose": "striker_identical_control_cpu_mjx_parity",
        "cpu_engine": f"MuJoCo {mujoco.__version__}",
        "accelerated_implementation": args.implementation,
        "jax_backend": jax.default_backend(),
        "contract": str(args.contract.resolve()),
        "kick_prior": prior_metadata,
        "pattern": args.pattern,
        "amplitude": args.amplitude,
        "seed": args.seed,
        "steps": args.steps,
        "control_dt_s": env.dt,
        "physics_substeps": env.n_substeps,
        "thresholds": thresholds,
        "summary": summary,
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    if args.strict and not summary["parity_gate_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
