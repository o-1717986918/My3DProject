#!/usr/bin/env python3
"""Replay one action trace in CPU MuJoCo and MJX-Warp and report divergence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.run_env import DirectionalRun
from my3d_rl.sim_parity import (
    ParityThresholds,
    generate_action_sequence,
    step_errors,
    summarize_trace,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v2.yaml"


def _box_lowest_height(
    geom_xpos: np.ndarray,
    geom_xmat: np.ndarray,
    geom_size: np.ndarray,
    geom_id: int,
    pitch_height: float,
) -> float:
    rotation = np.asarray(geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    lowest_z = float(geom_xpos[geom_id, 2]) - float(
        np.sum(np.abs(rotation[2, :]) * geom_size[geom_id])
    )
    return lowest_z - pitch_height


def _yaw(site_xmat: np.ndarray, site_id: int) -> float:
    rotation = np.asarray(site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _actual_cpu_contact(data: mujoco.MjData, pitch_geom: int, foot_geom: int) -> bool:
    pairs = data.contact.geom[: data.ncon]
    return bool(
        np.any(
            ((pairs[:, 0] == pitch_geom) & (pairs[:, 1] == foot_geom))
            | ((pairs[:, 0] == foot_geom) & (pairs[:, 1] == pitch_geom))
        )
    )


def _snapshot(
    *,
    data: Any,
    model: mujoco.MjModel,
    env: DirectionalRun,
    joint_target: np.ndarray,
    include_actual_contact: bool,
) -> dict[str, Any]:
    qpos = np.asarray(data.qpos)
    geom_xpos = np.asarray(data.geom_xpos)
    geom_xmat = np.asarray(data.geom_xmat)
    lowest = np.array(
        [
            _box_lowest_height(
                geom_xpos,
                geom_xmat,
                model.geom_size,
                env._left_foot_geom,
                env._pitch_height,
            ),
            _box_lowest_height(
                geom_xpos,
                geom_xmat,
                model.geom_size,
                env._right_foot_geom,
                env._pitch_height,
            ),
        ]
    )
    proxy = lowest <= float(env._config.foot_contact_tolerance)
    snapshot: dict[str, Any] = {
        "joint_target_rad": np.asarray(joint_target, dtype=np.float64).tolist(),
        "joint_position_rad": qpos[env._joint_qpos].astype(np.float64).tolist(),
        "root_position_m": qpos[env._root_qpos : env._root_qpos + 3]
        .astype(np.float64)
        .tolist(),
        "root_yaw_rad": _yaw(np.asarray(data.site_xmat), env._torso_site),
        "torso_quaternion_wxyz": np.asarray(data.xquat)[env._torso_body]
        .astype(np.float64)
        .tolist(),
        "foot_lowest_height_m": lowest.tolist(),
        "contact_proxy": proxy.tolist(),
    }
    if include_actual_contact:
        pitch_geom = model.geom("pitch").id
        snapshot["actual_contact"] = [
            _actual_cpu_contact(data, pitch_geom, env._left_foot_geom),
            _actual_cpu_contact(data, pitch_geom, env._right_foot_geom),
        ]
    return snapshot


def _load_actions(
    path: Path | None,
    *,
    pattern: str,
    steps: int,
    action_size: int,
    amplitude: float,
    seed: int,
) -> np.ndarray:
    if path is None:
        return generate_action_sequence(
            pattern=pattern,
            steps=steps,
            action_size=action_size,
            amplitude=amplitude,
            seed=seed,
        )
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "action" not in loaded.files:
                raise ValueError("NPZ action trace must contain an 'action' array")
            actions = np.asarray(loaded["action"], dtype=np.float32)
        finally:
            loaded.close()
    else:
        actions = np.asarray(loaded, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != action_size:
        raise ValueError(
            f"action trace must have shape [steps, {action_size}], got {actions.shape}"
        )
    if actions.shape[0] < steps:
        raise ValueError(
            f"action trace contains {actions.shape[0]} steps, requested {steps}"
        )
    if not np.isfinite(actions[:steps]).all():
        raise ValueError("action trace contains non-finite values")
    return np.clip(actions[:steps], -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--motion-reference", type=Path)
    parser.add_argument("--implementation", choices=("warp", "jax"), default="warp")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument(
        "--action-pattern", choices=("zero", "sine", "random"), default="sine"
    )
    parser.add_argument("--action-amplitude", type=float, default=0.15)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--root-position-tolerance", type=float, default=0.03)
    parser.add_argument("--joint-position-tolerance", type=float, default=0.05)
    parser.add_argument("--torso-angle-tolerance", type=float, default=0.08)
    parser.add_argument("--foot-height-tolerance", type=float, default=0.025)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.motion_reference is not None and not args.motion_reference.is_file():
        raise FileNotFoundError(args.motion_reference)

    contract = load_policy_contract(args.contract)
    reference_probability = 1.0 if args.motion_reference is not None else 0.0
    env = DirectionalRun(
        config_overrides={
            "impl": args.implementation,
            "use_fixed_command": True,
            "fixed_command": [1.7, 0.0, 0.0],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "reference_init_probability": reference_probability,
            "push_enable": False,
            "action_delay_max_steps": 0,
        },
        contract=contract,
        motion_reference=args.motion_reference,
    )
    actions = _load_actions(
        args.actions,
        pattern=args.action_pattern,
        steps=args.steps,
        action_size=env.action_size,
        amplitude=args.action_amplitude,
        seed=args.seed,
    )
    thresholds = ParityThresholds(
        joint_position_max_abs_rad=args.joint_position_tolerance,
        root_position_norm_m=args.root_position_tolerance,
        torso_orientation_rad=args.torso_angle_tolerance,
        foot_height_max_abs_m=args.foot_height_tolerance,
    )
    thresholds.validate()

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    accelerated_state = reset(jax.random.PRNGKey(args.seed))
    accelerated_state.data.qpos.block_until_ready()
    initial_gait_phase = float(np.asarray(accelerated_state.info["gait_phase"]))

    cpu_model = env.mj_model
    cpu_data = mujoco.MjData(cpu_model)
    cpu_data.qpos[:] = np.asarray(accelerated_state.data.qpos, dtype=np.float64)
    cpu_data.qvel[:] = np.asarray(accelerated_state.data.qvel, dtype=np.float64)
    cpu_data.ctrl[:] = np.asarray(accelerated_state.data.ctrl, dtype=np.float64)
    mujoco.mj_forward(cpu_model, cpu_data)

    initial_target = np.asarray(accelerated_state.data.ctrl)[env._pos_actuator]
    initial_cpu = _snapshot(
        data=cpu_data,
        model=cpu_model,
        env=env,
        joint_target=initial_target,
        include_actual_contact=True,
    )
    initial_accelerated = _snapshot(
        data=accelerated_state.data,
        model=cpu_model,
        env=env,
        joint_target=initial_target,
        include_actual_contact=False,
    )
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "time_seconds": 0.0,
            "reference_phase": initial_gait_phase,
            "action": np.zeros(env.action_size).tolist(),
            "cpu": initial_cpu,
            args.implementation: initial_accelerated,
            "errors": step_errors(initial_cpu, initial_accelerated),
        }
    ]

    for index, action in enumerate(actions, start=1):
        phase = accelerated_state.info["gait_phase"]
        targets = np.asarray(
            env.decode_action_targets(jnp.asarray(action), phase), dtype=np.float64
        )
        cpu_data.ctrl[env._pos_actuator] = targets
        for _ in range(env.n_substeps):
            mujoco.mj_step(cpu_model, cpu_data)

        accelerated_state = step(accelerated_state, jnp.asarray(action))
        accelerated_state.data.qpos.block_until_ready()
        accelerated_target = np.asarray(accelerated_state.data.ctrl)[env._pos_actuator]
        cpu_snapshot = _snapshot(
            data=cpu_data,
            model=cpu_model,
            env=env,
            joint_target=targets,
            include_actual_contact=True,
        )
        accelerated_snapshot = _snapshot(
            data=accelerated_state.data,
            model=cpu_model,
            env=env,
            joint_target=accelerated_target,
            include_actual_contact=False,
        )
        trace.append(
            {
                "step": index,
                "time_seconds": index * env.dt,
                "reference_phase": float(np.asarray(phase)),
                "action": action.astype(np.float64).tolist(),
                "cpu": cpu_snapshot,
                args.implementation: accelerated_snapshot,
                "errors": step_errors(cpu_snapshot, accelerated_snapshot),
            }
        )

    summary = summarize_trace(trace, thresholds)
    payload = {
        "schema_version": 1,
        "purpose": "short_horizon_identical_state_action_backend_parity",
        "contract": str(args.contract.resolve()),
        "motion_reference": (
            str(args.motion_reference.resolve()) if args.motion_reference else None
        ),
        "cpu_engine": f"MuJoCo {mujoco.__version__}",
        "accelerated_implementation": args.implementation,
        "jax_backend": jax.default_backend(),
        "seed": args.seed,
        "steps": args.steps,
        "control_dt_seconds": env.dt,
        "physics_substeps": env.n_substeps,
        "initial_gait_phase": initial_gait_phase,
        "action_source": (
            str(args.actions.resolve()) if args.actions else args.action_pattern
        ),
        "action_amplitude": args.action_amplitude if args.actions is None else None,
        "thresholds": asdict(thresholds),
        "summary": summary,
        "trace": trace,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()) if args.output else None,
                "accelerated_implementation": args.implementation,
                "steps": args.steps,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and not summary["parity_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
