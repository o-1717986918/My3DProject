#!/usr/bin/env python3
"""Replay identical kick controls in CPU MuJoCo and MJX and locate divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_env import DirectionalKick, TRANSITION_CONTRACT, default_config
from my3d_rl.kick_teacher import build_joint_delta_trajectory
from my3d_rl.sim_parity import quaternion_angle_error


def _load_condition(
    path: Path, condition_index: int, contract
) -> tuple[np.ndarray, np.ndarray]:
    source = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in source["records"]
        if int(record["condition_index"]) == condition_index
        and bool(record["accepted"])
    ]
    if len(matches) != 1:
        raise ValueError("condition index must select one accepted teacher record")
    record = matches[0]
    config = default_config()
    times = np.arange(config.episode_length) * config.ctrl_dt
    trajectory = build_joint_delta_trajectory(
        np.asarray(record["parameters"], dtype=np.float64), contract, times
    ).astype(np.float32)
    offset = np.array(
        [record["ball_x_offset_m"], record["ball_y_offset_m"]],
        dtype=np.float32,
    )
    return trajectory[None], offset[None]


def _load_entry(path: Path, rollout_id: int | None):
    with np.load(path, allow_pickle=False) as archive:
        split = np.asarray(archive["split"], dtype=np.uint8)
        ids = np.asarray(archive["rollout_id"], dtype=np.int32)
        candidates = np.flatnonzero(split == 1)
        if rollout_id is None:
            selected = int(candidates[0])
        else:
            matches = np.flatnonzero((ids == rollout_id) & (split == 1))
            if matches.size != 1:
                raise ValueError("rollout ID must select one validation entry")
            selected = int(matches[0])
        return (
            np.asarray(archive["qpos"][selected], dtype=np.float32),
            np.asarray(archive["qvel"][selected], dtype=np.float32),
            int(ids[selected]),
            int(np.asarray(archive["phase_bucket"])[selected]),
        )


def _snapshot(data: Any, env: DirectionalKick, target: np.ndarray) -> dict[str, Any]:
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    return {
        "joint_target_rad": np.asarray(target, dtype=np.float64).tolist(),
        "joint_position_rad": qpos[env._joint_qpos].tolist(),
        "joint_velocity_rad_s": qvel[env._joint_dof].tolist(),
        "root_position_m": qpos[
            env._mj_model.joint(env.prefix + "root").qposadr[0] :
            env._mj_model.joint(env.prefix + "root").qposadr[0] + 3
        ].tolist(),
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
    def max_abs(name: str) -> float:
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
        "target_max_abs_rad": max_abs("joint_target_rad"),
        "joint_position_max_abs_rad": max_abs("joint_position_rad"),
        "joint_velocity_max_abs_rad_s": max_abs("joint_velocity_rad_s"),
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


def _summary(trace: list[dict[str, Any]], thresholds: dict[str, float]):
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
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--contract", type=Path, default=TRANSITION_CONTRACT)
    parser.add_argument("--correction-onnx", type=Path)
    parser.add_argument("--implementation", choices=("jax", "warp"), default="jax")
    parser.add_argument("--rollout-id", type=int)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 150:
        raise ValueError("steps must be in [1, 150]")

    contract = load_policy_contract(args.contract)
    teacher, offsets = _load_condition(
        args.teacher_manifest, args.condition_index, contract
    )
    qpos, qvel, rollout_id, phase_bucket = _load_entry(
        args.transition_corpus, args.rollout_id
    )
    env = DirectionalKick(
        config_overrides={
            "impl": args.implementation,
            "target_distance_range": [2.0, 2.0],
            "target_angle_range": [0.0, 0.0],
            "fixed_action_mode": 0,
            "fixed_desired_arrival_speed": 0.8,
            "action_scale": (0.1 * np.asarray(default_config().action_scale)).tolist(),
        },
        contract=contract,
        teacher_joint_residuals=teacher,
        teacher_ball_offsets=offsets,
        transition_qpos=np.stack([qpos, qpos]),
        transition_qvel=np.stack([qvel, qvel]),
    )
    session = None
    input_name = None
    if args.correction_onnx is not None:
        session = ort.InferenceSession(
            str(args.correction_onnx.resolve()), providers=["CPUExecutionProvider"]
        )
        if session.get_inputs()[0].shape != list(contract.input_shape):
            raise ValueError("correction ONNX input shape mismatch")
        input_name = session.get_inputs()[0].name

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
    initial_cpu = _snapshot(cpu, env, initial_target)
    initial_accelerated = _snapshot(accelerated.data, env, initial_target)
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "time_s": 0.0,
            "action": np.zeros(contract.action_size).tolist(),
            "cpu": initial_cpu,
            args.implementation: initial_accelerated,
            "errors": _errors(initial_cpu, initial_accelerated),
        }
    ]
    for control_step in range(1, args.steps + 1):
        if session is None:
            action = np.zeros(contract.action_size, dtype=np.float32)
        else:
            observation = np.asarray(accelerated.obs["state"], dtype=np.float32)
            action = session.run(None, {input_name: observation[None]})[0][0]
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
    summary = _summary(trace, thresholds)
    payload = {
        "schema_version": 1,
        "purpose": "kick_identical_control_cpu_mjx_parity",
        "cpu_engine": f"MuJoCo {mujoco.__version__}",
        "accelerated_implementation": args.implementation,
        "jax_backend": jax.default_backend(),
        "contract": str(args.contract.resolve()),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "condition_index": args.condition_index,
        "transition_corpus": str(args.transition_corpus.resolve()),
        "rollout_id": rollout_id,
        "phase_bucket": phase_bucket,
        "correction_onnx": (
            str(args.correction_onnx.resolve()) if args.correction_onnx else None
        ),
        "steps": args.steps,
        "control_dt_s": env.dt,
        "physics_substeps": env.n_substeps,
        "thresholds": thresholds,
        "summary": summary,
        "trace": trace,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    if args.strict and not summary["parity_gate_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
