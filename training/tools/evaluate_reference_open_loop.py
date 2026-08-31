#!/usr/bin/env python3
"""Evaluate zero-residual tracking before spending PPO compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.motion_reference import sha256, validate_motion_reference
from my3d_rl.run_env import DirectionalRun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).parents[1] / "contracts" / "run_policy_v3.yaml",
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--vx", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.episodes < 1 or args.steps < 1:
        raise ValueError("episodes and steps must be positive")
    validation = validate_motion_reference(args.reference)
    if not validation["passed"]:
        raise ValueError("motion reference failed validation")
    contract = load_policy_contract(args.contract)
    reference_sha256 = sha256(args.reference)
    if contract.reference_sha256 and reference_sha256 != contract.reference_sha256:
        raise ValueError("motion reference SHA-256 differs from the policy contract")

    env = DirectionalRun(
        config_overrides={
            "impl": args.impl,
            "naconmax": max(2048, 16 * args.episodes),
            "episode_length": args.steps,
            "use_fixed_command": True,
            "fixed_command": [args.vx, 0.0, 0.0],
            "reference_init_probability": 1.0,
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "push_enable": False,
            "action_delay_max_steps": 0,
            "foot_contact_tolerance": 0.0,
        },
        contract=contract,
        motion_reference=args.reference,
    )
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    initial_state = reset(keys)
    initial_xy = initial_state.data.qpos[:, env._root_qpos : env._root_qpos + 2]
    zero_action = jp.zeros((args.episodes, env.action_size), dtype=jp.float32)

    def scan_step(state, unused):
        del unused
        next_state = step(state, zero_action)
        diagnostics = jp.stack(
            [
                next_state.done,
                next_state.metrics["diagnostic/local_velocity_x"],
                next_state.metrics["diagnostic/local_velocity_y"],
                next_state.metrics["diagnostic/yaw_rate"],
                next_state.metrics["diagnostic/torso_height"],
                next_state.metrics["reward/flight"],
                next_state.metrics["reward/motion_joint"],
                next_state.metrics["reward/motion_contact"],
                next_state.data.qpos[:, env._root_qpos],
                next_state.data.qpos[:, env._root_qpos + 1],
            ],
            axis=-1,
        )
        return next_state, diagnostics

    @jax.jit
    def rollout(state):
        return jax.lax.scan(scan_step, state, None, length=args.steps)

    _, values = rollout(initial_state)
    values = np.asarray(values)
    done = values[:, :, 0] > 0.0
    has_done = done.any(axis=0)
    first_done = np.argmax(done, axis=0) + 1
    episode_length = np.where(has_done, first_done, args.steps)
    alive = np.arange(args.steps)[:, None] < episode_length[None, :]

    def alive_mean(index: int) -> float:
        return float(np.sum(values[:, :, index] * alive) / np.sum(alive))

    terminal_index = np.maximum(episode_length - 1, 0)
    terminal_xy = values[terminal_index, np.arange(args.episodes), 8:10]
    initial_xy = np.asarray(initial_xy)
    payload = {
        "schema_version": 1,
        "purpose": "zero_residual_reference_trackability_diagnostic",
        "reference": str(args.reference.resolve()),
        "reference_sha256": reference_sha256,
        "implementation": args.impl,
        "backend": jax.default_backend(),
        "episodes": args.episodes,
        "steps": args.steps,
        "command_vx_m_s": args.vx,
        "gait_frequency_hz": float(np.asarray(initial_state.info["gait_frequency"])[0]),
        "zero_action_target_invariant": True,
        "episode_length": {
            "mean_steps": float(np.mean(episode_length)),
            "median_steps": float(np.median(episode_length)),
            "minimum_steps": int(np.min(episode_length)),
            "maximum_steps": int(np.max(episode_length)),
            "full_episode_rate": float(np.mean(~has_done)),
        },
        "alive_weighted": {
            "local_velocity_x_m_s": alive_mean(1),
            "local_velocity_y_m_s": alive_mean(2),
            "yaw_rate_rad_s": alive_mean(3),
            "torso_height_m": alive_mean(4),
            "motion_joint": alive_mean(6),
            "motion_contact": alive_mean(7),
        },
        "proxy_flight_episode_rate": float(
            np.mean(np.any((values[:, :, 5] > 0.0) & alive, axis=0))
        ),
        "terminal_absolute_lateral_drift_m": {
            "median": float(np.median(np.abs(terminal_xy[:, 1] - initial_xy[:, 1]))),
            "p90": float(
                np.percentile(np.abs(terminal_xy[:, 1] - initial_xy[:, 1]), 90)
            ),
        },
        "finite": bool(np.isfinite(values).all()),
        "release_candidate": False,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
