#!/usr/bin/env python3
"""Evaluate an Apollo-compatible ONNX policy on incoming goalkeeper shots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.apollo_walk_jax import load_apollo_walk_jax
from my3d_rl.goalkeeper_env import ApolloGoalkeeperBlock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument(
        "--warp-graph-mode",
        choices=("auto", "none", "jax", "warp", "warp_staged"),
        default="auto",
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_261_510)
    parser.add_argument("--shot-side", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument(
        "--shot-speed",
        type=float,
        help="fix incoming speed in m/s instead of sampling the training range",
    )
    parser.add_argument(
        "--shot-lateral",
        type=float,
        help="fix the unsigned goal-plane offset in metres",
    )
    parser.add_argument(
        "--shot-start-distance",
        type=float,
        help="fix the shot start distance in metres",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.impl != "warp" and args.warp_graph_mode != "auto":
        raise ValueError("--warp-graph-mode requires --impl warp")

    environment_overrides = {
        "impl": args.impl,
        "warp_graph_mode": args.warp_graph_mode,
        "naconmax": max(2048, 16 * args.episodes),
        "fixed_shot_side": args.shot_side,
        "reset_joint_noise": 0.01,
        "reset_root_velocity_noise": 0.02,
        "reset_yaw_range": 0.03,
    }
    for argument, config_name in (
        (args.shot_speed, "shot_speed_range"),
        (args.shot_lateral, "shot_lateral_range"),
        (args.shot_start_distance, "shot_start_distance_range"),
    ):
        if argument is not None:
            if not np.isfinite(argument) or argument <= 0.0:
                raise ValueError(f"{config_name} override must be positive and finite")
            environment_overrides[config_name] = [argument, argument]
    env = ApolloGoalkeeperBlock(config_overrides=environment_overrides)
    policy = load_apollo_walk_jax(args.onnx)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    initial_state = reset(keys)

    def scan_step(state, unused):
        del unused
        actions = policy(state.obs["state"][..., : policy.observation_size])
        next_state = step(state, actions)
        diagnostics = jp.stack(
            [
                next_state.metrics["event/goalkeeper_contact"],
                next_state.metrics["event/goalkeeper_save"],
                next_state.metrics["event/goalkeeper_concede"],
                next_state.metrics["cost/fall"],
                next_state.metrics["diagnostic/goalkeeper_error_m"],
                next_state.metrics["diagnostic/goalkeeper_time_to_plane_s"],
            ],
            axis=-1,
        )
        return next_state, diagnostics

    @jax.jit
    def rollout(state):
        return jax.lax.scan(
            scan_step, state, None, length=env._config.episode_length
        )

    _, diagnostics = rollout(initial_state)
    values = np.asarray(diagnostics)
    contact = values[:, :, 0].max(axis=0) > 0.5
    saved = values[:, :, 1].max(axis=0) > 0.5
    conceded = values[:, :, 2].max(axis=0) > 0.5
    fallen = values[:, :, 3].max(axis=0) > 0.5
    unresolved = ~(saved | conceded)
    minimum_error = values[:, :, 4].min(axis=0)
    initial_lateral = np.abs(
        np.asarray(initial_state.info["shot_lateral_m"], dtype=np.float64)
    )
    shot_speed = np.asarray(
        initial_state.info["shot_speed_mps"], dtype=np.float64
    )
    payload = {
        "schema_version": 1,
        "purpose": "goalkeeper_standing_block_evaluation",
        "onnx": str(args.onnx.resolve()),
        "actor_observation_size": policy.observation_size,
        "implementation": args.impl,
        "warp_graph_mode": args.warp_graph_mode,
        "episodes": args.episodes,
        "seed": args.seed,
        "shot_side": args.shot_side,
        "fixed_shot_speed_mps": args.shot_speed,
        "fixed_shot_lateral_m": args.shot_lateral,
        "fixed_shot_start_distance_m": args.shot_start_distance,
        "shot_lateral_range_m": [
            float(initial_lateral.min()),
            float(initial_lateral.max()),
        ],
        "shot_speed_range_mps": [
            float(shot_speed.min()),
            float(shot_speed.max()),
        ],
        "contact_count": int(contact.sum()),
        "save_count": int(saved.sum()),
        "concede_count": int(conceded.sum()),
        "fall_count": int(fallen.sum()),
        "unresolved_count": int(unresolved.sum()),
        "save_rate": float(saved.mean()),
        "concede_rate": float(conceded.mean()),
        "contact_rate": float(contact.mean()),
        "minimum_keeper_error_m": {
            "median": float(np.median(minimum_error)),
            "p90": float(np.percentile(minimum_error, 90)),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
