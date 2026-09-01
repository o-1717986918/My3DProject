#!/usr/bin/env python3
"""Compile and step the finite multi-motion soccer environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp

from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.soccer_motion_env import FiniteSoccerMotionTracking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--failure-report", type=Path)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    corpus = load_soccer_motion_corpus(
        args.corpus_root, failure_report=args.failure_report
    )
    env = FiniteSoccerMotionTracking(
        corpus, config_overrides={"impl": args.impl}
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(args.seed))
    initial_motion = int(state.info["motion"])
    initial_frame = int(state.info["reference_frame"])
    for _ in range(args.steps):
        state = step(state, jp.zeros(env.action_size))
    payload = {
        "status": "passed",
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "motion_count": corpus.motion_count,
        "maximum_frames": corpus.maximum_frames,
        "initial_motion": initial_motion,
        "initial_frame": initial_frame,
        "final_frame": int(state.info["reference_frame"]),
        "actor_observation_shape": list(state.obs["state"].shape),
        "privileged_observation_shape": list(
            state.obs["privileged_state"].shape
        ),
        "reward": float(state.reward),
        "done": float(state.done),
        "torso_height_m": float(state.metrics["diagnostic/torso_height"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
