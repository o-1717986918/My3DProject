#!/usr/bin/env python3
"""Verify JAX GPU execution and a batched Booster T1 rollout."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
from mujoco_playground import registry


def block(tree):
    return jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'gpu'")

    env = registry.load("T1JoystickFlatTerrain", config_overrides={"impl": args.impl})
    if env.action_size != 23:
        raise RuntimeError(f"T1 action size is {env.action_size}, expected 23")

    keys = jax.random.split(jax.random.PRNGKey(0), args.num_envs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    actions = jnp.zeros((args.num_envs, env.action_size), dtype=jnp.float32)

    compile_started = time.perf_counter()
    state = block(reset(keys))
    state = block(step(state, actions))
    compile_seconds = time.perf_counter() - compile_started

    rollout_started = time.perf_counter()
    for _ in range(args.steps):
        state = step(state, actions)
    block(state.reward)
    rollout_seconds = time.perf_counter() - rollout_started
    memory_stats = jax.devices()[0].memory_stats() or {}

    payload = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "environment": "T1JoystickFlatTerrain",
        "implementation": args.impl,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "compile_and_first_step_seconds": round(compile_seconds, 3),
        "rollout_seconds": round(rollout_seconds, 3),
        "environment_steps_per_second": round(
            args.num_envs * args.steps / rollout_seconds, 1
        ),
        "bytes_in_use": memory_stats.get("bytes_in_use"),
        "peak_bytes_in_use": memory_stats.get("peak_bytes_in_use"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
