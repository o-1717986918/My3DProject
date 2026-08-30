#!/usr/bin/env python3
"""Compile and step the first RCSS-physics MJX kick task."""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from my3d_rl.kick_env import DirectionalKick


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()

    env = DirectionalKick(config_overrides={"impl": args.impl})
    keys = jax.random.split(jax.random.PRNGKey(7), args.num_envs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(keys)
    state = step(state, jnp.zeros((args.num_envs, env.action_size)))
    state.reward.block_until_ready()
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "implementation": args.impl,
                "num_envs": args.num_envs,
                "action_size": env.action_size,
                "observation_size": env.observation_size,
                "n_substeps": env.n_substeps,
                "finite_reward": bool(jnp.isfinite(state.reward).all()),
                "done_count": int(state.done.sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
