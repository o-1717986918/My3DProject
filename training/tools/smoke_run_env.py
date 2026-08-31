#!/usr/bin/env python3
"""Compile and step the RCSS-physics running task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from my3d_rl.contract import load_policy_contract
from my3d_rl.run_env import DirectionalRun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--contract-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()

    contract_path = (
        Path(__file__).parents[1]
        / "contracts"
        / f"run_policy_{args.contract_version}.yaml"
    )
    env = DirectionalRun(
        config_overrides={"impl": args.impl},
        contract=load_policy_contract(contract_path),
    )
    keys = jax.random.split(jax.random.PRNGKey(11), args.num_envs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(keys)
    actions = jnp.zeros((args.num_envs, env.action_size), dtype=jnp.float32)
    for _ in range(args.steps):
        state = step(state, actions)
    state.reward.block_until_ready()
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "implementation": args.impl,
                "num_envs": args.num_envs,
                "action_size": env.action_size,
                "observation_size": env.observation_size,
                "privileged_observation_size": int(
                    state.obs["privileged_state"].shape[-1]
                ),
                "n_substeps": env.n_substeps,
                "finite_observation": bool(jnp.isfinite(state.obs["state"]).all()),
                "finite_reward": bool(jnp.isfinite(state.reward).all()),
                "done_count": int(state.done.sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
