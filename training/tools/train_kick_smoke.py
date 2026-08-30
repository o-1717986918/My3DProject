#!/usr/bin/env python3
"""Run a small, real PPO update against the RCSS-physics kick task.

This is an integration test for the complete training path.  Its checkpoint is
not a competition policy and must not be promoted through the release gates.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import time
from typing import Any, Mapping

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src import wrapper

from my3d_rl.kick_env import DirectionalKick


def _json_value(value: Any) -> Any:
    """Convert scalar JAX/NumPy values into JSON-compatible Python values."""
    try:
        return value.item()
    except (AttributeError, ValueError):
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/win98/rl_runs/kick-smoke"),
    )
    args = parser.parse_args()

    batch_size = 64
    num_minibatches = 4
    if (batch_size * num_minibatches) % args.num_envs:
        raise ValueError(
            "num_envs must divide batch_size * num_minibatches "
            f"({batch_size * num_minibatches})"
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = DirectionalKick(config_overrides={"impl": args.impl})
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    history: list[dict[str, Any]] = []

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
        row = {
            "num_steps": num_steps,
            **{key: _json_value(value) for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    started = time.monotonic()
    _, _, final_metrics = ppo.train(
        environment=env,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        episode_length=env._config.episode_length,
        action_repeat=1,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        learning_rate=3.0e-4,
        entropy_cost=5.0e-3,
        discounting=0.995,
        unroll_length=16,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=2,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        bootstrap_on_timeout=False,
        network_factory=network_factory,
        seed=args.seed,
        num_evals=1,
        num_eval_envs=16,
        run_evals=False,
        progress_fn=progress,
        save_checkpoint_path=str(args.run_dir / "checkpoints"),
    )
    elapsed = time.monotonic() - started

    manifest = {
        "purpose": "training_pipeline_smoke_test_only",
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "devices": [str(device) for device in jax.devices()],
        "num_envs": args.num_envs,
        "requested_timesteps": args.num_timesteps,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "final_metrics": {
            key: _json_value(value) for key, value in final_metrics.items()
        },
        "progress": history,
    }
    (args.run_dir / "smoke-result.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
