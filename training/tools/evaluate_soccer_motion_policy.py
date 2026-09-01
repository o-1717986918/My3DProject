#!/usr/bin/env python3
"""Fixed-seed MJX evaluation for finite soccer-motion checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.soccer_motion_env import (
    DEFAULT_CONTRACT,
    FiniteSoccerMotionTracking,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument(
        "--profile", default="soccer_motion_residual_v1"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.zero_policy == (args.checkpoint is not None):
        raise ValueError("select exactly one of --zero-policy or --checkpoint")
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    corpus = load_soccer_motion_corpus(args.corpus_root)
    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    env = FiniteSoccerMotionTracking(
        corpus,
        config_overrides={
            "impl": args.impl,
            "naconmax": max(2048, 8 * args.episodes),
            "reset_joint_noise": 0.002,
            "reset_root_velocity_noise": 0.005,
            "reset_yaw_range": 0.01,
            "action_delay_max_steps": 0,
        },
        contract=contract,
    )
    if args.zero_policy:
        def policy(observation: dict[str, jax.Array], rng: jax.Array):
            del rng
            return jp.zeros(env.action_size), {}
    else:
        preprocess = (
            running_statistics.normalize
            if profile.normalize_observations
            else brax_types.identity_observation_preprocessor
        )
        networks = profile.network_factory()(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=preprocess,
        )
        params = ppo_checkpoint.load(args.checkpoint)
        policy = ppo_networks.make_inference_fn(networks)(
            params, deterministic=True
        )

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    policy_batch = jax.vmap(policy)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    initial = reset(keys)
    initial_motion = np.asarray(initial.info["motion"])
    initial_frame = np.asarray(initial.info["reference_frame"])

    def scan_step(carry: tuple[Any, jax.Array], unused: None):
        del unused
        state, rng = carry
        rng, action_rng = jax.random.split(rng)
        action_keys = jax.random.split(action_rng, args.episodes)
        actions = policy_batch(state.obs, action_keys)[0]
        next_state = step(state, actions)
        diagnostics = jp.stack(
            [
                next_state.done,
                next_state.metrics["cost/fall"],
                next_state.metrics["reward/completion"],
                next_state.metrics["reward/motion_joint"],
                next_state.metrics["reward/motion_contact"],
                next_state.metrics["diagnostic/reference_phase"],
                jp.mean(jp.abs(actions), axis=1),
                jp.max(jp.abs(actions), axis=1),
            ],
            axis=-1,
        )
        return (next_state, rng), diagnostics

    @jax.jit
    def rollout(state, rng):
        return jax.lax.scan(
            scan_step,
            (state, rng),
            None,
            length=env._config.episode_length,
        )

    (_, _), diagnostics = rollout(initial, jax.random.PRNGKey(args.seed + 1))
    values = np.asarray(diagnostics)
    done = values[:, :, 0] > 0.0
    has_done = done.any(axis=0)
    first_done_index = np.argmax(done, axis=0)
    episode_length = np.where(
        has_done, first_done_index + 1, env._config.episode_length
    )
    episode_index = np.arange(args.episodes)
    terminal = values[first_done_index, episode_index]
    fall = has_done & (terminal[:, 1] > 0.0)
    completed = has_done & (terminal[:, 2] > 0.0)
    timeout = ~fall & ~completed
    mean_motion_joint = np.empty(args.episodes)
    mean_motion_contact = np.empty(args.episodes)
    mean_action = np.empty(args.episodes)
    maximum_action = np.empty(args.episodes)
    for episode, length in enumerate(episode_length):
        segment = values[: int(length), episode]
        mean_motion_joint[episode] = np.mean(segment[:, 3])
        mean_motion_contact[episode] = np.mean(segment[:, 4])
        mean_action[episode] = np.mean(segment[:, 6])
        maximum_action[episode] = np.max(segment[:, 7])

    per_motion: list[dict[str, Any]] = []
    for motion, relative_path in enumerate(corpus.relative_paths):
        selected = initial_motion == motion
        count = int(np.sum(selected))
        per_motion.append(
            {
                "motion": motion,
                "relative_path": relative_path,
                "episodes": count,
                "completion_rate": (
                    float(np.mean(completed[selected])) if count else None
                ),
                "fall_rate": float(np.mean(fall[selected])) if count else None,
                "median_episode_length": (
                    float(np.median(episode_length[selected])) if count else None
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "k1_fixed_seed_finite_motion_policy_evaluation",
        "policy": "zero_residual" if args.zero_policy else "checkpoint",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "profile": profile.name,
        "implementation": args.impl,
        "backend": jax.default_backend(),
        "episodes": args.episodes,
        "seed": args.seed,
        "corpus_root": str(args.corpus_root.resolve()),
        "motion_count": corpus.motion_count,
        "initial_frame": {
            "minimum": int(np.min(initial_frame)),
            "median": float(np.median(initial_frame)),
            "maximum": int(np.max(initial_frame)),
        },
        "completion_rate": float(np.mean(completed)),
        "fall_rate": float(np.mean(fall)),
        "timeout_rate": float(np.mean(timeout)),
        "episode_length": {
            "mean": float(np.mean(episode_length)),
            "p10": float(np.percentile(episode_length, 10)),
            "median": float(np.median(episode_length)),
            "p90": float(np.percentile(episode_length, 90)),
        },
        "mean_motion_joint_reward": float(np.mean(mean_motion_joint)),
        "mean_motion_contact_reward": float(np.mean(mean_motion_contact)),
        "mean_abs_action": float(np.mean(mean_action)),
        "maximum_abs_action": float(np.max(maximum_action)),
        "invalid_value_count": int(np.sum(~np.isfinite(values))),
        "per_motion": per_motion,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
