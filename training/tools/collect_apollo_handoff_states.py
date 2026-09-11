#!/usr/bin/env python3
"""Collect live frozen-Apollo Walk states for handoff training resets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.apollo_run_env import ApolloHandoffWaypointRun, ApolloWaypointRun
from my3d_rl.apollo_walk_jax import load_apollo_walk_jax
from my3d_rl.run_env import DirectionalRun


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_MODEL = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo_rebuild"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_new_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("run-dir must be absolute")
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("run-dir must stay outside the repository")
    if resolved.exists():
        raise FileExistsError(f"run-dir already exists: {resolved}")
    return resolved


def _keep_active(old: Any, new: Any, active: jax.Array) -> Any:
    if (
        not hasattr(new, "ndim")
        or new.ndim == 0
        or new.shape[0] != active.shape[0]
    ):
        return new
    mask = active.reshape(active.shape + (1,) * (new.ndim - active.ndim))
    return jp.where(mask, new, old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--burn-in-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_261_413)
    args = parser.parse_args()
    if (
        not args.model.is_file()
        or args.num_envs < 4
        or args.steps < 2
        or not 0 <= args.burn_in_steps < args.steps
        or args.seed < 0
    ):
        raise ValueError("handoff collection arguments are invalid")
    run_dir = _external_new_directory(args.run_dir)
    env = ApolloWaypointRun(
        config_overrides={
            "impl": args.impl,
            "episode_length": args.steps + 1,
            "naconmax": max(2048, 16 * args.num_envs),
            "use_fixed_command": False,
            "lin_vel_x": [-0.5, 1.0],
            "lin_vel_y": [-0.5, 0.5],
            "ang_vel_yaw": [-0.5, 0.5],
            "stand_probability": 0.0,
            "command_resample_steps": args.steps + 1,
            "reset_joint_noise": 0.0,
            "reset_joint_velocity_noise": 0.0,
            "reset_policy_action_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "push_enable": False,
            "action_delay_max_steps": 0,
        }
    )
    policy = load_apollo_walk_jax(args.model)
    reset = jax.jit(
        jax.vmap(lambda rng: DirectionalRun.reset(env, rng))
    )
    batched_step = jax.vmap(
        lambda state, action: DirectionalRun.step(env, state, action)
    )
    initial = reset(
        jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    )

    def scan_step(carry, step_index):
        state, active = carry
        action = policy(state.obs["state"])
        candidate = batched_step(state, action)
        fallen = candidate.metrics["cost/fall"] > 0.5
        valid = active & ~fallen & (step_index >= args.burn_in_steps)
        next_active = active & ~fallen
        next_state = jax.tree.map(
            lambda old, new: _keep_active(old, new, next_active),
            state,
            candidate,
        )
        record = {
            "qpos": candidate.data.qpos,
            "qvel": candidate.data.qvel,
            "last_action": candidate.info["last_action"],
            "last_last_action": candidate.info["last_last_action"],
            "source_command": candidate.info["command"],
            "source_step": jp.full(
                (args.num_envs,), step_index + 1, dtype=jp.int32
            ),
            "valid": valid,
            "fallen": active & fallen,
        }
        return (next_state, next_active), record

    (_, _), rollout = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            (state, jp.ones(args.num_envs, dtype=bool)),
            jp.arange(args.steps, dtype=jp.int32),
        )
    )(initial)
    arrays = jax.tree.map(np.asarray, rollout)
    valid = arrays.pop("valid").reshape(-1).astype(bool)
    fallen = arrays.pop("fallen")
    flattened = {
        name: value.reshape((-1,) + value.shape[2:])[valid]
        for name, value in arrays.items()
    }
    if not flattened["qpos"].shape[0]:
        raise ValueError("frozen Walk produced no valid handoff states")
    for name, value in flattened.items():
        if not np.isfinite(value).all():
            raise ValueError(f"collected {name} contains non-finite values")

    run_dir.mkdir(parents=True)
    corpus_path = run_dir / "apollo-handoff-states.npz"
    np.savez_compressed(corpus_path, **flattened)

    replay_steps = 25
    replay_env = ApolloHandoffWaypointRun(
        entry_policy=policy,
        entry_corpus=corpus_path,
        config_overrides={
            "impl": args.impl,
            "episode_length": replay_steps + 1,
            "naconmax": max(2048, 16 * args.num_envs),
            "use_fixed_command": False,
            "lin_vel_x": [-0.5, 1.0],
            "lin_vel_y": [-0.5, 0.5],
            "ang_vel_yaw": [-0.5, 0.5],
            "stand_probability": 0.0,
            "command_resample_steps": replay_steps + 1,
            "reset_joint_noise": 0.0,
            "reset_joint_velocity_noise": 0.0,
            "reset_policy_action_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "push_enable": False,
            "action_delay_max_steps": 0,
        },
    )
    replay_reset = jax.jit(jax.vmap(replay_env.reset_source_replay))
    replay_step = jax.vmap(
        lambda state, action: DirectionalRun.step(replay_env, state, action)
    )
    replay_initial = replay_reset(
        jax.random.split(jax.random.PRNGKey(args.seed + 1), args.num_envs)
    )

    def replay_scan_step(state, unused):
        action = policy(state.obs["state"])
        state = replay_step(state, action)
        return state, state.metrics["cost/fall"] > 0.5

    _, replay_fallen = jax.jit(
        lambda state: jax.lax.scan(
            replay_scan_step, state, xs=None, length=replay_steps
        )
    )(replay_initial)
    replay_falls = int(np.sum(np.any(np.asarray(replay_fallen), axis=0)))
    if replay_falls:
        raise ValueError(
            f"restored source-command replay fell in {replay_falls}/"
            f"{args.num_envs} episodes"
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "frozen_apollo_walk_handoff_state_corpus",
        "promotable": False,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip(),
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "implementation": args.impl,
        "backend": jax.default_backend(),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "burn_in_steps": args.burn_in_steps,
        "samples": int(flattened["qpos"].shape[0]),
        "falls": int(np.sum(fallen)),
        "source_replay": {
            "episodes": args.num_envs,
            "steps": replay_steps,
            "falls": replay_falls,
        },
        "command_ranges": {
            "vx": [-0.5, 1.0],
            "vy": [-0.5, 0.5],
            "yaw": [-0.5, 0.5],
        },
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": _sha256(corpus_path),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
