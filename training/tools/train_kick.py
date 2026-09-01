#!/usr/bin/env python3
"""Train a resumable guarded kick correction policy from a teacher table."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src import wrapper
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_env import DEFAULT_CONTRACT, DirectionalKick, default_config
from my3d_rl.kick_teacher import build_joint_delta_trajectory
from my3d_rl.t1_control import KICK_ACTION_SCALE


def _json_value(value: Any) -> Any:
    try:
        return value.item()
    except (AttributeError, ValueError):
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_teacher_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in source["records"]
        if bool(record["accepted"])
        and record["mode"] == "pass"
        and abs(float(record["distance_m"]) - 2.0) < 1.0e-9
        and abs(float(record["angle_deg"])) < 1.0e-9
    ]
    if not records:
        raise ValueError("teacher manifest has no accepted 2 m forward-pass records")
    config = default_config()
    times = np.arange(config.episode_length, dtype=np.float64) * config.ctrl_dt
    contract = load_policy_contract(DEFAULT_CONTRACT)
    trajectories = np.stack(
        [
            build_joint_delta_trajectory(
                np.asarray(record["parameters"], dtype=np.float64),
                contract,
                times,
            )
            for record in records
        ]
    ).astype(np.float32)
    offsets = np.asarray(
        [
            [float(record["ball_x_offset_m"]), float(record["ball_y_offset_m"])]
            for record in records
        ],
        dtype=np.float32,
    )
    return trajectories, offsets, [int(record["condition_index"]) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--num-evals", type=int, default=8)
    parser.add_argument("--num-eval-envs", type=int, default=32)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.run_dir.is_absolute() or args.run_dir.is_relative_to(Path.cwd()):
        raise ValueError("run-dir must be an absolute path outside the repository")
    batch_size = 256
    num_minibatches = 4
    unroll_length = 16
    if (batch_size * num_minibatches) % args.num_envs:
        raise ValueError("num-envs must divide batch-size times num-minibatches")
    minimum_timesteps = batch_size * num_minibatches * unroll_length
    effective_timesteps = max(args.num_timesteps, minimum_timesteps)

    trajectories, offsets, condition_indices = _load_teacher_table(
        args.teacher_manifest
    )
    environment_overrides = {
        "impl": args.impl,
        "naconmax": max(2048, 16 * args.num_envs),
        "target_distance_range": [2.0, 2.0],
        "target_angle_range": [0.0, 0.0],
        "fixed_action_mode": 0,
        "fixed_desired_arrival_speed": 0.8,
        "action_scale": (0.1 * KICK_ACTION_SCALE).tolist(),
    }
    env = DirectionalKick(
        config_overrides=environment_overrides,
        teacher_joint_residuals=trajectories,
        teacher_ball_offsets=offsets,
    )
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="normal",
        noise_std_type="log",
        init_noise_std=0.05,
        mean_clip_scale=1.0,
        mean_kernel_init_fn=jax.nn.initializers.normal,
        mean_kernel_init_kwargs={"stddev": 0.0},
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    progress_path = args.run_dir / "progress.jsonl"
    manifest_path = args.run_dir / "run-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": "formal_teacher_table_kick_correction_training",
        "promotable": False,
        "promotion_blocker": "requires exact CPU, ONNX, three-seed and server gates",
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "git_revision": _git_revision(),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "teacher_condition_indices": condition_indices,
        "num_envs": args.num_envs,
        "requested_timesteps": args.num_timesteps,
        "effective_timesteps": effective_timesteps,
        "seed": args.seed,
        "environment_config": env._config.to_dict(),
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "restore_checkpoint": (
            str(args.restore_checkpoint) if args.restore_checkpoint else None
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
        row = {
            "num_steps": num_steps,
            "wall_time_unix": time.time(),
            **{key: _json_value(value) for key, value in metrics.items()},
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    started = time.monotonic()
    try:
        _, _, final_metrics = ppo.train(
            environment=env,
            num_timesteps=effective_timesteps,
            num_envs=args.num_envs,
            episode_length=env._config.episode_length,
            action_repeat=1,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            learning_rate=1.0e-4,
            entropy_cost=1.0e-4,
            discounting=0.995,
            unroll_length=unroll_length,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=4,
            normalize_observations=True,
            reward_scaling=1.0,
            clipping_epsilon=0.15,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            bootstrap_on_timeout=False,
            network_factory=network_factory,
            seed=args.seed,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            deterministic_eval=True,
            run_evals=True,
            progress_fn=progress,
            restore_checkpoint_path=(
                str(args.restore_checkpoint) if args.restore_checkpoint else None
            ),
            save_checkpoint_path=str(checkpoint_dir),
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    manifest.update(
        {
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
            "final_metrics": {
                key: _json_value(value) for key, value in final_metrics.items()
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
