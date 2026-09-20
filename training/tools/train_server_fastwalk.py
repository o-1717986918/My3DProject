#!/usr/bin/env python3
"""Conservatively continue FastWalkV2 from projected real-match Walk states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import jax
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src import wrapper
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.server_run_env import ServerFastWalkRun
from my3d_rl.training_schedule import compatible_num_evals, effective_timesteps


REPOSITORY_ROOT = Path(__file__).parents[2]
CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v2.yaml"
PROFILE_NAME = "legacy_phase_soccer_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    value = jax.device_get(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def fastwalk_overrides(
    *, impl: str, graph_mode: str, num_envs: int
) -> dict[str, object]:
    """Match the deployed forward command without making falling taboo."""
    return {
        "impl": impl,
        "warp_graph_mode": graph_mode,
        "naconmax": max(2048, 16 * num_envs),
        "action_clip": 10.0,
        "episode_length": 150,
        "use_fixed_command": True,
        "fixed_command": [1.5, 0.0, 0.0],
        "gait_frequency": [1.75, 1.75],
        "stand_probability": 0.0,
        "command_resample_steps": 150,
        "reset_joint_noise": 0.0,
        "reset_joint_velocity_noise": 0.0,
        "reset_policy_action_noise": 0.0,
        "reset_root_velocity_noise": 0.0,
        "reset_yaw_range": 0.0,
        "push_enable": False,
        "action_delay_max_steps": 0,
        "reward.tracking_linear": 8.0,
        "reward.tracking_yaw": 6.0,
        "reward.upright": 3.0,
        "reward.height": 2.0,
        "reward.alive": 0.75,
        "reward.lateral_tracking": -10.0,
        "reward.yaw_rate_error": -8.0,
        "reward.path_lateral": -12.0,
        "reward.heading_drift": -8.0,
        "reward.vertical_velocity": -0.50,
        "reward.angular_xy": -0.45,
        "reward.action_rate": -0.05,
        "reward.action_acceleration": -0.02,
        "reward.foot_slip": -0.035,
        "reward.pose": -0.04,
        "reward.fall": -20.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-corpus", type=Path, required=True)
    parser.add_argument("--restore-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument(
        "--warp-graph-mode",
        choices=("auto", "none", "jax", "warp", "warp_staged"),
        default="auto",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=196608)
    parser.add_argument("--num-eval-envs", type=int, default=8)
    parser.add_argument("--num-evals", type=int, default=2)
    parser.add_argument("--near-ball-probability", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if (
        not args.reset_corpus.is_file()
        or not args.restore_checkpoint.is_dir()
        or not args.run_dir.is_absolute()
        or run_dir.is_relative_to(REPOSITORY_ROOT.resolve())
        or run_dir.exists()
        or args.num_envs < 1 or args.num_eval_envs < 1
        or not 0.0 <= args.near_ball_probability <= 1.0
        or (args.impl != "warp" and args.warp_graph_mode != "auto")
    ):
        raise ValueError("invalid source/checkpoint/run directory or training options")
    corpus_manifest = json.loads(
        (args.reset_corpus.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        corpus_manifest.get("purpose")
        != "server_projected_phase_v2_fast_walk_reset_states"
        or corpus_manifest.get("corpus_sha256") != sha256(args.reset_corpus)
        or corpus_manifest.get("contract_sha256") != sha256(CONTRACT)
    ):
        raise ValueError("reset corpus manifest/hash/contract mismatch")
    profile = get_ppo_profile(PROFILE_NAME)
    contract = load_policy_contract(CONTRACT)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("checkpoint profile does not match phase-v2 contract")
    epoch_size = (
        profile.batch_size * profile.num_minibatches * profile.unroll_length
    )
    if epoch_size % args.num_envs:
        raise ValueError("num-envs must divide one PPO epoch")
    steps = effective_timesteps(args.num_timesteps, epoch_size)
    evals = compatible_num_evals(steps, epoch_size, args.num_evals)
    overrides = fastwalk_overrides(
        impl=args.impl, graph_mode=args.warp_graph_mode, num_envs=args.num_envs
    )
    train_env = ServerFastWalkRun(
        args.reset_corpus, split=0,
        near_ball_probability=args.near_ball_probability,
        config_overrides=overrides, contract=contract,
    )
    eval_env = ServerFastWalkRun(
        args.reset_corpus, split=1,
        near_ball_probability=None,
        config_overrides=overrides, contract=contract,
    )
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run-manifest.json"
    progress_path = run_dir / "progress.jsonl"
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "server_state_fast_walk_continuation",
        "status": "running",
        "promotable": False,
        "git_revision": revision,
        "reset_corpus": str(args.reset_corpus.resolve()),
        "reset_corpus_sha256": sha256(args.reset_corpus),
        "restore_checkpoint": str(args.restore_checkpoint.resolve()),
        "profile": PROFILE_NAME,
        "contract": contract.policy_name,
        "environment_config": train_env._config.to_dict(),
        "train_entries": corpus_manifest["train_entries"],
        "validation_entries": corpus_manifest["validation_entries"],
        "near_ball_probability_train": args.near_ball_probability,
        "near_ball_probability_validation": eval_env._near_probability,
        "implementation": args.impl,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "num_envs": args.num_envs,
        "requested_timesteps": args.num_timesteps,
        "effective_timesteps": steps,
        "seed": args.seed,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
        row = {"num_steps": num_steps, "wall_time_unix": time.time()}
        row.update({key: json_value(value) for key, value in metrics.items()})
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    started = time.monotonic()
    try:
        _, _, final_metrics = ppo.train(
            environment=train_env,
            eval_env=eval_env,
            num_timesteps=steps,
            num_envs=args.num_envs,
            episode_length=train_env._config.episode_length,
            action_repeat=1,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            learning_rate=profile.learning_rate,
            entropy_cost=profile.entropy_cost,
            discounting=profile.discounting,
            unroll_length=profile.unroll_length,
            batch_size=profile.batch_size,
            num_minibatches=profile.num_minibatches,
            num_updates_per_batch=profile.num_updates_per_batch,
            normalize_observations=profile.normalize_observations,
            reward_scaling=1.0,
            clipping_epsilon=0.2,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            desired_kl=profile.desired_kl,
            learning_rate_schedule=(
                "ADAPTIVE_KL" if profile.adaptive_kl else None
            ),
            learning_rate_schedule_min_lr=profile.learning_rate_min,
            learning_rate_schedule_max_lr=profile.learning_rate_max,
            bootstrap_on_timeout=False,
            network_factory=profile.network_factory(),
            seed=args.seed,
            num_evals=evals,
            num_eval_envs=args.num_eval_envs,
            deterministic_eval=True,
            run_evals=True,
            progress_fn=progress,
            restore_checkpoint_path=str(args.restore_checkpoint),
            save_checkpoint_path=str(run_dir / "checkpoints"),
        )
    except BaseException as exc:
        manifest.update({
            "status": "failed",
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    manifest.update({
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "final_metrics": json_value(final_metrics),
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "effective_timesteps": steps,
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
