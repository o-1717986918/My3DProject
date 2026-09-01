#!/usr/bin/env python3
"""Train K2 ball outcomes while retaining fixed Apollo contact recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping

from brax.training.agents.ppo import train as ppo
import jax
from mujoco_playground._src import wrapper

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.soccer_ball_motion_env import (
    BallConditionedSoccerMotionTracking,
    DEFAULT_CONTRACT,
)
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.training_dashboard import TrainingDashboard
from tools.train_soccer_motion import (
    _corpus_manifest,
    _effective_timesteps,
    _external_new_directory,
    _git_revision,
    _json_value,
    _sha256,
    _tree_sha256,
)


def _load_bootstrap_gate(
    report_path: Path, restore_checkpoint: Path
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    if not restore_checkpoint.is_dir():
        raise FileNotFoundError(restore_checkpoint)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("purpose") != "k2_zero_row_ball_target_checkpoint_transfer":
        raise ValueError("bootstrap report has the wrong purpose")
    if report.get("status") != "complete" or not report.get("parity", {}).get(
        "passed", False
    ):
        raise ValueError("bootstrap transfer did not complete its parity gate")
    reported_checkpoint = Path(report.get("target_checkpoint", ""))
    if reported_checkpoint.resolve() != restore_checkpoint.resolve():
        raise ValueError("restore checkpoint differs from the bootstrap report")
    observed_hash = _tree_sha256(restore_checkpoint)
    if observed_hash != report.get("target_checkpoint_tree_sha256"):
        raise ValueError("bootstrap checkpoint tree hash differs")
    if report["parity"]["policy_max_abs"] > report["parity"]["required_max_abs"]:
        raise ValueError("bootstrap actor parity exceeds its declared threshold")
    if report["parity"]["value_max_abs"] > report["parity"]["required_max_abs"]:
        raise ValueError("bootstrap critic parity exceeds its declared threshold")
    return {
        "report": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "checkpoint_tree_sha256": observed_hash,
        "source_checkpoint_tree_sha256": report[
            "source_checkpoint_tree_sha256"
        ],
        "parity": report["parity"],
        "authorization_scope": "k2_ball_training_initialization_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--bootstrap-report", type=Path, required=True)
    parser.add_argument("--restore-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--profile",
        choices=(
            "soccer_ball_motion_smoke_v1",
            "soccer_ball_motion_residual_v1",
        ),
        default="soccer_ball_motion_residual_v1",
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-timesteps", type=int, default=196_608)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-evals", type=int, default=3)
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260990)
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--target-angle-degrees", type=float, default=0.0)
    args = parser.parse_args()
    if min(
        args.num_timesteps,
        args.num_envs,
        args.num_evals,
        args.num_eval_envs,
    ) < 1:
        raise ValueError("training counts must be positive")
    if args.target_distance <= 0.0:
        raise ValueError("target distance must be positive")

    run_dir = _external_new_directory(args.run_dir)
    revision = _git_revision()
    bootstrap_gate = _load_bootstrap_gate(
        args.bootstrap_report, args.restore_checkpoint
    )
    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    if (profile.batch_size * profile.num_minibatches) % args.num_envs:
        raise ValueError(
            "num_envs must divide batch_size * num_minibatches "
            f"({profile.batch_size * profile.num_minibatches})"
        )
    optimizer_step_timesteps = (
        profile.batch_size * profile.num_minibatches * profile.unroll_length
    )
    (
        effective_timesteps,
        evaluation_intervals,
        optimizer_steps_per_interval,
    ) = _effective_timesteps(
        args.num_timesteps,
        optimizer_step_timesteps=optimizer_step_timesteps,
        num_evals=args.num_evals,
    )

    corpus = load_soccer_motion_corpus(args.corpus_root)
    angle_rad = args.target_angle_degrees * 3.141592653589793 / 180.0
    overrides = {
        "impl": args.impl,
        "naconmax": max(2048, 8 * args.num_envs),
        "target_distance_range": [args.target_distance, args.target_distance],
        "target_angle_range": [angle_rad, angle_rad],
    }
    train_env = BallConditionedSoccerMotionTracking(
        corpus, config_overrides=overrides, contract=contract
    )
    eval_env = BallConditionedSoccerMotionTracking(
        corpus, config_overrides=overrides, contract=contract, prefix="eval_"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    progress_path = run_dir / "progress.jsonl"
    dashboard_path = run_dir / "tensorboard"
    manifest_path = run_dir / "run-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": "k2_fixed_motion_ball_target_residual_training",
        "promotable": False,
        "promotion_blocker": (
            "requires exact-CPU fixed-2m contact/recovery and target gate"
        ),
        "policy_contract": contract.policy_name,
        "policy_contract_path": str(args.contract.resolve()),
        "policy_contract_sha256": _sha256(args.contract),
        "profile": profile.__dict__,
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "git_revision": revision,
        "seed": args.seed,
        "requested_timesteps": args.num_timesteps,
        "optimizer_step_timesteps": optimizer_step_timesteps,
        "evaluation_intervals": evaluation_intervals,
        "optimizer_steps_per_evaluation_interval": optimizer_steps_per_interval,
        "effective_timesteps": effective_timesteps,
        "num_envs": args.num_envs,
        "num_eval_envs": args.num_eval_envs,
        "environment_config": train_env._config.to_dict(),
        "corpus_root": str(args.corpus_root.resolve()),
        "corpus": _corpus_manifest(corpus),
        "restore_checkpoint": str(args.restore_checkpoint.resolve()),
        "bootstrap_gate": bootstrap_gate,
        "curriculum": {
            "motion": int(train_env._config.fixed_motion_index),
            "start_frame_min": int(train_env._config.fixed_start_frame_min),
            "start_frame_max": int(train_env._config.fixed_start_frame_max),
            "target_distance_m": args.target_distance,
            "target_angle_degrees": args.target_angle_degrees,
            "post_contact_controller": "apollo_zero_command_walk",
            "post_contact_recovery_steps": int(
                train_env._config.post_contact_recovery_steps
            ),
        },
        "visualization": {
            "format": "tensorboard_event",
            "log_dir": str(dashboard_path.resolve()),
            "launch": f"tensorboard --logdir {dashboard_path.resolve()} --port 6006",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dashboard = TrainingDashboard(dashboard_path)
    last_num_steps = 0

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
        nonlocal last_num_steps
        last_num_steps = num_steps
        wall_time_unix = time.time()
        row = {
            "num_steps": num_steps,
            "wall_time_unix": wall_time_unix,
            **{key: _json_value(value) for key, value in metrics.items()},
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        dashboard.write(num_steps, metrics, wall_time_unix=wall_time_unix)
        print(json.dumps(row, sort_keys=True), flush=True)

    started = time.monotonic()
    try:
        _, _, final_metrics = ppo.train(
            environment=train_env,
            eval_env=eval_env,
            num_timesteps=effective_timesteps,
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
            bootstrap_on_timeout=False,
            network_factory=profile.network_factory(),
            seed=args.seed,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            deterministic_eval=True,
            run_evals=True,
            progress_fn=progress,
            restore_checkpoint_path=str(args.restore_checkpoint),
            save_checkpoint_path=str(checkpoint_dir),
        )
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "observed_final_timesteps": last_num_steps,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        dashboard.close()
    manifest.update(
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "observed_final_timesteps": last_num_steps,
            "timestep_accounting_passed": last_num_steps == effective_timesteps,
            "final_metrics": {
                key: _json_value(value) for key, value in final_metrics.items()
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
