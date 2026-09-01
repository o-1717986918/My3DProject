#!/usr/bin/env python3
"""Run resumable PPO for finite PAiD-to-T1 soccer motion tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

from brax.training.agents.ppo import train as ppo
import jax
import numpy as np
from mujoco_playground._src import wrapper

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import PROFILES, get_ppo_profile
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.soccer_motion_env import (
    DEFAULT_CONTRACT,
    FiniteSoccerMotionTracking,
)
from my3d_rl.training_dashboard import TrainingDashboard


STAGES: dict[str, dict[str, Any]] = {
    "reference_track": {
        "reset_joint_noise": 0.002,
        "reset_root_velocity_noise": 0.005,
        "reset_yaw_range": 0.01,
        "action_delay_max_steps": 0,
    },
    "robust_track": {
        "reset_joint_noise": 0.01,
        "reset_root_velocity_noise": 0.03,
        "reset_yaw_range": 0.03,
        "action_delay_max_steps": 1,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    if status:
        raise RuntimeError("formal soccer-motion PPO requires a clean Git tree")
    return revision


def _external_new_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("run directory must be absolute")
    resolved = path.resolve()
    repository_root = Path(__file__).parents[2].resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise ValueError("run directory must stay outside the repository")
    if resolved.exists():
        raise FileExistsError(f"run directory already exists: {resolved}")
    return resolved


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_curriculum_gate(
    comparison_path: Path, restore_checkpoint: Path
) -> dict[str, Any]:
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    if not restore_checkpoint.is_dir():
        raise FileNotFoundError(restore_checkpoint)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("purpose") != "k1_paired_exact_cpu_policy_comparison":
        raise ValueError("curriculum evidence is not a paired exact-CPU comparison")
    if comparison.get("curriculum_advance_passed") is not True:
        raise ValueError("curriculum evidence did not authorize PPO initialization")
    candidate_path = Path(comparison.get("candidate", ""))
    if not candidate_path.is_file():
        raise FileNotFoundError("curriculum candidate evaluation is unavailable")
    candidate_sha256 = _sha256(candidate_path)
    if candidate_sha256 != comparison.get("candidate_sha256"):
        raise ValueError("curriculum candidate evaluation hash differs")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("purpose") != "k1_exact_cpu_fixed_motion_phase_grid":
        raise ValueError("curriculum candidate is not an exact-CPU evaluation")
    evaluated_checkpoint = Path(candidate.get("checkpoint", ""))
    if evaluated_checkpoint.resolve() != restore_checkpoint.resolve():
        raise ValueError(
            "restore checkpoint differs from the curriculum-gated candidate"
        )
    return {
        "comparison": str(comparison_path.resolve()),
        "comparison_sha256": _sha256(comparison_path),
        "candidate_evaluation": str(candidate_path.resolve()),
        "candidate_evaluation_sha256": candidate_sha256,
        "checkpoint_tree_sha256": _tree_sha256(restore_checkpoint),
        "checkpoint_tree_hash_algorithm": (
            "sha256_over_sorted_relative_path_nul_file_sha256_newline"
        ),
        "paired_trials": int(comparison["paired_trials"]),
        "completion_rate_delta": float(comparison["completion_rate_delta"]),
        "mean_survival_fraction_delta": float(
            comparison["mean_survival_fraction_delta"]
        ),
        "curriculum_advance_passed": True,
        "authorization_scope": "ppo_initialization_only_not_runtime_promotion",
    }


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    return array.tolist()


def _corpus_manifest(corpus) -> dict[str, Any]:
    return {
        "motion_count": corpus.motion_count,
        "maximum_frames": corpus.maximum_frames,
        "relative_paths": list(corpus.relative_paths),
        "sha256": list(corpus.sha256),
        "lengths": corpus.lengths.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=DEFAULT_CONTRACT
    )
    parser.add_argument(
        "--profile",
        choices=tuple(
            name for name in PROFILES if name.startswith("soccer_motion_")
        ),
        default="soccer_motion_residual_v1",
    )
    parser.add_argument(
        "--stage", choices=tuple(STAGES), default="reference_track"
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-timesteps", type=int, default=262_144)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-evals", type=int, default=5)
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument(
        "--curriculum-comparison",
        type=Path,
        help=(
            "paired exact-CPU comparison that authorized the restored "
            "checkpoint for PPO initialization"
        ),
    )
    args = parser.parse_args()
    if min(
        args.num_timesteps,
        args.num_envs,
        args.num_evals,
        args.num_eval_envs,
    ) < 1:
        raise ValueError("training counts must be positive")
    if not args.failure_report.is_file():
        raise FileNotFoundError(args.failure_report)
    if (args.restore_checkpoint is None) != (args.curriculum_comparison is None):
        raise ValueError(
            "restore-checkpoint and curriculum-comparison must be supplied together"
        )

    run_dir = _external_new_directory(args.run_dir)
    revision = _git_revision()
    curriculum_gate = (
        _load_curriculum_gate(args.curriculum_comparison, args.restore_checkpoint)
        if args.restore_checkpoint is not None
        and args.curriculum_comparison is not None
        else None
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
    minimum_epoch_timesteps = (
        profile.batch_size * profile.num_minibatches * profile.unroll_length
    )
    effective_timesteps = max(args.num_timesteps, minimum_epoch_timesteps)

    train_corpus = load_soccer_motion_corpus(
        args.corpus_root, failure_report=args.failure_report
    )
    # Evaluation deliberately removes failure-focused sampling. This measures
    # the complete valid phase distribution rather than the training course.
    eval_corpus = load_soccer_motion_corpus(args.corpus_root)
    overrides = {
        "impl": args.impl,
        "naconmax": max(2048, 8 * args.num_envs),
        **STAGES[args.stage],
    }
    train_env = FiniteSoccerMotionTracking(
        train_corpus, config_overrides=overrides, contract=contract
    )
    eval_env = FiniteSoccerMotionTracking(
        eval_corpus, config_overrides=overrides, contract=contract
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    progress_path = run_dir / "progress.jsonl"
    dashboard_path = run_dir / "tensorboard"
    manifest_path = run_dir / "run-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": "k1_finite_multi_motion_residual_tracking",
        "stage": args.stage,
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
        "minimum_epoch_timesteps": minimum_epoch_timesteps,
        "effective_timesteps": effective_timesteps,
        "num_envs": args.num_envs,
        "num_eval_envs": args.num_eval_envs,
        "environment_config": train_env._config.to_dict(),
        "corpus_root": str(args.corpus_root.resolve()),
        "corpus": _corpus_manifest(train_corpus),
        "failure_report": str(args.failure_report.resolve()),
        "failure_report_sha256": _sha256(args.failure_report),
        "training_reset_sampling": "failure_focused_nonperiodic",
        "evaluation_reset_sampling": "uniform_nonperiodic",
        "restore_checkpoint": (
            str(args.restore_checkpoint.resolve())
            if args.restore_checkpoint
            else None
        ),
        "curriculum_gate": curriculum_gate,
        "visualization": {
            "format": "tensorboard_event",
            "log_dir": str(dashboard_path.resolve()),
            "launch": f"tensorboard --logdir {dashboard_path.resolve()} --port 6006",
            "checkpoint_replay": (
                "PYTHONPATH=training python "
                "training/tools/view_soccer_motion_policy.py "
                f"{args.corpus_root.resolve()} --checkpoint <checkpoint> "
                f"--profile {args.profile} --hold"
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dashboard = TrainingDashboard(dashboard_path)

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
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
            desired_kl=profile.desired_kl,
            learning_rate_schedule=(
                "ADAPTIVE_KL" if profile.adaptive_kl else None
            ),
            learning_rate_schedule_min_lr=profile.learning_rate_min,
            learning_rate_schedule_max_lr=profile.learning_rate_max,
            bootstrap_on_timeout=False,
            network_factory=profile.network_factory(),
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
