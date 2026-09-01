#!/usr/bin/env python3
"""Train the privileged long-horizon approach-and-kick teacher."""

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
from my3d_rl.kick_teacher import build_joint_delta_trajectory
from my3d_rl.striker_env import (
    DEFAULT_CONTRACT,
    LongHorizonStriker,
)


STAGES: dict[str, dict[str, Any]] = {
    "near_ball": {
        "robot_distance_range": [0.42, 0.70],
        "robot_lateral_range": [-0.06, 0.06],
        "robot_yaw_noise_range": [-0.08, 0.08],
        "target_angle_range": [-0.15, 0.15],
        "target_distance_range": [1.5, 3.5],
        "reset_joint_noise": 0.005,
        "reset_root_velocity_noise": 0.01,
    },
    "closed_loop": {
        "robot_distance_range": [0.55, 1.40],
        "robot_lateral_range": [-0.15, 0.15],
        "robot_yaw_noise_range": [-0.20, 0.20],
        "target_angle_range": [-0.261799, 0.261799],
        "target_distance_range": [2.0, 3.5],
        "reset_joint_noise": 0.01,
        "reset_root_velocity_noise": 0.03,
    },
    "robust": {
        "robot_distance_range": [0.55, 2.50],
        "robot_lateral_range": [-0.40, 0.40],
        "robot_yaw_noise_range": [-0.45, 0.45],
        "target_angle_range": [-0.523599, 0.523599],
        "target_distance_range": [2.0, 5.0],
        "reset_joint_noise": 0.02,
        "reset_root_velocity_noise": 0.06,
    },
}

# Several task variables are physically constant until first contact and then
# change abruptly.  An online Welford normalizer assigns them near-zero
# variance before contact, which can saturate the policy as soon as the ball
# moves.  The observation contract already uses bounded SI-scale quantities;
# keep those semantics stable for teacher training and later distillation.
NORMALIZE_OBSERVATIONS = False


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


def _load_parity_report(path: Path, implementation: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("purpose") != "striker_identical_control_cpu_mjx_parity":
        raise ValueError("parity report has the wrong purpose")
    if report.get("accelerated_implementation") != implementation:
        raise ValueError("parity report backend does not match --impl")
    if not bool(report.get("summary", {}).get("parity_gate_passed")):
        raise ValueError("parity report did not pass")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "summary": report["summary"],
    }


def _load_kick_prior(
    path: Path,
    contract,
    *,
    condition_index: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    records = source.get("records")
    if records is None:
        parameters = source.get("parameters")
        selected_condition = None
    else:
        eligible = [
            record
            for record in records
            if bool(record.get("accepted"))
            and record.get("mode") == "pass"
            and abs(float(record.get("distance_m", -1.0)) - 2.0) < 1.0e-9
            and abs(float(record.get("angle_deg", 999.0))) < 1.0e-9
            and (
                condition_index is None
                or int(record.get("condition_index", -1)) == condition_index
            )
        ]
        if not eligible:
            raise ValueError("kick-prior manifest has no requested accepted record")
        selected = min(
            eligible,
            key=lambda record: (
                float(record["ball_x_offset_m"]) + 0.01
            ) ** 2
            + (float(record["ball_y_offset_m"]) + 0.04) ** 2,
        )
        parameters = selected.get("parameters")
        selected_condition = int(selected["condition_index"])
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (14,) or not np.isfinite(parameters).all():
        raise ValueError("kick-prior parameters must be 14 finite values")
    duration_s = float(source.get("spec", {}).get("duration_s", 1.2))
    if not 0.5 <= duration_s <= 3.0:
        raise ValueError("kick-prior duration must be in [0.5, 3.0] seconds")
    times = np.arange(round(duration_s * 50.0), dtype=np.float64) / 50.0
    trajectory = build_joint_delta_trajectory(parameters, contract, times).astype(
        np.float32
    )
    return trajectory, {
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "purpose": source.get("purpose"),
        "condition_index": selected_condition,
        "duration_s": duration_s,
        "steps": int(trajectory.shape[0]),
        "parameters": parameters.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kick_prior_manifest", type=Path)
    parser.add_argument("--kick-prior-condition-index", type=int, default=1)
    parser.add_argument(
        "--stage", choices=tuple(STAGES), default="near_ball"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument(
        "--allow-unverified-backend-smoke",
        action="store_true",
        help="allow diagnostics that are explicitly barred from promotion",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-timesteps", type=int, default=2_097_152)
    parser.add_argument("--seed", type=int, default=11_101)
    parser.add_argument("--num-evals", type=int, default=8)
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=1.0e-4)
    parser.add_argument("--init-noise-std", type=float, default=0.05)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.parity_report is not None:
        parity_metadata = _load_parity_report(args.parity_report, args.impl)
    elif args.allow_unverified_backend_smoke:
        parity_metadata = None
    else:
        raise ValueError(
            "formal teacher training requires --parity-report; use the explicit "
            "smoke override only for non-promotable diagnostics"
        )
    if not args.run_dir.is_absolute() or args.run_dir.is_relative_to(Path.cwd()):
        raise ValueError("run-dir must be an absolute path outside the repository")
    if args.restore_checkpoint is not None and not args.restore_checkpoint.exists():
        raise FileNotFoundError(args.restore_checkpoint)
    if (
        args.num_envs < 1
        or args.num_eval_envs < 1
        or args.num_evals < 1
        or args.learning_rate <= 0.0
        or args.entropy_cost < 0.0
        or not 0.0 < args.init_noise_std <= 0.2
    ):
        raise ValueError("invalid PPO optimization configuration")

    batch_size = 256
    num_minibatches = 4
    unroll_length = 32
    if (batch_size * num_minibatches) % args.num_envs:
        raise ValueError("num-envs must divide batch-size times num-minibatches")
    minimum_timesteps = batch_size * num_minibatches * unroll_length
    effective_timesteps = max(args.num_timesteps, minimum_timesteps)

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "striker_policy_v1":
        raise ValueError("teacher training requires striker_policy_v1")
    kick_prior, kick_prior_metadata = _load_kick_prior(
        args.kick_prior_manifest,
        contract,
        condition_index=args.kick_prior_condition_index,
    )
    environment_overrides = {
        **STAGES[args.stage],
        "impl": args.impl,
        "naconmax": max(2048, 16 * args.num_envs),
        "fixed_action_mode": 0,
        "fixed_desired_arrival_speed": 0.8,
    }
    env = LongHorizonStriker(
        config_overrides=environment_overrides,
        contract=contract,
        kick_prior_joint_residuals=kick_prior,
    )
    eval_overrides = {
        **environment_overrides,
        # Evaluation always covers at least the complete closed-loop stage so
        # a narrow curriculum cannot look promotable by construction.
        **(
            STAGES["closed_loop"]
            if args.stage == "near_ball"
            else STAGES[args.stage]
        ),
    }
    eval_env = LongHorizonStriker(
        config_overrides=eval_overrides,
        contract=contract,
        kick_prior_joint_residuals=kick_prior,
    )
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        policy_obs_key="teacher_state",
        value_obs_key="privileged_state",
        distribution_type="normal",
        noise_std_type="log",
        init_noise_std=args.init_noise_std,
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
        "purpose": "privileged_long_horizon_striker_teacher_training",
        "promotable": False,
        "promotion_blocker": (
            "teacher is training-only; requires exact CPU gates and history "
            "student distillation before competition deployment"
        ),
        "stage": args.stage,
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "backend_parity": parity_metadata,
        "unverified_backend_smoke": args.allow_unverified_backend_smoke,
        "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "git_revision": _git_revision(),
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "kick_prior": kick_prior_metadata,
        "num_envs": args.num_envs,
        "requested_timesteps": args.num_timesteps,
        "effective_timesteps": effective_timesteps,
        "seed": args.seed,
        "environment_config": env._config.to_dict(),
        "evaluation_environment_config": eval_env._config.to_dict(),
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "optimizer": {
            "learning_rate": args.learning_rate,
            "entropy_cost": args.entropy_cost,
            "init_noise_std": args.init_noise_std,
            "batch_size": batch_size,
            "num_minibatches": num_minibatches,
            "unroll_length": unroll_length,
            "normalize_observations": NORMALIZE_OBSERVATIONS,
        },
        "restore_checkpoint": (
            str(args.restore_checkpoint.resolve())
            if args.restore_checkpoint is not None
            else None
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
            eval_env=eval_env,
            num_timesteps=effective_timesteps,
            num_envs=args.num_envs,
            episode_length=env._config.episode_length,
            action_repeat=1,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            learning_rate=args.learning_rate,
            entropy_cost=args.entropy_cost,
            discounting=0.995,
            unroll_length=unroll_length,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=4,
            normalize_observations=NORMALIZE_OBSERVATIONS,
            reward_scaling=1.0,
            clipping_epsilon=0.15,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            bootstrap_on_timeout=True,
            network_factory=network_factory,
            seed=args.seed,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            deterministic_eval=True,
            run_evals=True,
            progress_fn=progress,
            restore_checkpoint_path=(
                str(args.restore_checkpoint)
                if args.restore_checkpoint is not None
                else None
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
