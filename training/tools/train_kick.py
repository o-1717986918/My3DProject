#!/usr/bin/env python3
"""Train a resumable guarded kick correction above a frozen base policy."""

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

from my3d_rl.contract import PolicyContract, load_policy_contract
from my3d_rl.kick_env import DirectionalKick, TRANSITION_CONTRACT, default_config
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


def _load_teacher_table(
    path: Path,
    contract: PolicyContract,
    *,
    condition_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in source["records"]
        if bool(record["accepted"])
        and record["mode"] == "pass"
        and abs(float(record["distance_m"]) - 2.0) < 1.0e-9
        and abs(float(record["angle_deg"])) < 1.0e-9
        and (
            condition_index is None
            or int(record["condition_index"]) == condition_index
        )
    ]
    if not records:
        raise ValueError("teacher manifest has no accepted 2 m forward-pass records")
    config = default_config()
    times = np.arange(config.episode_length, dtype=np.float64) * config.ctrl_dt
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


def _load_transition_corpus(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"transition corpus manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "kick_policy_v3_walk_to_kick_transition_corpus":
        raise ValueError("transition corpus manifest has the wrong purpose")
    if manifest.get("npz_sha256") != _sha256(path):
        raise ValueError("transition corpus NPZ hash does not match its manifest")
    with np.load(path, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "split", "rollout_id", "phase_bucket"}
        if not required <= set(archive.files):
            raise ValueError("transition corpus is missing required arrays")
        qpos = np.asarray(archive["qpos"], dtype=np.float32)
        qvel = np.asarray(archive["qvel"], dtype=np.float32)
        split = np.asarray(archive["split"], dtype=np.uint8)
        rollout_id = np.asarray(archive["rollout_id"], dtype=np.int32)
        phase_bucket = np.asarray(archive["phase_bucket"], dtype=np.int32)
    if (
        qpos.ndim != 2
        or qvel.ndim != 2
        or split.shape != (qpos.shape[0],)
        or rollout_id.shape != split.shape
        or phase_bucket.shape != split.shape
        or qvel.shape[0] != qpos.shape[0]
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
    ):
        raise ValueError("transition corpus arrays have incompatible shapes")
    if len(set(rollout_id.tolist())) != rollout_id.size:
        raise ValueError("transition corpus contains duplicate rollout IDs")
    if not set(split.tolist()) <= {0, 1}:
        raise ValueError("transition corpus split must contain only train/validation")
    train = split == 0
    validation = split == 1
    if np.count_nonzero(train) < 2 or np.count_nonzero(validation) < 2:
        raise ValueError("transition corpus needs at least two train and validation rows")
    train_ids = set(rollout_id[train].tolist())
    validation_ids = set(rollout_id[validation].tolist())
    if train_ids & validation_ids:
        raise ValueError("transition corpus leaks rollout IDs across splits")
    metadata = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "npz": str(path.resolve()),
        "npz_sha256": _sha256(path),
        "train_entries": int(np.count_nonzero(train)),
        "validation_entries": int(np.count_nonzero(validation)),
        "train_phase_buckets": sorted(set(phase_bucket[train].tolist())),
        "validation_phase_buckets": sorted(set(phase_bucket[validation].tolist())),
        "teacher_condition_index": int(manifest["teacher_condition_index"]),
    }
    return qpos[train], qvel[train], qpos[validation], qvel[validation], metadata


def _load_parity_report(path: Path, implementation: str) -> dict[str, Any]:
    """Validate the CPU/accelerated-backend equivalence gate for formal training."""
    parity = json.loads(path.read_text(encoding="utf-8"))
    if parity.get("purpose") != "kick_identical_control_cpu_mjx_parity":
        raise ValueError("parity report has the wrong purpose")
    if parity.get("accelerated_implementation") != implementation:
        raise ValueError("parity report backend does not match --impl")
    if not bool(parity.get("summary", {}).get("parity_gate_passed")):
        raise ValueError("parity report did not pass")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "summary": parity["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("--contract", type=Path, default=TRANSITION_CONTRACT)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument(
        "--allow-unverified-backend-smoke",
        action="store_true",
        help="allow a non-promotable run without a passing CPU/backend report",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--num-evals", type=int, default=8)
    parser.add_argument("--num-eval-envs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=1.0e-4)
    parser.add_argument("--init-noise-std", type=float, default=0.05)
    parser.add_argument("--correction-scale", type=float, default=0.1)
    parser.add_argument("--gate-success-reward", type=float, default=20.0)
    parser.add_argument("--fall-penalty", type=float, default=20.0)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument(
        "--base-kick-onnx",
        type=Path,
        help="freeze a kick-policy-v3 behavior clone under the PPO correction",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    parity_metadata: dict[str, Any] | None = None
    if args.parity_report is not None:
        parity_metadata = _load_parity_report(args.parity_report, args.impl)
    elif not args.allow_unverified_backend_smoke:
        raise ValueError(
            "formal training requires --parity-report; use the explicit smoke "
            "override only for non-promotable diagnostics"
        )

    if not args.run_dir.is_absolute() or args.run_dir.is_relative_to(Path.cwd()):
        raise ValueError("run-dir must be an absolute path outside the repository")
    if args.base_kick_onnx is not None and not args.base_kick_onnx.is_file():
        raise FileNotFoundError(args.base_kick_onnx)
    if (
        args.learning_rate <= 0.0
        or args.entropy_cost < 0.0
        or not 0.0 < args.init_noise_std <= 0.2
        or not 0.0 < args.correction_scale <= 0.1
        or args.gate_success_reward <= 0.0
        or args.fall_penalty <= 0.0
    ):
        raise ValueError("PPO optimization and safety scales are invalid")
    batch_size = 256
    num_minibatches = 4
    unroll_length = 16
    if (batch_size * num_minibatches) % args.num_envs:
        raise ValueError("num-envs must divide batch-size times num-minibatches")
    minimum_timesteps = batch_size * num_minibatches * unroll_length
    effective_timesteps = max(args.num_timesteps, minimum_timesteps)

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3" or contract.observation_size != 98:
        raise ValueError("formal transition training requires kick_policy_v3")
    (
        train_qpos,
        train_qvel,
        validation_qpos,
        validation_qvel,
        transition_metadata,
    ) = _load_transition_corpus(args.transition_corpus)
    trajectories, offsets, condition_indices = _load_teacher_table(
        args.teacher_manifest,
        contract,
        condition_index=transition_metadata["teacher_condition_index"],
    )
    environment_overrides = {
        "impl": args.impl,
        "naconmax": max(2048, 16 * args.num_envs),
        "target_distance_range": [2.0, 2.0],
        "target_angle_range": [0.0, 0.0],
        "fixed_action_mode": 0,
        "fixed_desired_arrival_speed": 0.8,
        "action_scale": (args.correction_scale * KICK_ACTION_SCALE).tolist(),
        "gate_success_reward": args.gate_success_reward,
        "fall_penalty": args.fall_penalty,
    }
    env = DirectionalKick(
        config_overrides=environment_overrides,
        contract=contract,
        teacher_joint_residuals=trajectories,
        teacher_ball_offsets=offsets,
        transition_qpos=train_qpos,
        transition_qvel=train_qvel,
        base_kick_policy_path=args.base_kick_onnx,
    )
    eval_env = DirectionalKick(
        config_overrides=environment_overrides,
        contract=contract,
        teacher_joint_residuals=trajectories,
        teacher_ball_offsets=offsets,
        transition_qpos=validation_qpos,
        transition_qvel=validation_qvel,
        base_kick_policy_path=args.base_kick_onnx,
    )
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        policy_obs_key="state",
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
        "purpose": (
            "formal_base_policy_kick_correction_training"
            if args.base_kick_onnx is not None
            else "formal_teacher_table_kick_correction_training"
        ),
        "promotable": False,
        "promotion_blocker": "requires exact CPU, ONNX, three-seed and server gates",
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
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "teacher_condition_indices": condition_indices,
        "transition_corpus": transition_metadata,
        "num_envs": args.num_envs,
        "requested_timesteps": args.num_timesteps,
        "effective_timesteps": effective_timesteps,
        "seed": args.seed,
        "environment_config": env._config.to_dict(),
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "optimizer": {
            "learning_rate": args.learning_rate,
            "entropy_cost": args.entropy_cost,
            "init_noise_std": args.init_noise_std,
            "correction_scale": args.correction_scale,
            "gate_success_reward": args.gate_success_reward,
            "fall_penalty": args.fall_penalty,
        },
        "restore_checkpoint": (
            str(args.restore_checkpoint) if args.restore_checkpoint else None
        ),
        "base_kick_onnx": (
            str(args.base_kick_onnx.resolve())
            if args.base_kick_onnx is not None
            else None
        ),
        "base_kick_onnx_sha256": (
            _sha256(args.base_kick_onnx)
            if args.base_kick_onnx is not None
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
