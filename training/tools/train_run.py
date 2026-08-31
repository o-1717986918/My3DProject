#!/usr/bin/env python3
"""Train a resumable PPO fast-locomotion policy on exact RCSS assets."""

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
import jax.numpy as jp
from brax.training import types as brax_types
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from mujoco_playground._src import wrapper
import numpy as np
import onnxruntime as ort

from my3d_rl.legacy_policy import load_onnx_teacher_params
from my3d_rl.contract import load_policy_contract
from my3d_rl.motion_reference import validate_motion_reference
from my3d_rl.ppo_profile import PROFILES, get_ppo_profile
from my3d_rl.run_env import DirectionalRun


STAGES: dict[str, dict[str, Any]] = {
    "balance": {
        "use_fixed_command": True,
        "fixed_command": [0.0, 0.0, 0.0],
        "reset_joint_noise": 0.02,
        "reset_root_velocity_noise": 0.03,
        "reset_yaw_range": 0.05,
        "push_enable": False,
        "reward.tracking_linear": 1.0,
        "reward.tracking_yaw": 0.5,
        "reward.upright": 2.0,
        "reward.height": 2.0,
        "reward.alive": 1.0,
        "reward.vertical_velocity": -0.5,
        "reward.angular_xy": -0.5,
        "reward.action_rate": -0.2,
        "reward.action_acceleration": -0.05,
        "reward.joint_velocity": -0.0002,
        "reward.pose": -0.5,
        "reward.fall": -100.0,
    },
    "stand": {
        "lin_vel_x": [0.0, 0.6],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_yaw": [0.0, 0.0],
        "stand_probability": 0.35,
        "push_enable": False,
        "reward.upright": 1.0,
        "reward.height": 1.5,
        "reward.alive": 0.5,
    },
    "fast_walk": {
        "lin_vel_x": [0.0, 1.2],
        "lin_vel_y": [-0.15, 0.15],
        "ang_vel_yaw": [-0.25, 0.25],
        "stand_probability": 0.20,
        "push_enable": False,
    },
    "run": {
        "lin_vel_x": [0.4, 1.8],
        "lin_vel_y": [-0.20, 0.20],
        "ang_vel_yaw": [-0.35, 0.35],
        "stand_probability": 0.15,
        "reset_joint_noise": 0.04,
        "push_enable": True,
        "push_interval_steps": 175,
        "action_delay_max_steps": 1,
        "reward.tracking_yaw": 5.0,
        "reward.flight": 5.0,
        "reward.single_support": 0.5,
    },
    "phase_run": {
        "lin_vel_x": [0.8, 1.8],
        "lin_vel_y": [-0.10, 0.10],
        "ang_vel_yaw": [-0.20, 0.20],
        "gait_frequency": [1.5, 2.0],
        "swing_period": 0.20,
        "stand_probability": 0.10,
        "reset_joint_noise": 0.03,
        "push_enable": False,
        "action_delay_max_steps": 0,
        "reward.tracking_yaw": 5.0,
        "reward.flight": 1.0,
        "reward.single_support": 0.25,
        "reward.phase_swing": 2.0,
        "reward.lateral_tracking": -2.0,
        "reward.yaw_rate_error": -2.0,
    },
    "straight_recovery": {
        "use_fixed_command": True,
        "fixed_command": [1.5, 0.0, 0.0],
        "gait_frequency": [1.75, 1.75],
        "swing_period": 0.20,
        "stand_probability": 0.0,
        "reset_joint_noise": 0.02,
        "reset_root_velocity_noise": 0.05,
        "reset_yaw_range": 0.10,
        "push_enable": False,
        "action_delay_max_steps": 0,
        "reward.tracking_yaw": 10.0,
        "reward.flight": 0.0,
        "reward.single_support": 0.25,
        "reward.phase_swing": 2.0,
        "reward.lateral_tracking": -20.0,
        "reward.yaw_rate_error": -20.0,
    },
    "motion_track": {
        "use_fixed_command": True,
        "fixed_command": [1.8, 0.0, 0.0],
        "gait_frequency": [1.85, 1.85],
        "swing_period": 0.20,
        "stand_probability": 0.0,
        "reset_joint_noise": 0.015,
        "reset_root_velocity_noise": 0.03,
        "reset_yaw_range": 0.05,
        "reference_init_probability": 0.20,
        "push_enable": False,
        "action_delay_max_steps": 0,
        "reward.tracking_linear": 1.0,
        "reward.tracking_yaw": 2.0,
        "reward.flight": 0.0,
        "reward.single_support": 0.25,
        "reward.phase_swing": 0.5,
        "reward.motion_joint": 8.0,
        "reward.motion_joint_velocity": 1.0,
        "reward.motion_contact": 2.0,
        "reward.motion_action": 4.0,
        "reward.alive": 0.5,
        "reward.fall": -100.0,
        "reward.lateral_tracking": -4.0,
        "reward.yaw_rate_error": -4.0,
        "reward.pose": 0.0,
    },
    "motion_straight": {
        "use_fixed_command": True,
        "fixed_command": [1.8, 0.0, 0.0],
        "gait_frequency": [1.515, 1.515],
        "swing_period": 0.24,
        "stand_probability": 0.0,
        "reset_joint_noise": 0.015,
        "reset_root_velocity_noise": 0.03,
        "reset_yaw_range": 0.05,
        "reference_init_probability": 0.10,
        "foot_contact_tolerance": 0.0,
        "push_enable": False,
        "action_delay_max_steps": 0,
        "reward.tracking_linear": 2.0,
        "reward.tracking_yaw": 10.0,
        "reward.flight": 0.5,
        "reward.single_support": 0.5,
        "reward.phase_swing": 1.0,
        "reward.motion_joint": 8.0,
        "reward.motion_joint_velocity": 1.0,
        "reward.motion_contact": 3.0,
        "reward.motion_action": 4.0,
        "reward.lateral_tracking": -40.0,
        "reward.yaw_rate_error": -40.0,
        "reward.alive": 0.5,
        "reward.fall": -100.0,
        "reward.pose": 0.0,
    },
    "reference_residual": {
        "use_fixed_command": True,
        "fixed_command": [1.8, 0.0, 0.0],
        "stand_probability": 0.0,
        "reset_joint_noise": 0.005,
        "reset_root_velocity_noise": 0.01,
        "reset_yaw_range": 0.02,
        "reference_init_probability": 1.0,
        "foot_contact_tolerance": 0.0,
        "push_enable": False,
        "action_delay_max_steps": 0,
        "reward.tracking_linear": 2.0,
        "reward.tracking_yaw": 6.0,
        "reward.flight": 0.0,
        "reward.single_support": 0.0,
        "reward.phase_swing": 0.0,
        "reward.motion_joint": 10.0,
        "reward.motion_joint_velocity": 2.0,
        "reward.motion_contact": 4.0,
        "reward.motion_action": 2.0,
        "reward.lateral_tracking": -20.0,
        "reward.yaw_rate_error": -20.0,
        "reward.alive": 0.5,
        "reward.fall": -100.0,
        "reward.pose": 0.0,
    },
    "omni": {
        "lin_vel_x": [0.0, 1.8],
        "lin_vel_y": [-0.40, 0.40],
        "ang_vel_yaw": [-0.60, 0.60],
        "stand_probability": 0.20,
        "reset_joint_noise": 0.05,
        "push_enable": True,
        "push_interval_steps": 150,
        "action_delay_max_steps": 1,
    },
}


def _json_value(value: Any) -> Any:
    try:
        return value.item()
    except (AttributeError, ValueError):
        return value


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_restore_params(
    env: DirectionalRun, profile: Any, model_path: Path, seed: int
) -> tuple[tuple[Any, Any, Any], float]:
    network_factory = profile.network_factory()
    networks = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )
    policy_key, value_key = jax.random.split(jax.random.PRNGKey(seed))
    policy_params = networks.policy_network.init(policy_key)
    policy_params = load_onnx_teacher_params(policy_params, model_path)
    value_params = networks.value_network.init(value_key)
    actor_size = env.observation_size["state"][-1]
    privileged_size = env.observation_size["privileged_state"][-1]
    observation_spec = {
        "state": jax.ShapeDtypeStruct((actor_size,), jp.float32),
        "privileged_state": jax.ShapeDtypeStruct((privileged_size,), jp.float32),
    }
    normalizer_params = running_statistics.init_state(observation_spec)

    parity_rng = np.random.default_rng(seed + 101)
    observations = parity_rng.normal(0.0, 0.25, (128, actor_size)).astype(np.float32)
    jax_actions = networks.policy_network.apply(
        normalizer_params,
        policy_params,
        {
            "state": jp.asarray(observations),
            "privileged_state": jp.zeros((128, privileged_size), dtype=jp.float32),
        },
    )[0]
    session = ort.InferenceSession(
        str(model_path.resolve()), providers=["CPUExecutionProvider"]
    )
    onnx_actions = session.run(
        None, {session.get_inputs()[0].name: observations[:, :78]}
    )[0]
    max_error = float(np.max(np.abs(np.asarray(jax_actions) - onnx_actions)))
    if max_error > 2.0e-5:
        raise ValueError(f"teacher import parity error {max_error:.3e} exceeds 2e-5")
    return (normalizer_params, policy_params, value_params), max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), default="fast_walk")
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=1_048_576)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=8)
    parser.add_argument(
        "--network-profile",
        choices=tuple(PROFILES),
        default="t1_tanh_v1",
    )
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--bootstrap-onnx", type=Path)
    parser.add_argument("--motion-reference", type=Path)
    parser.add_argument("--reference-phase-weights", type=Path)
    parser.add_argument(
        "--fixed-vx",
        type=float,
        help="override reference_residual forward speed for a staged curriculum",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.run_dir.is_absolute() or args.run_dir.is_relative_to(Path.cwd()):
        raise ValueError("run-dir must live outside the repository")
    profile = get_ppo_profile(args.network_profile)
    contract_path = (
        Path(__file__).parents[1] / "contracts" / f"{profile.policy_contract}.yaml"
    )
    contract = load_policy_contract(contract_path)
    if args.restore_checkpoint and args.bootstrap_onnx:
        raise ValueError("restore-checkpoint and bootstrap-onnx are mutually exclusive")
    if args.fixed_vx is not None and (
        args.stage != "reference_residual" or not 0.2 <= args.fixed_vx <= 3.0
    ):
        raise ValueError(
            "fixed-vx must be in [0.2, 3.0] and requires reference_residual"
        )
    if args.bootstrap_onnx and profile.factory_kind != "legacy_teacher":
        raise ValueError("bootstrap-onnx requires the legacy_warmstart_v1 profile")
    if (
        args.stage in {"motion_track", "motion_straight", "reference_residual"}
        and args.motion_reference is None
    ):
        raise ValueError(f"{args.stage} requires --motion-reference")
    if args.motion_reference and not args.motion_reference.is_file():
        raise FileNotFoundError(args.motion_reference)
    motion_reference_validation = None
    if args.motion_reference:
        motion_reference_validation = validate_motion_reference(args.motion_reference)
        if not motion_reference_validation["passed"]:
            raise ValueError(
                "motion reference failed validation: "
                + "; ".join(motion_reference_validation["errors"])
            )
        if (
            contract.reference_sha256 is not None
            and _sha256(args.motion_reference) != contract.reference_sha256
        ):
            raise ValueError(
                "motion reference SHA-256 differs from the policy contract"
            )
    phase_sampling = None
    if args.reference_phase_weights:
        if args.stage != "reference_residual" or args.motion_reference is None:
            raise ValueError(
                "reference phase weights require the reference_residual stage"
            )
        phase_sampling = json.loads(
            args.reference_phase_weights.read_text(encoding="utf-8")
        )
        if phase_sampling.get("reference_sha256") != _sha256(args.motion_reference):
            raise ValueError("phase weights were generated for another reference")
        weights = np.asarray(phase_sampling.get("weights"), dtype=np.float64)
        expected_frames = motion_reference_validation["frame_count"]
        if (
            weights.shape != (expected_frames,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.sum(weights) > 0.0
        ):
            raise ValueError("reference phase weights are invalid")
        phase_sampling["weights"] = (weights / np.sum(weights)).tolist()
    if (profile.batch_size * profile.num_minibatches) % args.num_envs:
        raise ValueError(
            "num_envs must divide batch_size * num_minibatches "
            f"({profile.batch_size * profile.num_minibatches})"
        )
    minimum_epoch_timesteps = (
        profile.batch_size * profile.num_minibatches * profile.unroll_length
    )
    effective_num_timesteps = max(args.num_timesteps, minimum_epoch_timesteps)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir / "checkpoints"
    progress_path = args.run_dir / "progress.jsonl"
    stage_overrides = {
        "impl": args.impl,
        "naconmax": max(2048, 16 * args.num_envs),
        "action_clip": max(abs(value) for value in contract.action_clip),
        **STAGES[args.stage],
    }
    if args.fixed_vx is not None:
        stage_overrides["fixed_command"] = [args.fixed_vx, 0.0, 0.0]
    if phase_sampling is not None:
        stage_overrides["reference_phase_sampling_weights"] = phase_sampling["weights"]
    env = DirectionalRun(
        config_overrides=stage_overrides,
        contract=contract,
        motion_reference=args.motion_reference,
    )
    eval_env = None
    if args.motion_reference:
        evaluation_reference_probability = (
            stage_overrides.get("reference_init_probability", 0.0)
            if contract.control_mode == "motion_reference_residual_joint_position"
            else 0.0
        )
        evaluation_overrides = {
            **stage_overrides,
            "reference_init_probability": evaluation_reference_probability,
        }
        # Failure-focused resets are a training curriculum. Held-out metrics
        # retain uniform phases so they stay comparable to earlier profiles.
        if phase_sampling is not None:
            evaluation_overrides["reference_phase_sampling_weights"] = []
        eval_env = DirectionalRun(
            config_overrides=evaluation_overrides,
            contract=contract,
            motion_reference=args.motion_reference,
        )
    network_factory = profile.network_factory()
    restore_params = None
    teacher_parity_max_abs_error = None
    if args.bootstrap_onnx:
        restore_params, teacher_parity_max_abs_error = _teacher_restore_params(
            env, profile, args.bootstrap_onnx, args.seed
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": "formal_fast_locomotion_training",
        "stage": args.stage,
        "policy_contract": contract.policy_name,
        "network_profile": profile.name,
        "network_profile_config": profile.__dict__,
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "devices": [str(device) for device in jax.devices()],
        "python": platform.python_version(),
        "jax": jax.__version__,
        "git_revision": _git_revision(),
        "num_envs": args.num_envs,
        "requested_timesteps": args.num_timesteps,
        "minimum_epoch_timesteps": minimum_epoch_timesteps,
        "effective_timesteps": effective_num_timesteps,
        "seed": args.seed,
        "environment_config": env._config.to_dict(),
        "evaluation_environment_config": (
            eval_env._config.to_dict() if eval_env is not None else None
        ),
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "restore_checkpoint": (
            str(args.restore_checkpoint) if args.restore_checkpoint else None
        ),
        "bootstrap_onnx": (
            str(args.bootstrap_onnx.resolve()) if args.bootstrap_onnx else None
        ),
        "bootstrap_onnx_sha256": (
            _sha256(args.bootstrap_onnx) if args.bootstrap_onnx else None
        ),
        "teacher_parity_max_abs_error": teacher_parity_max_abs_error,
        "motion_reference": (
            str(args.motion_reference.resolve()) if args.motion_reference else None
        ),
        "motion_reference_sha256": (
            _sha256(args.motion_reference) if args.motion_reference else None
        ),
        "motion_reference_validation": motion_reference_validation,
        "reference_phase_weights": (
            str(args.reference_phase_weights.resolve())
            if args.reference_phase_weights
            else None
        ),
        "reference_phase_weights_sha256": (
            _sha256(args.reference_phase_weights)
            if args.reference_phase_weights
            else None
        ),
        "reference_phase_sampling": phase_sampling,
    }
    manifest_path = args.run_dir / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def progress(num_steps: int, metrics: Mapping[str, Any]) -> None:
        row = {
            "num_steps": num_steps,
            "wall_time_unix": time.time(),
            **{key: _json_value(value) for key, value in metrics.items()},
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    started = time.monotonic()
    try:
        _, _, final_metrics = ppo.train(
            environment=env,
            eval_env=eval_env,
            num_timesteps=effective_num_timesteps,
            num_envs=args.num_envs,
            episode_length=env._config.episode_length,
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
            learning_rate_schedule=("ADAPTIVE_KL" if profile.adaptive_kl else None),
            learning_rate_schedule_min_lr=profile.learning_rate_min,
            learning_rate_schedule_max_lr=profile.learning_rate_max,
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
            restore_params=restore_params,
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
