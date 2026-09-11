#!/usr/bin/env python3
"""Behavior-clone Apollo Walk into the privileged striker actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from brax.training import types as brax_types
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jp
import numpy as np
import optax

from my3d_rl.contract import load_policy_contract
from my3d_rl.striker_env import DEFAULT_CONTRACT
from tools.collect_striker_walk_clone import _external_new_directory
from tools.train_striker_teacher import make_striker_network_factory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_clone_dataset(
    dataset_path: Path,
    manifest_path: Path,
    *,
    observation_size: int,
    action_size: int,
    contract_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("purpose")
        != "apollo_walk_to_privileged_striker_actor_clone_corpus"
        or manifest.get("dataset_sha256") != _sha256(dataset_path)
        or (
            contract_sha256 is not None
            and manifest.get("contract_sha256") != contract_sha256
        )
    ):
        raise ValueError("Walk-clone dataset manifest is incomplete or mismatched")
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {"teacher_observation", "direct_joint_delta", "split"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Walk-clone dataset lacks {sorted(missing)}")
        data = {name: np.asarray(archive[name]) for name in required}
    count = data["teacher_observation"].shape[0]
    if data["teacher_observation"].shape != (count, observation_size):
        raise ValueError("Walk-clone teacher observation shape is invalid")
    if data["direct_joint_delta"].shape != (count, action_size):
        raise ValueError("Walk-clone action shape is invalid")
    if not np.isfinite(data["teacher_observation"]).all() or not np.isfinite(
        data["direct_joint_delta"]
    ).all():
        raise ValueError("Walk-clone dataset contains non-finite values")
    if np.max(np.abs(data["direct_joint_delta"])) > 1.0 + 1.0e-6:
        raise ValueError("Walk-clone joint deltas exceed direct actor bounds")
    if not set(data["split"].tolist()) <= {0, 1}:
        raise ValueError("Walk-clone split must be binary")
    if not np.any(data["split"] == 0) or not np.any(data["split"] == 1):
        raise ValueError("Walk-clone dataset needs train and validation rows")
    return {
        "observation": data["teacher_observation"].astype(np.float32),
        "action": data["direct_joint_delta"].astype(np.float32),
        "split": data["split"].astype(np.int8),
    }, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20_261_325)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1 or args.learning_rate <= 0.0:
        raise ValueError("Walk-clone optimization settings are invalid")
    if not args.base_checkpoint.is_dir():
        raise FileNotFoundError(args.base_checkpoint)
    run_dir = _external_new_directory(args.run_dir)
    contract = load_policy_contract(args.contract)
    if contract.policy_name != "striker_policy_v1":
        raise ValueError("Walk cloning requires striker_policy_v1")
    data, source_manifest = _load_clone_dataset(
        args.dataset,
        args.dataset_manifest,
        observation_size=138,
        action_size=contract.action_size,
        contract_sha256=_sha256(args.contract),
    )

    checkpoint_params = ppo_checkpoint.load(args.base_checkpoint)
    checkpoint_config = ppo_checkpoint.load_config(args.base_checkpoint)
    factory_config = checkpoint_config.get("network_factory_kwargs", {})
    if (
        checkpoint_config.get("normalize_observations") is not False
        or factory_config.get("policy_obs_key") != "teacher_state"
        or factory_config.get("value_obs_key") != "privileged_state"
    ):
        raise ValueError("base checkpoint does not use the striker teacher network")
    normalizer_params, actor_params, critic_params = checkpoint_params
    networks = make_striker_network_factory(
        init_noise_std=float(factory_config["init_noise_std"])
    )(
        {"state": 102, "teacher_state": 138, "privileged_state": 138},
        contract.action_size,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )

    @jax.jit
    def actor_mean(params: Any, observations: jax.Array) -> jax.Array:
        mean, unused_std = networks.policy_network.apply(
            normalizer_params,
            params,
            {"teacher_state": observations},
        )
        return mean

    optimizer = optax.adamw(args.learning_rate, weight_decay=1.0e-6)
    optimizer_state = optimizer.init(actor_params)

    @jax.jit
    def train_step(params, state, observations, targets):
        def loss_fn(candidate):
            prediction = actor_mean(candidate, observations)
            return jp.mean(jp.square(prediction - targets))

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, next_state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), next_state, loss

    train_indices = np.flatnonzero(data["split"] == 0)
    validation_indices = np.flatnonzero(data["split"] == 1)
    initial_validation = np.asarray(
        actor_mean(
            actor_params,
            jp.asarray(data["observation"][validation_indices]),
        )
    )
    initial_validation_mse = float(
        np.mean(
            np.square(
                initial_validation - data["action"][validation_indices]
            )
        )
    )
    rng = np.random.default_rng(args.seed)
    best_params = actor_params
    best_validation_mse = initial_validation_mse
    history: list[dict[str, float | int]] = []
    report_interval = max(1, args.steps // 50)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        batch = rng.choice(train_indices, size=args.batch_size, replace=True)
        actor_params, optimizer_state, loss = train_step(
            actor_params,
            optimizer_state,
            jp.asarray(data["observation"][batch]),
            jp.asarray(data["action"][batch]),
        )
        if step % report_interval == 0 or step == 1 or step == args.steps:
            validation_prediction = np.asarray(
                actor_mean(
                    actor_params,
                    jp.asarray(data["observation"][validation_indices]),
                )
            )
            validation_mse = float(
                np.mean(
                    np.square(
                        validation_prediction
                        - data["action"][validation_indices]
                    )
                )
            )
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_params = actor_params
            row = {
                "step": step,
                "train_mse": float(loss),
                "validation_mse": validation_mse,
                "best_validation_mse": best_validation_mse,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    run_dir.mkdir(parents=True)
    checkpoint_root = run_dir / "checkpoints"
    ppo_checkpoint.save(
        checkpoint_root,
        args.steps,
        [normalizer_params, best_params, critic_params],
        checkpoint_config,
    )
    checkpoint_path = checkpoint_root / f"{args.steps:012d}"
    final_train = np.asarray(
        actor_mean(best_params, jp.asarray(data["observation"][train_indices]))
    )
    final_validation = np.asarray(
        actor_mean(
            best_params,
            jp.asarray(data["observation"][validation_indices]),
        )
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "apollo_walk_initialized_privileged_striker_actor",
        "promotable": False,
        "promotion_blocker": "requires independent closed-loop direct-decoder replay",
        "policy_action_semantics": "physical_joint_delta_rad",
        "contract_scope": "joint_order_and_future_student_observation",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip(),
        "backend": jax.default_backend(),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "source_walk_policy_sha256": source_manifest["walk_policy_sha256"],
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "initial_validation_mse": initial_validation_mse,
        "best_validation_mse": best_validation_mse,
        "train_mse": float(
            np.mean(np.square(final_train - data["action"][train_indices]))
        ),
        "validation_mse": float(
            np.mean(
                np.square(
                    final_validation - data["action"][validation_indices]
                )
            )
        ),
        "validation_max_abs_error": float(
            np.max(
                np.abs(
                    final_validation - data["action"][validation_indices]
                )
            )
        ),
        "history": history,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
