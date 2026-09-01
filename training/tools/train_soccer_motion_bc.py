#!/usr/bin/env python3
"""Fine-tune the compatible K1 PPO actor on selected phase-teacher actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.soccer_motion_bc import (
    action_error_metrics,
    load_soccer_motion_teacher_dataset,
    motion_balanced_indices,
)
from my3d_rl.training_dashboard import TrainingDashboard


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "training" / "contracts" / "soccer_motion_policy_v2.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_external_new_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output directory must stay outside the repository")
    if resolved.exists():
        raise FileExistsError(f"output directory already exists: {resolved}")
    return resolved


def _clean_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    if status:
        raise RuntimeError("formal behavior cloning requires a clean Git tree")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--base-anchor-weight", type=float, default=0.05)
    parser.add_argument("--action-bound-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260950)
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size) < 1
        or args.learning_rate <= 0.0
        or args.base_anchor_weight < 0.0
        or args.action_bound_weight < 0.0
    ):
        raise ValueError("behavior cloning settings are invalid")
    output_dir = _require_external_new_directory(args.output_dir)
    revision = _clean_revision()
    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "complete_teacher_corpus_selected"
        or selection.get("teacher_gates_passed") != selection.get("motion_count")
        or selection["combined_dataset"]["sha256"] != _sha256(args.dataset)
    ):
        raise ValueError("teacher selection manifest is incomplete or mismatched")
    data = load_soccer_motion_teacher_dataset(
        args.dataset,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )
    anchored_base_action = np.clip(
        data["base_action"], *contract.action_clip
    ).astype(np.float32)
    base_params = ppo_checkpoint.load(args.base_checkpoint)
    checkpoint_config = ppo_checkpoint.load_config(args.base_checkpoint)
    preprocess = (
        brax_types.identity_observation_preprocessor
        if not profile.normalize_observations
        else None
    )
    if preprocess is None:
        raise ValueError("K1 BC currently requires the non-normalized PPO profile")
    networks = profile.network_factory()(
        {"state": contract.observation_size, "privileged_state": 118},
        contract.action_size,
        preprocess_observations_fn=preprocess,
    )
    normalizer_params, initial_actor_params, critic_params = base_params
    optimizer = optax.adamw(args.learning_rate, weight_decay=1.0e-6)
    optimizer_state = optimizer.init(initial_actor_params)

    @jax.jit
    def actor_mean(actor_params: Any, observations: jax.Array) -> jax.Array:
        mean, unused_std = networks.policy_network.apply(
            normalizer_params,
            actor_params,
            {"state": observations},
        )
        return mean

    @jax.jit
    def train_step(
        actor_params: Any,
        state: Any,
        observations: jax.Array,
        teacher_actions: jax.Array,
        base_actions: jax.Array,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        def loss_fn(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            prediction = actor_mean(candidate, observations)
            teacher_loss = jp.mean(jp.square(prediction - teacher_actions))
            anchor_loss = jp.mean(jp.square(prediction - base_actions))
            bound_loss = jp.mean(jp.square(jp.maximum(jp.abs(prediction) - 1.0, 0.0)))
            loss = (
                teacher_loss
                + args.base_anchor_weight * anchor_loss
                + args.action_bound_weight * bound_loss
            )
            return loss, {
                "loss/total": loss,
                "loss/teacher": teacher_loss,
                "loss/base_anchor": anchor_loss,
                "loss/action_bound": bound_loss,
            }

        (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            actor_params
        )
        updates, next_state = optimizer.update(gradients, state, actor_params)
        next_params = optax.apply_updates(actor_params, updates)
        metrics["diagnostic/gradient_norm"] = optax.global_norm(gradients)
        metrics["loss/total"] = loss
        return next_params, next_state, metrics

    output_dir.mkdir(parents=True)
    dashboard_path = output_dir / "tensorboard"
    manifest_path = output_dir / "run-manifest.json"
    progress_path = output_dir / "progress.jsonl"
    dashboard = TrainingDashboard(dashboard_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": "k1_b_soccer_motion_teacher_behavior_clone",
        "promotable": False,
        "promotion_blocker": "requires blind dense-grid exact CPU, DAgger and three seeds",
        "git_revision": revision,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": _sha256(args.selection_manifest),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "profile": args.profile,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "base_anchor_weight": args.base_anchor_weight,
        "base_anchor_target": "base_actor_action_clipped_to_policy_contract",
        "action_bound_weight": args.action_bound_weight,
        "seed": args.seed,
        "train_samples": int(np.sum(data["split"] == 0)),
        "validation_samples": int(np.sum(data["split"] == 1)),
        "motion_count": int(len(set(data["motion"].tolist()))),
        "visualization": {
            "format": "tensorboard_event",
            "log_dir": str(dashboard_path.resolve()),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    actor_params = initial_actor_params
    best_actor_params = initial_actor_params
    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(args.seed)
    report_interval = max(1, args.steps // 50)
    validation_mask = data["split"] == 1
    train_mask = data["split"] == 0
    started = time.monotonic()
    try:
        for step in range(1, args.steps + 1):
            indices = motion_balanced_indices(
                rng,
                data["motion"],
                data["split"],
                batch_size=args.batch_size,
            )
            actor_params, optimizer_state, metrics = train_step(
                actor_params,
                optimizer_state,
                jp.asarray(data["observation"][indices]),
                jp.asarray(data["teacher_action"][indices]),
                jp.asarray(anchored_base_action[indices]),
            )
            if step % report_interval == 0 or step == 1 or step == args.steps:
                validation_prediction = np.asarray(
                    actor_mean(
                        actor_params,
                        jp.asarray(data["observation"][validation_mask]),
                    )
                )
                validation_loss = float(
                    np.mean(
                        np.square(
                            validation_prediction
                            - data["teacher_action"][validation_mask]
                        )
                    )
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_actor_params = actor_params
                row = {
                    "step": step,
                    **{name: float(value) for name, value in metrics.items()},
                    "validation/teacher_mse": validation_loss,
                    "validation/best_teacher_mse": best_validation_loss,
                }
                history.append(row)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                dashboard.write(step, row)
                print(json.dumps(row, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    finally:
        dashboard.close()

    train_prediction = np.asarray(
        actor_mean(best_actor_params, jp.asarray(data["observation"][train_mask]))
    )
    validation_prediction = np.asarray(
        actor_mean(
            best_actor_params, jp.asarray(data["observation"][validation_mask])
        )
    )
    checkpoint_root = output_dir / "checkpoints"
    ppo_checkpoint.save(
        checkpoint_root,
        args.steps,
        [normalizer_params, best_actor_params, critic_params],
        checkpoint_config,
    )
    checkpoint_path = checkpoint_root / f"{args.steps:012d}"
    manifest.update(
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "best_validation_teacher_mse": best_validation_loss,
            "train": action_error_metrics(
                train_prediction,
                data["teacher_action"][train_mask],
                anchored_base_action[train_mask],
                data["motion"][train_mask],
            ),
            "validation": action_error_metrics(
                validation_prediction,
                data["teacher_action"][validation_mask],
                anchored_base_action[validation_mask],
                data["motion"][validation_mask],
            ),
            "checkpoint": str(checkpoint_path.resolve()),
            "history": history,
            "selection_boundary": (
                "supervised validation is not a promotion test; select only on a "
                "new blind dense exact-CPU phase grid"
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
