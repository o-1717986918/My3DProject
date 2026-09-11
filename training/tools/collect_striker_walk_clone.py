#!/usr/bin/env python3
"""Collect Apollo Walk labels in the full-circle striker setup task."""

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

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_env import DEFAULT_WALK_POLICY
from my3d_rl.striker_env import DEFAULT_CONTRACT, LongHorizonStriker
from tools.train_striker_teacher import STAGES, _load_parity_report


REPOSITORY_ROOT = Path(__file__).parents[2]


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


def _split_by_environment(
    environment_ids: np.ndarray,
    *,
    validation_folds: int,
    validation_fold_index: int,
) -> np.ndarray:
    """Keep correlated frames from one rollout on only one side of the split."""
    if validation_folds < 2 or not 0 <= validation_fold_index < validation_folds:
        raise ValueError("invalid validation fold")
    return (environment_ids % validation_folds == validation_fold_index).astype(
        np.int8
    )


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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20_261_324)
    parser.add_argument("--validation-folds", type=int, default=5)
    parser.add_argument("--validation-fold-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_envs < args.validation_folds or args.steps < 1 or args.seed < 0:
        raise ValueError("collection settings are invalid")
    run_dir = _external_new_directory(args.run_dir)
    parity = _load_parity_report(args.parity_report, args.impl)
    contract = load_policy_contract(args.contract)
    if contract.policy_name != "striker_policy_v1":
        raise ValueError("walk cloning requires striker_policy_v1")

    overrides = {
        **STAGES["ball_reposition"],
        "impl": args.impl,
        "episode_length": max(args.steps, 2),
        "naconmax": max(2048, 16 * args.num_envs),
        "fixed_action_mode": 0,
        "fixed_desired_arrival_speed": 0.8,
    }
    env = LongHorizonStriker(config_overrides=overrides, contract=contract)
    reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.vmap(env.step)
    initial = reset(
        jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    )

    def scan_step(carry, unused):
        state, active = carry
        candidate = batched_step(
            state, jp.zeros((args.num_envs, env.action_size))
        )
        setup_ready = (
            candidate.info["kick_settled_steps"]
            >= env._config.kick_settled_confirmation_steps
        ) & ~candidate.info["contacted"]
        contacted = candidate.info["contacted"]
        fallen = candidate.metrics["cost/fall"] > 0
        terminal = candidate.done.astype(bool) | contacted | setup_ready
        next_active = active & ~terminal
        next_state = jax.tree.map(
            lambda old, new: _keep_active(old, new, next_active),
            state,
            candidate,
        )
        record = {
            "teacher_observation": state.obs["teacher_state"],
            "student_observation": state.obs["state"],
            "walk_action": candidate.info["walk_last_action"],
            "contact_distance": state.metrics["diagnostic/contact_distance"],
            "heading_error": state.metrics["diagnostic/heading_error"],
            "initial_robot_distance": state.info["initial_robot_distance"],
            "initial_robot_bearing": state.info["initial_robot_bearing"],
            "initial_robot_yaw_error": state.info["initial_robot_yaw_error"],
            "valid": active,
            "terminal_contact": active & contacted,
            "terminal_fall": active & fallen,
            "terminal_setup": active & setup_ready,
        }
        return (next_state, next_active), record

    (_, _), rollout = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            (state, jp.ones(args.num_envs, dtype=bool)),
            xs=None,
            length=args.steps,
        )
    )(initial)
    arrays = {name: np.asarray(value) for name, value in rollout.items()}
    valid = arrays.pop("valid").reshape(-1).astype(bool)
    environment_ids = np.tile(np.arange(args.num_envs), args.steps)[valid]
    step_ids = np.repeat(np.arange(args.steps), args.num_envs)[valid]
    flattened: dict[str, np.ndarray] = {
        name: value.reshape((-1,) + value.shape[2:])[valid]
        for name, value in arrays.items()
        if not name.startswith("terminal_")
    }
    flattened["environment_id"] = environment_ids.astype(np.int32)
    flattened["step"] = step_ids.astype(np.int32)
    # LongHorizonStriker applies Apollo Walk as default_pose + 0.25 * action.
    # Clone the physical joint delta, not the raw Walk network value: the
    # latter legitimately exceeds +/-1 and would be clipped by a direct actor.
    flattened["direct_joint_delta"] = (
        0.25 * flattened["walk_action"]
    ).astype(np.float32)
    flattened["split"] = _split_by_environment(
        environment_ids,
        validation_folds=args.validation_folds,
        validation_fold_index=args.validation_fold_index,
    )
    for name, value in flattened.items():
        if not np.isfinite(value).all():
            raise ValueError(f"collected {name} contains non-finite values")
    if not np.any(flattened["split"] == 0) or not np.any(
        flattened["split"] == 1
    ):
        raise ValueError("dataset needs both train and validation samples")

    run_dir.mkdir(parents=True)
    dataset_path = run_dir / "walk-clone-dataset.npz"
    np.savez_compressed(dataset_path, **flattened)
    actions = flattened["walk_action"]
    joint_delta = flattened["direct_joint_delta"]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "apollo_walk_to_privileged_striker_actor_clone_corpus",
        "promotable": False,
        "promotion_blocker": "requires behavior cloning and closed-loop replay",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip(),
        "implementation": args.impl,
        "backend": jax.default_backend(),
        "backend_parity": parity,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_steps": args.steps,
        "samples": int(valid.sum()),
        "train_samples": int(np.sum(flattened["split"] == 0)),
        "validation_samples": int(np.sum(flattened["split"] == 1)),
        "terminal": {
            name.removeprefix("terminal_"): int(np.asarray(value).sum())
            for name, value in arrays.items()
            if name.startswith("terminal_")
        },
        "walk_action": {
            "maximum_abs": float(np.max(np.abs(actions))),
            "fraction_abs_above_1": float(np.mean(np.abs(actions) > 1.0)),
        },
        "direct_joint_delta": {
            "scale_from_walk_action": 0.25,
            "units": "rad",
            "maximum_abs": float(np.max(np.abs(joint_delta))),
            "fraction_abs_above_1": float(np.mean(np.abs(joint_delta) > 1.0)),
        },
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "contract_scope": "joint_order_and_future_student_observation",
        "walk_policy": str(DEFAULT_WALK_POLICY.resolve()),
        "walk_policy_sha256": _sha256(DEFAULT_WALK_POLICY),
        "environment_config": env._config.to_dict(),
        "split": {
            "unit": "environment_rollout",
            "validation_folds": args.validation_folds,
            "validation_fold_index": args.validation_fold_index,
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
