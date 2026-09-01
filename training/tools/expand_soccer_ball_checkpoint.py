#!/usr/bin/env python3
"""Create a K2 ball-conditioned checkpoint without forgetting K1-D."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from brax.training import types as brax_types
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.soccer_ball_policy import (
    SOCCER_BALL_ACTOR_SIZE,
    SOCCER_BALL_FEATURE_SIZE,
    SOCCER_BALL_PRIVILEGED_SIZE,
    SOCCER_MOTION_ACTOR_SIZE,
)
from my3d_rl.soccer_ball_transfer import (
    expand_privileged_observation,
    expand_soccer_motion_params,
)
from my3d_rl.soccer_motion_policy import (
    SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
SOURCE_CONTRACT = (
    REPOSITORY_ROOT / "training" / "contracts" / "soccer_motion_policy_v2.yaml"
)
TARGET_CONTRACT = (
    REPOSITORY_ROOT
    / "training"
    / "contracts"
    / "soccer_ball_motion_policy_v1.yaml"
)
TREE_HASH_ALGORITHM = "sha256_over_sorted_relative_path_nul_file_sha256_newline"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
        raise RuntimeError("formal checkpoint transfer requires a clean Git tree")
    return revision


def _external_new_directory(path: Path) -> Path:
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
        raise FileExistsError(resolved)
    return resolved


def _networks(profile_name: str, actor_size: int, privileged_size: int) -> Any:
    profile = get_ppo_profile(profile_name)
    if profile.normalize_observations:
        raise ValueError("K2 zero-row transfer requires unnormalized observations")
    return profile.network_factory()(
        {"state": actor_size, "privileged_state": privileged_size},
        23,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )


def _parity_report(
    source_params: Any,
    target_params: Any,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    source_networks = _networks(
        "soccer_motion_residual_v3",
        SOCCER_MOTION_ACTOR_SIZE,
        SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE,
    )
    target_networks = _networks(
        "soccer_ball_motion_residual_v1",
        SOCCER_BALL_ACTOR_SIZE,
        SOCCER_BALL_PRIVILEGED_SIZE,
    )
    rng = np.random.default_rng(seed)
    actor = rng.uniform(-1.0, 1.0, (samples, SOCCER_MOTION_ACTOR_SIZE)).astype(
        np.float32
    )
    privileged = rng.uniform(
        -1.0,
        1.0,
        (samples, SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE),
    ).astype(np.float32)
    target_actor = np.concatenate(
        [
            actor,
            np.zeros((samples, SOCCER_BALL_FEATURE_SIZE), dtype=np.float32),
        ],
        axis=1,
    )
    target_privileged = expand_privileged_observation(privileged)
    source_policy = source_networks.policy_network.apply(
        source_params[0], source_params[1], {"state": jp.asarray(actor)}
    )[0]
    target_policy = target_networks.policy_network.apply(
        target_params[0], target_params[1], {"state": jp.asarray(target_actor)}
    )[0]
    source_value = source_networks.value_network.apply(
        source_params[0],
        source_params[2],
        {"privileged_state": jp.asarray(privileged)},
    )
    target_value = target_networks.value_network.apply(
        target_params[0],
        target_params[2],
        {"privileged_state": jp.asarray(target_privileged)},
    )
    policy_error = np.abs(np.asarray(source_policy) - np.asarray(target_policy))
    value_error = np.abs(np.asarray(source_value) - np.asarray(target_value))
    return {
        "samples": samples,
        "seed": seed,
        "zero_appended_features": SOCCER_BALL_FEATURE_SIZE,
        "policy_max_abs": float(np.max(policy_error)),
        "policy_mean_abs": float(np.mean(policy_error)),
        "value_max_abs": float(np.max(value_error)),
        "value_mean_abs": float(np.mean(value_error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260989)
    args = parser.parse_args()
    if min(args.step, args.samples) < 1:
        raise ValueError("step and parity sample count must be positive")
    if not args.source_checkpoint.is_dir():
        raise FileNotFoundError(args.source_checkpoint)
    output_dir = _external_new_directory(args.output_dir)
    revision = _clean_revision()
    source_contract = load_policy_contract(SOURCE_CONTRACT)
    target_contract = load_policy_contract(TARGET_CONTRACT)
    if source_contract.observation_size != SOCCER_MOTION_ACTOR_SIZE:
        raise ValueError("source contract shape differs from transfer implementation")
    if target_contract.observation_size != SOCCER_BALL_ACTOR_SIZE:
        raise ValueError("target contract shape differs from transfer implementation")

    source_params = ppo_checkpoint.load(args.source_checkpoint)
    target_params = expand_soccer_motion_params(source_params)
    parity = _parity_report(
        source_params, target_params, samples=args.samples, seed=args.seed
    )
    threshold = 5.0e-7
    parity["required_max_abs"] = threshold
    parity["passed"] = (
        parity["policy_max_abs"] <= threshold
        and parity["value_max_abs"] <= threshold
    )
    if not parity["passed"]:
        raise RuntimeError(f"zero-extension parity failed: {parity}")

    output_dir.mkdir(parents=True)
    checkpoint_root = output_dir / "checkpoints"
    target_profile = get_ppo_profile("soccer_ball_motion_residual_v1")
    config = ppo_checkpoint.network_config(
        {
            "state": SOCCER_BALL_ACTOR_SIZE,
            "privileged_state": SOCCER_BALL_PRIVILEGED_SIZE,
        },
        target_contract.action_size,
        target_profile.normalize_observations,
        target_profile.network_factory(),
    )
    ppo_checkpoint.save(checkpoint_root, args.step, target_params, config)
    checkpoint_path = checkpoint_root / f"{args.step:012d}"
    report = {
        "schema_version": 1,
        "purpose": "k2_zero_row_ball_target_checkpoint_transfer",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "requires target-conditioned ball training and exact-CPU gates",
        "git_revision": revision,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_tree_sha256": _tree_sha256(args.source_checkpoint),
        "checkpoint_tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "source_contract": str(SOURCE_CONTRACT.resolve()),
        "source_contract_sha256": _sha256(SOURCE_CONTRACT),
        "target_contract": str(TARGET_CONTRACT.resolve()),
        "target_contract_sha256": _sha256(TARGET_CONTRACT),
        "target_profile": target_profile.name,
        "target_checkpoint": str(checkpoint_path.resolve()),
        "target_checkpoint_tree_sha256": _tree_sha256(checkpoint_path),
        "actor_shape": [SOCCER_BALL_ACTOR_SIZE, target_contract.action_size],
        "critic_observation_size": SOCCER_BALL_PRIVILEGED_SIZE,
        "parity": parity,
        "authorization_scope": "training_initialization_only_not_runtime_promotion",
    }
    report_path = output_dir / "transfer-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
