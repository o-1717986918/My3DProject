#!/usr/bin/env python3
"""Create an equal-weight parameter average of aligned soccer PPO runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import numpy as np


REPOSITORY_ROOT = Path(__file__).parents[2]


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
        raise RuntimeError("formal checkpoint averaging requires a clean Git tree")
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
        raise FileExistsError(f"output directory already exists: {resolved}")
    return resolved


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mean_tree(trees: list[Any]) -> Any:
    if len(trees) < 2:
        raise ValueError("checkpoint averaging requires at least two trees")
    structure = jax.tree.structure(trees[0])
    if any(jax.tree.structure(tree) != structure for tree in trees[1:]):
        raise ValueError("checkpoint parameter structures differ")

    def average(*leaves: Any) -> np.ndarray:
        arrays = [np.asarray(leaf) for leaf in leaves]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("checkpoint parameter shapes differ")
        if not np.issubdtype(arrays[0].dtype, np.floating):
            if not all(np.array_equal(arrays[0], array) for array in arrays[1:]):
                raise ValueError("non-floating checkpoint leaves differ")
            return arrays[0]
        result = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64)
        return result.astype(arrays[0].dtype)

    return jax.tree.map(average, trees[0], *trees[1:])


def _trees_equal(first: Any, second: Any) -> bool:
    if jax.tree.structure(first) != jax.tree.structure(second):
        return False
    return all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(first), jax.tree.leaves(second), strict=True
        )
    )


def _select_normalizer(
    base_normalizer: Any,
    candidate_normalizers: list[Any],
    *,
    normalize_observations: bool,
) -> tuple[Any, str]:
    if normalize_observations:
        if any(
            not _trees_equal(base_normalizer, normalizer)
            for normalizer in candidate_normalizers
        ):
            raise ValueError("enabled observation normalizers differ")
        return base_normalizer, "enabled_and_required_exactly_equal"
    return base_normalizer, "disabled_copied_from_retained_base"


def _config_signature(config: dict[str, Any]) -> dict[str, Any]:
    network = config["network_factory_kwargs"]
    return {
        "action_size": config["action_size"],
        "normalize_observations": config["normalize_observations"],
        "observation_size": config["observation_size"],
        "policy_hidden_layer_sizes": list(network["policy_hidden_layer_sizes"]),
        "value_hidden_layer_sizes": list(network["value_hidden_layer_sizes"]),
        "distribution_type": network["distribution_type"],
        "policy_obs_key": network["policy_obs_key"],
        "value_obs_key": network["value_obs_key"],
        "state_dependent_std": network["state_dependent_std"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    if len(args.checkpoints) < 2 or args.step < 0:
        raise ValueError("supply at least two checkpoints and a non-negative step")
    resolved = [path.resolve() for path in args.checkpoints]
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint paths must be unique")
    if any(not path.is_dir() for path in resolved):
        raise FileNotFoundError("every checkpoint must be an existing directory")
    base_checkpoint = args.base_checkpoint.resolve()
    if not base_checkpoint.is_dir():
        raise FileNotFoundError("base checkpoint must be an existing directory")
    if base_checkpoint in resolved:
        raise ValueError("base checkpoint must not duplicate an averaged candidate")
    output_dir = _external_new_directory(args.output_dir)
    revision = _clean_revision()

    params = [ppo_checkpoint.load(path) for path in resolved]
    configs = [ppo_checkpoint.load_config(path) for path in resolved]
    base_params = ppo_checkpoint.load(base_checkpoint)
    base_config = ppo_checkpoint.load_config(base_checkpoint)
    signatures = [_config_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]) or (
        _config_signature(base_config) != signatures[0]
    ):
        raise ValueError("checkpoint network configurations differ")
    normalizer, normalizer_rule = _select_normalizer(
        base_params[0],
        [value[0] for value in params],
        normalize_observations=bool(configs[0]["normalize_observations"]),
    )
    averaged = [
        normalizer,
        _mean_tree([value[1] for value in params]),
        _mean_tree([value[2] for value in params]),
    ]

    checkpoint_root = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=False)
    ppo_checkpoint.save(checkpoint_root, args.step, averaged, configs[0])
    checkpoint_path = checkpoint_root / f"{args.step:012d}"
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "k1_c_equal_weight_aligned_ppo_parameter_average",
        "git_revision": revision,
        "aggregation": "arithmetic_mean_of_aligned_actor_and_critic_parameters",
        "weights": [1.0 / len(resolved)] * len(resolved),
        "normalizer": normalizer_rule,
        "retained_base": {
            "checkpoint": str(base_checkpoint),
            "checkpoint_tree_sha256": _tree_sha256(base_checkpoint),
        },
        "configuration_signature": signatures[0],
        "sources": [
            {
                "checkpoint": str(path),
                "checkpoint_tree_sha256": _tree_sha256(path),
            }
            for path in resolved
        ],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_tree_sha256": _tree_sha256(checkpoint_path),
        "checkpoint_tree_hash_algorithm": (
            "sha256_over_sorted_relative_path_nul_file_sha256_newline"
        ),
        "authorization_scope": "candidate_for_new_blind_grid_not_runtime_promotion",
    }
    manifest_path = output_dir / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
