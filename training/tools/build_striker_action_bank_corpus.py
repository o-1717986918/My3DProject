#!/usr/bin/env python3
"""Build a grouped trigger-state/action-outcome corpus from exact CPU reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_fixed_prior(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("fixed_kick_prior_index", None)
    return result


def load_action_reports(
    report_paths: list[Path],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Validate counterfactual reports and return aligned selector arrays."""
    if len(report_paths) < 2:
        raise ValueError("at least two exact CPU action reports are required")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    reference = sources[0]
    all_reference_rollouts = reference.get("rollouts", [])
    if (
        reference.get("purpose") != "striker_closed_loop_exact_cpu_evaluation"
        or not all_reference_rollouts
    ):
        raise ValueError("reference report is not an exact CPU striker evaluation")
    reference_rollouts = [
        row
        for row in all_reference_rollouts
        if bool(row.get("triggered"))
        and len(row.get("trigger_observation", ())) == 102
        and len(row.get("trigger_walk_last_action", ())) == 23
        and len(row.get("trigger_privileged_observation", ())) == 138
    ]
    eligible_rollout_ids = {int(row["seed"]) for row in reference_rollouts}
    excluded_rollout_ids = [
        int(row["seed"])
        for row in all_reference_rollouts
        if int(row["seed"]) not in eligible_rollout_ids
    ]
    if not reference_rollouts:
        raise ValueError("reference report contains no complete trigger states")
    reference_config = _without_fixed_prior(reference["environment_config"])
    reference_seeds = np.asarray(
        [row["seed"] for row in reference_rollouts], dtype=np.int64
    )
    observations = np.asarray(
        [row["trigger_observation"] for row in reference_rollouts],
        dtype=np.float32,
    )
    walk_last_action = np.asarray(
        [row["trigger_walk_last_action"] for row in reference_rollouts],
        dtype=np.float32,
    )
    privileged_observations = np.asarray(
        [row["trigger_privileged_observation"] for row in reference_rollouts],
        dtype=np.float32,
    )
    if (
        observations.shape != (reference_seeds.size, 102)
        or walk_last_action.shape != (reference_seeds.size, 23)
        or privileged_observations.shape != (reference_seeds.size, 138)
        or not np.isfinite(observations).all()
        or not np.isfinite(walk_last_action).all()
        or not np.isfinite(privileged_observations).all()
    ):
        raise ValueError("reports must contain finite 102-value trigger observations")

    success_rows = []
    fall_rows = []
    final_goal_distance_rows = []
    action_indices = []
    evidence = []
    for path, source in zip(report_paths, sources, strict=True):
        source_by_seed = {
            int(row["seed"]): row for row in source.get("rollouts", [])
        }
        try:
            rollouts = [source_by_seed[int(seed)] for seed in reference_seeds]
        except KeyError as error:
            raise ValueError("action report is missing a reference rollout") from error
        action_index = int(source["environment_config"]["fixed_kick_prior_index"])
        seeds = np.asarray([row["seed"] for row in rollouts], dtype=np.int64)
        candidate_observations = np.asarray(
            [row["trigger_observation"] for row in rollouts], dtype=np.float32
        )
        candidate_walk_last_action = np.asarray(
            [row["trigger_walk_last_action"] for row in rollouts],
            dtype=np.float32,
        )
        candidate_privileged_observations = np.asarray(
            [row["trigger_privileged_observation"] for row in rollouts],
            dtype=np.float32,
        )
        if (
            source.get("purpose")
            != "striker_closed_loop_exact_cpu_evaluation"
            or source.get("contract_sha256") != reference.get("contract_sha256")
            or source.get("kick_prior") != reference.get("kick_prior")
            or source.get("success_definition") != reference.get("success_definition")
            or _without_fixed_prior(source["environment_config"]) != reference_config
            or not np.array_equal(seeds, reference_seeds)
            or not np.array_equal(candidate_observations, observations)
            or not np.array_equal(candidate_walk_last_action, walk_last_action)
            or not np.array_equal(
                candidate_privileged_observations, privileged_observations
            )
        ):
            raise ValueError("exact CPU action reports are not counterfactually aligned")
        success_rows.append([bool(row["succeeded"]) for row in rollouts])
        fall_rows.append([bool(row["fallen"]) for row in rollouts])
        distances = [float(row["final_goal_distance_m"]) for row in rollouts]
        if not np.isfinite(distances).all() or np.any(np.asarray(distances) < 0.0):
            raise ValueError("final goal distances must be finite and nonnegative")
        final_goal_distance_rows.append(distances)
        action_indices.append(action_index)
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "fixed_kick_prior_index": action_index,
                "successes": int(sum(success_rows[-1])),
                "falls": int(sum(fall_rows[-1])),
            }
        )
    if len(set(action_indices)) != len(action_indices):
        raise ValueError("action reports contain duplicate fixed prior indices")
    return (
        observations,
        walk_last_action,
        privileged_observations,
        np.asarray(success_rows, dtype=np.uint8),
        np.asarray(fall_rows, dtype=np.uint8),
        np.asarray(final_goal_distance_rows, dtype=np.float32),
        reference_seeds,
        np.asarray(action_indices, dtype=np.int32),
        {
            "contract": reference["contract"],
            "contract_sha256": reference["contract_sha256"],
            "kick_prior": reference["kick_prior"],
            "success_definition": reference["success_definition"],
            "environment_config": reference_config,
            "reports": evidence,
            "excluded_untriggered_rollout_ids": excluded_rollout_ids,
            "trigger_history": reference.get("trigger_history"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=13_101)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--trigger-history", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not 0.1 <= args.validation_fraction <= 0.4:
        raise ValueError("validation fraction must be in [0.1, 0.4]")
    if not args.output_prefix.is_absolute() or args.output_prefix.is_relative_to(
        Path.cwd()
    ):
        raise ValueError("output prefix must be absolute and outside the repository")
    manifest_path = args.output_prefix.with_suffix(".json")
    npz_path = args.output_prefix.with_suffix(".npz")
    if manifest_path.exists() or npz_path.exists():
        raise FileExistsError("corpus outputs already exist")

    (
        observations,
        walk_last_action,
        privileged_observations,
        success,
        fall,
        final_goal_distance,
        rollout_ids,
        action_indices,
        evidence,
    ) = load_action_reports(args.reports)
    order = np.random.default_rng(args.seed).permutation(rollout_ids.size)
    validation_count = int(round(args.validation_fraction * rollout_ids.size))
    validation_count = min(max(validation_count, 1), rollout_ids.size - 2)
    split = np.zeros(rollout_ids.size, dtype=np.uint8)
    split[order[:validation_count]] = 1
    oracle = success.any(axis=0)
    actor_history = None
    if args.trigger_history is not None:
        history_evidence = evidence.get("trigger_history")
        if (
            not isinstance(history_evidence, dict)
            or history_evidence.get("sha256") != _sha256(args.trigger_history)
        ):
            raise ValueError("trigger-history hash does not match the reference report")
        with np.load(args.trigger_history, allow_pickle=False) as archive:
            history_ids = np.asarray(archive["rollout_id"], dtype=np.int64)
            histories = np.asarray(archive["actor_history"], dtype=np.float32)
        history_by_id = {
            int(rollout_id): histories[index]
            for index, rollout_id in enumerate(history_ids)
        }
        try:
            actor_history = np.stack(
                [history_by_id[int(rollout_id)] for rollout_id in rollout_ids]
            )
        except KeyError as error:
            raise ValueError("trigger history is missing a corpus rollout") from error
        if actor_history.shape != (rollout_ids.size, 50, 102):
            raise ValueError("trigger history has an invalid shape")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    archive_arrays = {
        "actor_observation": observations,
        "selector_observation": np.concatenate(
            [observations, walk_last_action], axis=1
        ).astype(np.float32),
        "walk_last_action": walk_last_action,
        "privileged_observation": privileged_observations,
        "success": success,
        "fall": fall,
        "final_goal_distance_m": final_goal_distance,
        "rollout_id": rollout_ids,
        "action_prior_index": action_indices,
        "split": split,
    }
    if actor_history is not None:
        archive_arrays["actor_history"] = actor_history
    np.savez_compressed(npz_path, **archive_arrays)
    manifest = {
        "schema_version": 2,
        "purpose": "exact_cpu_striker_trigger_action_bank_corpus",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "requires learned selection on untouched rollouts",
        "npz": str(npz_path.resolve()),
        "npz_sha256": _sha256(npz_path),
        "seed": args.seed,
        "rollouts": int(rollout_ids.size),
        "actions": int(action_indices.size),
        "action_prior_indices": action_indices.tolist(),
        "observation_size": int(observations.shape[1]),
        "selector_observation_size": int(
            observations.shape[1] + walk_last_action.shape[1]
        ),
        "privileged_observation_size": int(privileged_observations.shape[1]),
        "history_frames": 50 if actor_history is not None else 0,
        "selector_observation_fields": [
            "striker_policy_v1_actor_observation",
            "apollo_walk_last_action",
        ],
        "validation_fraction": args.validation_fraction,
        "training_rollouts": int(np.sum(split == 0)),
        "validation_rollouts": int(np.sum(split == 1)),
        "successes_by_action": success.sum(axis=1).astype(int).tolist(),
        "falls_by_action": fall.sum(axis=1).astype(int).tolist(),
        "final_goal_distance_m_by_action": [
            {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
            for values in final_goal_distance
        ],
        "oracle": {
            "all_successes": int(oracle.sum()),
            "all_rate": float(oracle.mean()),
            "training_successes": int(oracle[split == 0].sum()),
            "training_rate": float(oracle[split == 0].mean()),
            "validation_successes": int(oracle[split == 1].sum()),
            "validation_rate": float(oracle[split == 1].mean()),
        },
        **evidence,
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
