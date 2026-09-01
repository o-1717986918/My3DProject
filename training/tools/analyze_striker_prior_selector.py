#!/usr/bin/env python3
"""Audit trigger-state action identifiability with deterministic baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def knn_probabilities(
    observations: np.ndarray,
    success: np.ndarray,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    feature_indices: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    train = observations[train_rows][:, feature_indices]
    evaluation = observations[evaluation_rows][:, feature_indices]
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 1.0e-3)
    train = (train - mean) / std
    evaluation = (evaluation - mean) / std
    distances = np.sum(
        np.square(evaluation[:, None, :] - train[None, :, :]), axis=2
    )
    k = min(max(int(neighbors), 1), train.shape[0])
    nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
    train_labels = success[:, train_rows].T
    return train_labels[nearest].mean(axis=1)


def _decision_metrics(
    success: np.ndarray,
    fall: np.ndarray,
    rows: np.ndarray,
    chosen: np.ndarray,
) -> dict[str, float | int]:
    selected = np.flatnonzero(rows)
    succeeded = success[chosen, selected]
    fallen = fall[chosen, selected]
    return {
        "rollouts": int(selected.size),
        "successes": int(succeeded.sum()),
        "falls": int(fallen.sum()),
        "success_rate": float(succeeded.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be absolute and outside the repository")
    source = json.loads(args.corpus.with_suffix(".json").read_text(encoding="utf-8"))
    if source.get("npz_sha256") != _sha256(args.corpus):
        raise ValueError("corpus hash mismatch")
    with np.load(args.corpus, allow_pickle=False) as archive:
        actor_observations = np.asarray(
            archive["actor_observation"], dtype=np.float32
        )
        selector_observations = np.asarray(
            archive["selector_observation"], dtype=np.float32
        )
        privileged_observations = np.asarray(
            archive["privileged_observation"], dtype=np.float32
        )
        history = np.asarray(archive["actor_history"], dtype=np.float32)
        success = np.asarray(archive["success"], dtype=bool)
        fall = np.asarray(archive["fall"], dtype=bool)
        split = np.asarray(archive["split"], dtype=np.uint8)
        action_indices = np.asarray(archive["action_prior_index"], dtype=np.int32)
    train_rows = split == 0
    validation_rows = split == 1
    validation_indices = np.flatnonzero(validation_rows)
    trials = []
    profiles = {
        "privileged_138": (privileged_observations, np.arange(138)),
        "selector_125": (selector_observations, np.arange(125)),
        "actor_102": (actor_observations, np.arange(102)),
        "physical_74": (
            actor_observations,
            np.r_[0:52, 75:86, 91:102],
        ),
        "contact_state_62": (
            actor_observations,
            np.r_[6:52, 75:78, 81:85, 91:95, 97:102],
        ),
        "compact_19": (
            actor_observations,
            np.r_[0:6, 75:78, 81:85, 91:95, 97:102],
        ),
        "history_summary_431": (
            np.concatenate(
                [
                    selector_observations,
                    history.mean(axis=1),
                    history.std(axis=1),
                    history[:, -1] - history[:, 0],
                ],
                axis=1,
            ),
            np.arange(431),
        ),
        "history_10frame_1043": (
            np.concatenate(
                [
                    history[:, np.linspace(0, 49, 10, dtype=np.int32)].reshape(
                        history.shape[0], -1
                    ),
                    selector_observations[:, 102:125],
                ],
                axis=1,
            ),
            np.arange(1043),
        ),
    }
    for profile_name, (observations, feature_indices) in profiles.items():
        for neighbors in (1, 3, 5, 9, 15, 25, 41):
            probabilities = knn_probabilities(
                observations,
                success,
                train_rows,
                validation_rows,
                feature_indices,
                neighbors,
            )
            chosen = np.argmax(probabilities, axis=1)
            trials.append(
                {
                    "method": "knn",
                    "profile": profile_name,
                    "features": int(feature_indices.size),
                    "neighbors": neighbors,
                    **_decision_metrics(
                        success,
                        fall,
                        validation_rows,
                        chosen,
                    ),
                }
            )

    target_distance = actor_observations[:, 83]
    edges = np.arange(2.0, 3.5001, 0.25)
    train_bins = np.clip(np.digitize(target_distance[train_rows], edges), 1, len(edges))
    validation_bins = np.clip(
        np.digitize(target_distance[validation_rows], edges), 1, len(edges)
    )
    global_action = int(np.argmax(success[:, train_rows].sum(axis=1)))
    bin_actions = {}
    for bin_index in range(1, len(edges) + 1):
        rows = np.flatnonzero(train_rows)[train_bins == bin_index]
        bin_actions[bin_index] = (
            int(np.argmax(success[:, rows].sum(axis=1)))
            if rows.size >= 5
            else global_action
        )
    binned_choice = np.asarray(
        [bin_actions[int(bin_index)] for bin_index in validation_bins], dtype=np.int32
    )
    trials.append(
        {
            "method": "target_distance_bins",
            "bin_width_m": 0.25,
            "bin_local_actions": bin_actions,
            **_decision_metrics(success, fall, validation_rows, binned_choice),
        }
    )
    fixed_trials = []
    for local_index, prior_index in enumerate(action_indices):
        chosen = np.full(validation_indices.size, local_index, dtype=np.int32)
        fixed_trials.append(
            {
                "local_index": local_index,
                "prior_index": int(prior_index),
                **_decision_metrics(success, fall, validation_rows, chosen),
            }
        )
    best_trial = max(
        trials,
        key=lambda item: (-int(item["falls"]), int(item["successes"])),
    )
    report = {
        "schema_version": 1,
        "purpose": "striker_trigger_action_identifiability_audit",
        "promotable": False,
        "promotion_blocker": "validation split is consumed for route selection",
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": _sha256(args.corpus),
        "training_rollouts": int(train_rows.sum()),
        "validation_rollouts": int(validation_rows.sum()),
        "fixed_action_validation": fixed_trials,
        "oracle_validation": {
            "successes": int(success[:, validation_rows].any(axis=0).sum()),
            "rate": float(success[:, validation_rows].any(axis=0).mean()),
        },
        "trials": trials,
        "best_trial": best_trial,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
