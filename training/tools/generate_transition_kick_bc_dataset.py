#!/usr/bin/env python3
"""Replay successful per-state teachers into a grouped kick-policy-v3 BC set."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, kick_trial_success
from tools.generate_kick_switch_window_corpus import sha256_file
from tools.generate_kick_transition_corpus import stratified_rollout_split


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"
_WORKER_EVALUATOR: KickTeacherEvaluator | None = None


def expand_episode_split(
    episode_ids: np.ndarray, episode_splits: dict[int, int]
) -> np.ndarray:
    """Expand one split per episode to samples while rejecting missing IDs."""
    ids = np.asarray(episode_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("episode IDs must be one-dimensional")
    missing = set(ids.tolist()) - set(episode_splits)
    if missing:
        raise ValueError(f"episode split is missing IDs {sorted(missing)}")
    values = np.asarray([episode_splits[int(value)] for value in ids], dtype=np.uint8)
    if not set(values.tolist()) <= {0, 1}:
        raise ValueError("episode split values must be zero or one")
    return values


def _worker_initialize(spec: dict[str, Any], contract_path: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = KickTeacherEvaluator(
        KickTeacherSpec(**spec), contract=load_policy_contract(Path(contract_path))
    )


def _worker_replay(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("BC replay worker was not initialized")
    metrics = _WORKER_EVALUATOR.rollout(
        np.asarray(task["parameters"], dtype=np.float64),
        initial_qpos=np.asarray(task["qpos"], dtype=np.float64),
        initial_qvel=np.asarray(task["qvel"], dtype=np.float64),
        capture_targets=True,
    )
    observations = _WORKER_EVALUATOR.captured_observations
    actions = _WORKER_EVALUATOR.captured_actions
    targets = _WORKER_EVALUATOR.captured_targets
    if (
        observations.ndim != 2
        or observations.shape[0] < 1
        or observations.shape[1] != _WORKER_EVALUATOR.contract.observation_size
        or actions.shape
        != (observations.shape[0], _WORKER_EVALUATOR.contract.action_size)
        or targets.shape != actions.shape
    ):
        raise RuntimeError("captured teacher trajectory has incompatible arrays")
    return {
        **task,
        "observations": observations.astype(np.float32),
        "actions": actions.astype(np.float32),
        "targets": targets.astype(np.float32),
        "metrics": metrics,
        "success": bool(kick_trial_success(metrics)),
    }


def _accepted_condition(source: dict[str, Any], condition_index: int) -> dict[str, Any]:
    records = [
        record
        for record in source.get("records", [])
        if int(record["condition_index"]) == condition_index
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher record")
    return records[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("transition_labels", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seed", type=int, default=9601)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.workers < 1 or not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("workers or validation fraction are out of range")
    npz_path = args.output_prefix.with_suffix(".npz")
    json_path = args.output_prefix.with_suffix(".json")
    if not json_path.is_absolute() or json_path.is_relative_to(Path.cwd()):
        raise ValueError("output prefix must be absolute and outside the repository")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("BC dataset outputs already exist")

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3" or contract.observation_size != 98:
        raise ValueError("transition BC requires the kick_policy_v3 contract")
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    record = _accepted_condition(teacher, args.condition_index)
    corpus_manifest_path = args.transition_corpus.with_suffix(".json")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("npz_sha256") != sha256_file(args.transition_corpus)
        or corpus_manifest.get("contract_sha256") != sha256_file(args.contract)
        or int(corpus_manifest.get("teacher_condition_index", -1))
        != args.condition_index
    ):
        raise ValueError("transition corpus is invalid or bound to other inputs")
    with np.load(args.transition_corpus, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "rollout_id", "phase_bucket", "split"}
        if not required <= set(archive.files):
            raise ValueError("transition corpus is missing required arrays")
        corpus = {name: np.asarray(archive[name]) for name in required}

    labels_manifest = json.loads(args.transition_labels.read_text(encoding="utf-8"))
    if (
        labels_manifest.get("purpose")
        != "exact_cpu_per_transition_kick_teacher_labels"
        or not bool(labels_manifest.get("complete"))
        or labels_manifest.get("transition_corpus_sha256")
        != sha256_file(args.transition_corpus)
        or labels_manifest.get("contract_sha256") != sha256_file(args.contract)
        or labels_manifest.get("teacher_manifest_sha256")
        != sha256_file(args.teacher_manifest)
    ):
        raise ValueError("transition labels are incomplete or bound to other inputs")
    labels_npz = Path(str(labels_manifest["npz"]))
    if (
        not labels_npz.is_file()
        or labels_manifest.get("npz_sha256") != sha256_file(labels_npz)
    ):
        raise ValueError("transition label NPZ is missing or has a hash mismatch")
    with np.load(labels_npz, allow_pickle=False) as archive:
        required = {
            "corpus_index",
            "rollout_id",
            "phase_bucket",
            "parameters",
            "trained_success",
        }
        if not required <= set(archive.files):
            raise ValueError("transition labels are missing required arrays")
        labels = {name: np.asarray(archive[name]) for name in required}
    label_records = {
        int(label["corpus_index"]): label for label in labels_manifest["labels"]
    }
    if set(label_records) != set(labels["corpus_index"].tolist()):
        raise ValueError("transition label JSON and NPZ cover different states")
    successful = np.flatnonzero(labels["trained_success"] == 1)
    if successful.size < 2:
        raise ValueError("fewer than two successful transition teachers are available")

    tasks: list[dict[str, Any]] = []
    for label_row in successful:
        corpus_index = int(labels["corpus_index"][label_row])
        if not 0 <= corpus_index < corpus["qpos"].shape[0]:
            raise ValueError("transition label corpus index is out of range")
        if (
            int(corpus["split"][corpus_index]) != 0
            or int(corpus["rollout_id"][corpus_index])
            != int(labels["rollout_id"][label_row])
            or int(corpus["phase_bucket"][corpus_index])
            != int(labels["phase_bucket"][label_row])
        ):
            raise ValueError("transition label row does not match its corpus state")
        label_record = label_records[corpus_index]
        if not bool(label_record["trained_success"]):
            raise ValueError("successful label NPZ row disagrees with its JSON record")
        tasks.append(
            {
                "corpus_index": corpus_index,
                "episode_id": int(labels["rollout_id"][label_row]),
                "phase_bucket": int(labels["phase_bucket"][label_row]),
                "qpos": corpus["qpos"][corpus_index],
                "qvel": corpus["qvel"][corpus_index],
                # JSON retains the optimizer's float64 boundary values. The compact
                # NPZ is float32 and may round a value a few ulps outside its bound.
                "parameters": np.asarray(
                    label_record["parameters"], dtype=np.float64
                ),
            }
        )

    spec = {
        "target_distance_m": float(record["distance_m"]),
        "target_angle_deg": float(record["angle_deg"]),
        "requested_ball_speed_mps": float(record["requested_speed_mps"]),
        "desired_arrival_speed_mps": float(record["desired_arrival_speed_mps"]),
        "action_mode": str(record["mode"]),
        "evaluation_duration_s": 3.0,
    }
    replayed: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_initialize,
        initargs=(spec, str(args.contract.resolve())),
    ) as pool:
        futures = [pool.submit(_worker_replay, task) for task in tasks]
        for completed_count, future in enumerate(as_completed(futures), 1):
            node = future.result()
            replayed[int(node["corpus_index"])] = node
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(f"replayed {completed_count}/{len(tasks)} teachers", flush=True)
    ordered = [replayed[index] for index in sorted(replayed)]
    failed = [node for node in ordered if not bool(node["success"])]
    if failed:
        failed_ids = [int(node["episode_id"]) for node in failed]
        raise RuntimeError(f"successful labels failed deterministic replay: {failed_ids}")

    episode_ids = np.asarray([node["episode_id"] for node in ordered], np.int32)
    episode_buckets = np.asarray([node["phase_bucket"] for node in ordered], np.int32)
    episode_split_values = stratified_rollout_split(
        episode_buckets,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    episode_splits = {
        int(episode_id): int(split)
        for episode_id, split in zip(
            episode_ids, episode_split_values, strict=True
        )
    }
    sample_episode_ids = np.concatenate(
        [
            np.full(node["observations"].shape[0], node["episode_id"], np.int32)
            for node in ordered
        ]
    )
    sample_split = expand_episode_split(sample_episode_ids, episode_splits)
    arrays = {
        "observations": np.concatenate(
            [node["observations"] for node in ordered]
        ).astype(np.float32),
        "actions": np.concatenate([node["actions"] for node in ordered]).astype(
            np.float32
        ),
        "joint_targets_rad": np.concatenate(
            [node["targets"] for node in ordered]
        ).astype(np.float32),
        "episode_ids": sample_episode_ids,
        "corpus_index": np.concatenate(
            [
                np.full(node["observations"].shape[0], node["corpus_index"], np.int32)
                for node in ordered
            ]
        ),
        "phase_bucket": np.concatenate(
            [
                np.full(node["observations"].shape[0], node["phase_bucket"], np.int32)
                for node in ordered
            ]
        ),
        "control_index": np.concatenate(
            [
                np.arange(node["observations"].shape[0], dtype=np.int32)
                for node in ordered
            ]
        ),
        "split": sample_split,
    }
    if set(sample_episode_ids[sample_split == 0].tolist()) & set(
        sample_episode_ids[sample_split == 1].tolist()
    ):
        raise RuntimeError("BC dataset leaked an episode across its split")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "schema_version": 1,
        "purpose": "kick_policy_v3_exact_transition_behavior_clone_dataset",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "training data only; held-out closed-loop and server gates remain required",
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "transition_corpus": str(args.transition_corpus.resolve()),
        "transition_corpus_sha256": sha256_file(args.transition_corpus),
        "transition_labels": str(args.transition_labels.resolve()),
        "transition_labels_sha256": sha256_file(args.transition_labels),
        "transition_labels_npz": str(labels_npz.resolve()),
        "transition_labels_npz_sha256": sha256_file(labels_npz),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "seed": args.seed,
        "source_successful_teacher_episodes": int(successful.size),
        "replayed_teacher_episodes": len(ordered),
        "training_episodes": int(np.count_nonzero(episode_split_values == 0)),
        "validation_episodes": int(np.count_nonzero(episode_split_values == 1)),
        "sample_count": int(arrays["observations"].shape[0]),
        "training_samples": int(np.count_nonzero(sample_split == 0)),
        "validation_samples": int(np.count_nonzero(sample_split == 1)),
        "split_unit": "whole_exact_transition_teacher_episode",
        "phase_buckets": sorted(set(episode_buckets.tolist())),
        "npz": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
