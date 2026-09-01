#!/usr/bin/env python3
"""Collect kick-policy-v3 DAgger labels from exact transition train states."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, kick_trial_success
from tools.generate_kick_switch_window_corpus import sha256_file


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"
_WORKER_EVALUATOR: KickTeacherEvaluator | None = None
_WORKER_SESSION: ort.InferenceSession | None = None


def training_episode_ids(episode_ids: np.ndarray, split: np.ndarray) -> tuple[int, ...]:
    ids = np.asarray(episode_ids, dtype=np.int32)
    values = np.asarray(split, dtype=np.uint8)
    if ids.ndim != 1 or values.shape != ids.shape:
        raise ValueError("episode IDs and split must be aligned vectors")
    result: list[int] = []
    for episode_id in np.unique(ids):
        episode_values = np.unique(values[ids == episode_id])
        if episode_values.size != 1:
            raise ValueError("base dataset leaks an episode across partitions")
        if int(episode_values[0]) == 0:
            result.append(int(episode_id))
    if not result:
        raise ValueError("base dataset has no training episodes")
    return tuple(result)


def _worker_initialize(
    spec: dict[str, Any], contract_path: str, model_path: str
) -> None:
    global _WORKER_EVALUATOR, _WORKER_SESSION
    _WORKER_EVALUATOR = KickTeacherEvaluator(
        KickTeacherSpec(**spec), contract=load_policy_contract(Path(contract_path))
    )
    _WORKER_SESSION = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )


def _worker_collect(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None or _WORKER_SESSION is None:
        raise RuntimeError("transition DAgger worker was not initialized")
    times, observations, actions, metrics = _WORKER_EVALUATOR.dagger_demonstration(
        np.asarray(task["parameters"], dtype=np.float64),
        _WORKER_SESSION,
        initial_qpos=np.asarray(task["qpos"], dtype=np.float64),
        initial_qvel=np.asarray(task["qvel"], dtype=np.float64),
    )
    return {
        "episode_id": int(task["episode_id"]),
        "observations": observations.astype(np.float32),
        "actions": actions.astype(np.float32),
        "sample_count": int(times.size),
        "success": bool(kick_trial_success(metrics)),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("transition_labels", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be positive")
    npz_path = args.output_prefix.with_suffix(".npz")
    json_path = args.output_prefix.with_suffix(".json")
    if not json_path.is_absolute() or json_path.is_relative_to(Path.cwd()):
        raise ValueError("output prefix must be absolute and outside the repository")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("transition DAgger outputs already exist")

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3":
        raise ValueError("transition DAgger requires kick_policy_v3")
    probe = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    if (
        probe.get_inputs()[0].shape != list(contract.input_shape)
        or probe.get_outputs()[0].shape != list(contract.output_shape)
    ):
        raise ValueError("learner ONNX does not match the selected contract")

    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in teacher.get("records", [])
        if int(record["condition_index"]) == args.condition_index
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher record")
    record = records[0]
    corpus_manifest = json.loads(
        args.transition_corpus.with_suffix(".json").read_text(encoding="utf-8")
    )
    if (
        corpus_manifest.get("npz_sha256") != sha256_file(args.transition_corpus)
        or corpus_manifest.get("contract_sha256") != sha256_file(args.contract)
    ):
        raise ValueError("transition corpus hash or contract mismatch")
    with np.load(args.transition_corpus, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"])
        qvel = np.asarray(archive["qvel"])
        rollout_ids = np.asarray(archive["rollout_id"], dtype=np.int32)

    label_manifest = json.loads(args.transition_labels.read_text(encoding="utf-8"))
    if (
        label_manifest.get("transition_corpus_sha256")
        != sha256_file(args.transition_corpus)
        or label_manifest.get("contract_sha256") != sha256_file(args.contract)
        or not bool(label_manifest.get("complete"))
    ):
        raise ValueError("transition labels are incomplete or mismatched")
    labels = {
        int(label["rollout_id"]): label
        for label in label_manifest["labels"]
        if bool(label["trained_success"])
    }

    base_manifest_path = args.base_dataset.with_suffix(".json")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if (
        base_manifest.get("npz_sha256") != sha256_file(args.base_dataset)
        or base_manifest.get("contract_sha256") != sha256_file(args.contract)
    ):
        raise ValueError("base BC dataset hash or contract mismatch")
    with np.load(args.base_dataset, allow_pickle=False) as archive:
        required = {"observations", "actions", "episode_ids", "split"}
        if not required <= set(archive.files):
            raise ValueError("base BC dataset is missing required arrays")
        base = {name: np.asarray(archive[name]) for name in required}
    train_ids = training_episode_ids(base["episode_ids"], base["split"])
    corpus_by_rollout = {
        int(rollout_id): index for index, rollout_id in enumerate(rollout_ids)
    }
    if not set(train_ids) <= set(labels) & set(corpus_by_rollout):
        raise ValueError("base training episodes are not covered by exact labels")

    tasks = [
        {
            "episode_id": episode_id,
            "qpos": qpos[corpus_by_rollout[episode_id]],
            "qvel": qvel[corpus_by_rollout[episode_id]],
            "parameters": np.asarray(labels[episode_id]["parameters"], np.float64),
        }
        for episode_id in train_ids
    ]
    spec = {
        "target_distance_m": float(record["distance_m"]),
        "target_angle_deg": float(record["angle_deg"]),
        "requested_ball_speed_mps": float(record["requested_speed_mps"]),
        "desired_arrival_speed_mps": float(record["desired_arrival_speed_mps"]),
        "action_mode": str(record["mode"]),
        "evaluation_duration_s": 3.0,
    }
    collected: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_initialize,
        initargs=(spec, str(args.contract.resolve()), str(args.model.resolve())),
    ) as pool:
        futures = [pool.submit(_worker_collect, task) for task in tasks]
        for completed_count, future in enumerate(as_completed(futures), 1):
            node = future.result()
            collected[int(node["episode_id"])] = node
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(f"collected {completed_count}/{len(tasks)} DAgger rollouts", flush=True)
    ordered = [collected[episode_id] for episode_id in train_ids]
    dagger_observations = np.concatenate(
        [node["observations"] for node in ordered]
    ).astype(np.float32)
    dagger_actions = np.concatenate([node["actions"] for node in ordered]).astype(
        np.float32
    )
    dagger_episode_ids = np.concatenate(
        [np.full(node["sample_count"], node["episode_id"], np.int32) for node in ordered]
    )
    arrays = {
        "observations": np.concatenate(
            [base["observations"], dagger_observations]
        ).astype(np.float32),
        "actions": np.concatenate([base["actions"], dagger_actions]).astype(np.float32),
        "episode_ids": np.concatenate(
            [base["episode_ids"], dagger_episode_ids]
        ).astype(np.int32),
        "split": np.concatenate(
            [base["split"], np.zeros(dagger_episode_ids.size, dtype=np.uint8)]
        ),
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    report = {
        "schema_version": 1,
        "purpose": "kick_policy_v3_exact_transition_dagger_dataset",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "aggregated labels only; retraining and held-out physics remain required",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "transition_corpus": str(args.transition_corpus.resolve()),
        "transition_corpus_sha256": sha256_file(args.transition_corpus),
        "transition_labels": str(args.transition_labels.resolve()),
        "transition_labels_sha256": sha256_file(args.transition_labels),
        "base_dataset": str(args.base_dataset.resolve()),
        "base_dataset_sha256": sha256_file(args.base_dataset),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "base_samples": int(base["observations"].shape[0]),
        "dagger_samples": int(dagger_observations.shape[0]),
        "total_samples": int(arrays["observations"].shape[0]),
        "dagger_rollouts": len(ordered),
        "learner_successful_rollouts": sum(bool(node["success"]) for node in ordered),
        "learner_contact_rollouts": sum(
            bool(node["metrics"]["contact"]) for node in ordered
        ),
        "learner_fall_rollouts": sum(
            bool(node["metrics"]["fell"]) for node in ordered
        ),
        "npz": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
