#!/usr/bin/env python3
"""Measure oracle switch-window coverage of a small exact-CPU kick bank."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, kick_trial_success
from tools.generate_kick_switch_window_corpus import load_prototype, sha256_file


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"
_WORKER_EVALUATOR: KickTeacherEvaluator | None = None
_WORKER_PARAMETERS: list[np.ndarray] = []


def greedy_rollout_cover(
    success: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    eligible_rows: np.ndarray,
    *,
    maximum_prototypes: int,
) -> list[int]:
    """Greedily cover approach rollouts, preferring safer prototypes on ties."""
    success_values = np.asarray(success, dtype=bool)
    fall_values = np.asarray(fall, dtype=bool)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    eligible = np.asarray(eligible_rows, dtype=bool)
    if (
        success_values.ndim != 2
        or fall_values.shape != success_values.shape
        or ids.shape != (success_values.shape[1],)
        or eligible.shape != ids.shape
    ):
        raise ValueError("prototype matrices and rollout rows are misaligned")
    if maximum_prototypes < 1:
        raise ValueError("maximum prototypes must be positive")
    train_ids = np.unique(ids[eligible])
    if train_ids.size < 1:
        raise ValueError("eligible rows contain no approach rollouts")
    per_rollout = np.zeros((success_values.shape[0], train_ids.size), dtype=bool)
    per_rollout_falls = np.zeros_like(per_rollout)
    for column, rollout_id in enumerate(train_ids):
        rows = eligible & (ids == rollout_id)
        per_rollout[:, column] = np.any(success_values[:, rows], axis=1)
        per_rollout_falls[:, column] = np.any(fall_values[:, rows], axis=1)
    covered = np.zeros(train_ids.size, dtype=bool)
    selected: list[int] = []
    available = set(range(success_values.shape[0]))
    for _ in range(min(maximum_prototypes, success_values.shape[0])):
        ranked = sorted(
            available,
            key=lambda index: (
                int(np.count_nonzero(per_rollout[index] & ~covered)),
                -int(np.count_nonzero(per_rollout_falls[index])),
                int(np.count_nonzero(per_rollout[index])),
                -index,
            ),
            reverse=True,
        )
        choice = ranked[0]
        if not np.any(per_rollout[choice] & ~covered):
            break
        selected.append(choice)
        covered |= per_rollout[choice]
        available.remove(choice)
    return selected


def rollout_coverage(
    success: np.ndarray,
    rollout_ids: np.ndarray,
    rows: np.ndarray,
    prototype_indices: list[int],
) -> tuple[int, int]:
    values = np.asarray(success, dtype=bool)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    selected_rows = np.asarray(rows, dtype=bool)
    if values.ndim != 2 or ids.shape != (values.shape[1],) or selected_rows.shape != ids.shape:
        raise ValueError("success matrix and rollout rows are misaligned")
    if not prototype_indices:
        return 0, int(np.unique(ids[selected_rows]).size)
    possible = np.any(values[prototype_indices], axis=0)
    unique = np.unique(ids[selected_rows])
    covered = sum(bool(np.any(possible[selected_rows & (ids == rollout_id)])) for rollout_id in unique)
    return covered, int(unique.size)


def _parse_prototype(value: str) -> tuple[Path, int]:
    manifest, separator, rollout_id = value.rpartition(":")
    if not separator or not manifest:
        raise argparse.ArgumentTypeError("prototype must be MANIFEST:ROLLOUT_ID")
    try:
        parsed_id = int(rollout_id)
    except ValueError as error:
        raise argparse.ArgumentTypeError("prototype rollout ID must be an integer") from error
    return Path(manifest), parsed_id


def _worker_initialize(
    spec: dict[str, Any], contract_path: str, parameters: list[list[float]]
) -> None:
    global _WORKER_EVALUATOR, _WORKER_PARAMETERS
    _WORKER_EVALUATOR = KickTeacherEvaluator(
        KickTeacherSpec(**spec), contract=load_policy_contract(Path(contract_path))
    )
    _WORKER_PARAMETERS = [np.asarray(value, dtype=np.float64) for value in parameters]


def _worker_evaluate(task: tuple[int, int, np.ndarray, np.ndarray]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None or not _WORKER_PARAMETERS:
        raise RuntimeError("prototype-bank worker was not initialized")
    prototype_index, candidate_index, qpos, qvel = task
    metrics = _WORKER_EVALUATOR.rollout(
        _WORKER_PARAMETERS[prototype_index],
        initial_qpos=np.asarray(qpos, dtype=np.float64),
        initial_qvel=np.asarray(qvel, dtype=np.float64),
    )
    return {
        "prototype_index": prototype_index,
        "candidate_index": candidate_index,
        "success": bool(kick_trial_success(metrics)),
        "fall": bool(metrics["fell"]),
        "contact": bool(metrics["contact"]),
        "score": float(metrics["score"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("switch_corpus", type=Path)
    parser.add_argument(
        "--prototype", type=_parse_prototype, action="append", required=True
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-prototypes", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.workers < 1 or args.maximum_prototypes < 1:
        raise ValueError("workers and maximum prototypes must be positive")
    npz_path = args.output_prefix.with_suffix(".npz")
    json_path = args.output_prefix.with_suffix(".json")
    if not json_path.is_absolute() or json_path.is_relative_to(Path.cwd()):
        raise ValueError("output prefix must be absolute and outside the repository")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("prototype-bank outputs already exist")

    corpus_manifest_path = args.switch_corpus.with_suffix(".json")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("purpose")
        != "exact_cpu_walk_to_kick_switch_window_corpus"
        or corpus_manifest.get("npz_sha256") != sha256_file(args.switch_corpus)
    ):
        raise ValueError("switch-window corpus is invalid or has a hash mismatch")
    if corpus_manifest.get("contract_sha256") != sha256_file(args.contract):
        raise ValueError("switch-window corpus contract mismatch")
    with np.load(args.switch_corpus, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "approach_rollout_id", "split"}
        if not required <= set(archive.files):
            raise ValueError("switch-window corpus is missing required arrays")
        qpos = np.asarray(archive["qpos"])
        qvel = np.asarray(archive["qvel"])
        rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        split = np.asarray(archive["split"], dtype=np.uint8)
    if qpos.shape[0] != qvel.shape[0] or rollout_ids.shape != split.shape or qpos.shape[0] != split.size:
        raise ValueError("switch-window corpus arrays are misaligned")

    prototype_nodes: list[dict[str, Any]] = []
    parameter_rows: list[np.ndarray] = []
    seen: set[tuple[str, int]] = set()
    for manifest_path, rollout_id in args.prototype:
        identity = (sha256_file(manifest_path), rollout_id)
        if identity in seen:
            raise ValueError("duplicate prototype selection")
        seen.add(identity)
        parameters, source_npz, source = load_prototype(
            manifest_path, rollout_id=rollout_id
        )
        if source.get("contract_sha256") != corpus_manifest["contract_sha256"]:
            raise ValueError("prototype contract differs from switch corpus")
        prototype_nodes.append(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": identity[0],
                "npz": str(source_npz.resolve()),
                "npz_sha256": sha256_file(source_npz),
                "rollout_id": rollout_id,
            }
        )
        parameter_rows.append(parameters)

    record_source = json.loads(
        Path(corpus_manifest["teacher_manifest"]).read_text(encoding="utf-8")
    )
    records = [
        record
        for record in record_source["records"]
        if int(record["condition_index"]) == int(corpus_manifest["teacher_condition_index"])
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("switch corpus teacher condition cannot be resolved")
    record = records[0]
    spec = {
        "target_distance_m": float(record["distance_m"]),
        "target_angle_deg": float(record["angle_deg"]),
        "requested_ball_speed_mps": float(record["requested_speed_mps"]),
        "desired_arrival_speed_mps": float(record["desired_arrival_speed_mps"]),
        "action_mode": str(record["mode"]),
        "evaluation_duration_s": 3.0,
    }
    candidate_count = split.size
    prototype_count = len(parameter_rows)
    success = np.zeros((prototype_count, candidate_count), dtype=np.uint8)
    fall = np.zeros_like(success)
    contact = np.zeros_like(success)
    score = np.zeros((prototype_count, candidate_count), dtype=np.float32)
    tasks = [
        (prototype_index, candidate_index, qpos[candidate_index], qvel[candidate_index])
        for prototype_index in range(prototype_count)
        for candidate_index in range(candidate_count)
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_initialize,
        initargs=(spec, str(args.contract.resolve()), [row.tolist() for row in parameter_rows]),
    ) as pool:
        futures = [pool.submit(_worker_evaluate, task) for task in tasks]
        for completed_count, future in enumerate(as_completed(futures), 1):
            node = future.result()
            row = int(node["prototype_index"])
            column = int(node["candidate_index"])
            success[row, column] = bool(node["success"])
            fall[row, column] = bool(node["fall"])
            contact[row, column] = bool(node["contact"])
            score[row, column] = float(node["score"])
            if completed_count % 250 == 0 or completed_count == len(tasks):
                print(f"evaluated {completed_count}/{len(tasks)} trials", flush=True)

    train_rows = split == 0
    validation_rows = split == 1
    selected = greedy_rollout_cover(
        success,
        fall,
        rollout_ids,
        train_rows,
        maximum_prototypes=args.maximum_prototypes,
    )
    train_covered, train_total = rollout_coverage(
        success, rollout_ids, train_rows, selected
    )
    validation_covered, validation_total = rollout_coverage(
        success, rollout_ids, validation_rows, selected
    )
    all_indices = list(range(prototype_count))
    train_all_covered, _ = rollout_coverage(
        success, rollout_ids, train_rows, all_indices
    )
    validation_all_covered, _ = rollout_coverage(
        success, rollout_ids, validation_rows, all_indices
    )
    arrays = {
        "prototype_rollout_id": np.asarray(
            [node["rollout_id"] for node in prototype_nodes], dtype=np.int32
        ),
        "approach_rollout_id": rollout_ids,
        "split": split,
        "success": success,
        "fall": fall,
        "contact": contact,
        "score": score,
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    report = {
        "schema_version": 1,
        "purpose": "exact_cpu_kick_switch_prototype_bank_coverage",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "oracle coverage only; deployable trigger/action selector remains required",
        "switch_corpus": str(args.switch_corpus.resolve()),
        "switch_corpus_sha256": sha256_file(args.switch_corpus),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "candidate_entries": candidate_count,
        "prototypes": prototype_nodes,
        "maximum_selected_prototypes": args.maximum_prototypes,
        "greedy_selection_source": "training_approach_rollouts_only",
        "selected_prototype_indices": selected,
        "selected_prototype_rollout_ids": [
            prototype_nodes[index]["rollout_id"] for index in selected
        ],
        "train_rollouts_covered": train_covered,
        "train_rollouts": train_total,
        "train_oracle_coverage": train_covered / train_total,
        "validation_rollouts_covered": validation_covered,
        "validation_rollouts": validation_total,
        "validation_oracle_coverage": validation_covered / validation_total,
        "all_bank_train_rollouts_covered": train_all_covered,
        "all_bank_train_oracle_coverage": train_all_covered / train_total,
        "all_bank_validation_rollouts_covered": validation_all_covered,
        "all_bank_validation_oracle_coverage": (
            validation_all_covered / validation_total
        ),
        "selected_bank_candidate_falls": int(
            np.count_nonzero(np.any(fall[selected], axis=0)) if selected else 0
        ),
        "per_prototype": [
            {
                **prototype_nodes[index],
                "training_candidate_successes": int(np.count_nonzero(success[index, train_rows])),
                "validation_candidate_successes": int(np.count_nonzero(success[index, validation_rows])),
                "candidate_falls": int(np.count_nonzero(fall[index])),
            }
            for index in range(prototype_count)
        ],
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
