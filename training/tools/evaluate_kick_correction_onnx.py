#!/usr/bin/env python3
"""Evaluate a versioned kick correction in exact CPU MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_env import DEFAULT_CONTRACT
from my3d_rl.kick_teacher import (
    KickTeacherEvaluator,
    KickTeacherSpec,
    kick_trial_success,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forward_records(source: dict[str, object]) -> list[dict[str, object]]:
    records = [
        record
        for record in source["records"]
        if bool(record["accepted"])
        and record["mode"] == "pass"
        and abs(float(record["distance_m"]) - 2.0) < 1.0e-9
        and abs(float(record["angle_deg"])) < 1.0e-9
    ]
    if not records:
        raise ValueError("teacher manifest has no accepted 2 m forward-pass records")
    return records


def _load_validation_entries(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("npz_sha256") != _sha256(path):
        raise ValueError("transition corpus NPZ hash does not match its manifest")
    with np.load(path, allow_pickle=False) as archive:
        split = np.asarray(archive["split"], dtype=np.uint8)
        validation = split == 1
        qpos = np.asarray(archive["qpos"], dtype=np.float64)[validation]
        qvel = np.asarray(archive["qvel"], dtype=np.float64)[validation]
        rollout_id = np.asarray(archive["rollout_id"], dtype=np.int32)[validation]
        phase_bucket = np.asarray(archive["phase_bucket"], dtype=np.int32)[validation]
    if qpos.shape[0] < 2 or qvel.shape[0] != qpos.shape[0]:
        raise ValueError("transition corpus has fewer than two validation entries")
    return qpos, qvel, rollout_id, phase_bucket, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("correction_onnx", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--transition-corpus", type=Path)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--ball-x-min", type=float, default=-0.01)
    parser.add_argument("--ball-x-max", type=float, default=0.08)
    parser.add_argument("--ball-y-min", type=float, default=-0.08)
    parser.add_argument("--ball-y-max", type=float, default=0.08)
    parser.add_argument("--correction-scale", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials is not None and args.trials < 1:
        raise ValueError("--trials must be positive")

    contract = load_policy_contract(args.contract)
    source = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    records = _forward_records(source)
    session = ort.InferenceSession(
        str(args.correction_onnx.resolve()), providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].shape != list(contract.input_shape):
        raise ValueError("correction ONNX input does not match the selected contract")
    if session.get_outputs()[0].shape != list(contract.output_shape):
        raise ValueError("correction ONNX output does not match the selected contract")

    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(
            target_distance_m=2.0,
            target_angle_deg=0.0,
            requested_ball_speed_mps=1.43,
            desired_arrival_speed_mps=0.8,
            action_mode="pass",
        ),
        contract=contract,
    )
    rng = np.random.default_rng(args.seed)
    trials: list[dict[str, object]] = []
    evaluation_mode = "random_ball_pose"
    corpus_evidence: dict[str, object] | None = None

    if args.transition_corpus is not None:
        if contract.policy_name != "kick_policy_v3":
            raise ValueError("transition-corpus evaluation requires kick_policy_v3")
        qpos, qvel, rollout_ids, phase_bucket, corpus_manifest = (
            _load_validation_entries(args.transition_corpus)
        )
        if corpus_manifest.get("contract_sha256") != _sha256(args.contract):
            raise ValueError("transition corpus contract hash mismatch")
        condition_index = int(corpus_manifest["teacher_condition_index"])
        matches = [
            record
            for record in records
            if int(record["condition_index"]) == condition_index
        ]
        if len(matches) != 1:
            raise ValueError("corpus teacher condition is not uniquely available")
        selected = matches[0]
        row_indices = np.arange(qpos.shape[0])
        if args.trials is not None:
            row_indices = rng.choice(
                row_indices,
                size=min(args.trials, row_indices.size),
                replace=False,
            )
        for trial_index, row_index in enumerate(row_indices):
            baseline_metrics = evaluator.rollout(
                np.asarray(selected["parameters"], dtype=np.float64),
                initial_qpos=qpos[row_index],
                initial_qvel=qvel[row_index],
            )
            metrics = evaluator.rollout(
                np.asarray(selected["parameters"], dtype=np.float64),
                initial_qpos=qpos[row_index],
                initial_qvel=qvel[row_index],
                kick_correction_session=session,
                kick_correction_scale=args.correction_scale,
            )
            trials.append(
                {
                    "trial": trial_index,
                    "rollout_id": int(rollout_ids[row_index]),
                    "phase_bucket": int(phase_bucket[row_index]),
                    "selected_condition_index": condition_index,
                    "baseline_success": kick_trial_success(baseline_metrics),
                    "success": kick_trial_success(metrics),
                    "baseline_metrics": baseline_metrics,
                    "metrics": metrics,
                }
            )
        evaluation_mode = "held_out_transition_rollouts"
        corpus_evidence = {
            "npz": str(args.transition_corpus.resolve()),
            "npz_sha256": _sha256(args.transition_corpus),
            "manifest": str(args.transition_corpus.with_suffix(".json").resolve()),
            "phase_buckets": sorted(set(phase_bucket[row_indices].tolist())),
        }
    else:
        trial_count = args.trials if args.trials is not None else 300
        for trial_index in range(trial_count):
            ball_x = float(rng.uniform(args.ball_x_min, args.ball_x_max))
            ball_y = float(rng.uniform(args.ball_y_min, args.ball_y_max))

            def distance(record: dict[str, object]) -> float:
                return float(
                    ((float(record["ball_x_offset_m"]) - ball_x) / 0.045) ** 2
                    + ((float(record["ball_y_offset_m"]) - ball_y) / 0.04) ** 2
                )

            selected = min(records, key=distance)
            baseline_metrics = evaluator.rollout(
                np.asarray(selected["parameters"], dtype=np.float64),
                ball_x_offset_m=ball_x,
                ball_y_offset_m=ball_y,
            )
            metrics = evaluator.rollout(
                np.asarray(selected["parameters"], dtype=np.float64),
                ball_x_offset_m=ball_x,
                ball_y_offset_m=ball_y,
                kick_correction_session=session,
                kick_correction_scale=args.correction_scale,
            )
            trials.append(
                {
                    "trial": trial_index,
                    "ball_x_offset_m": ball_x,
                    "ball_y_offset_m": ball_y,
                    "selected_condition_index": int(selected["condition_index"]),
                    "baseline_success": kick_trial_success(baseline_metrics),
                    "success": kick_trial_success(metrics),
                    "baseline_metrics": baseline_metrics,
                    "metrics": metrics,
                }
            )

    successful = sum(bool(trial["success"]) for trial in trials)
    baseline_successful = sum(bool(trial["baseline_success"]) for trial in trials)
    required = int(np.ceil(0.9 * len(trials)))
    exact_cpu_passed = successful >= required
    report = {
        "purpose": "exact_cpu_versioned_kick_correction_evaluation",
        "promotable": False,
        "promotion_blocker": (
            "requires ONNX/source parity, three seeds, phase coverage and server gates"
        ),
        "evaluation_mode": evaluation_mode,
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "correction_onnx": str(args.correction_onnx.resolve()),
        "correction_onnx_sha256": _sha256(args.correction_onnx),
        "transition_corpus": corpus_evidence,
        "correction_scale": args.correction_scale,
        "seed": args.seed,
        "trial_count": len(trials),
        "successful_trials": successful,
        "success_rate": successful / len(trials),
        "baseline_successful_trials": baseline_successful,
        "baseline_success_rate": baseline_successful / len(trials),
        "contact_trials": sum(bool(trial["metrics"]["contact"]) for trial in trials),
        "fall_trials": sum(bool(trial["metrics"]["fell"]) for trial in trials),
        "gate": {"required_successes": required, "passed": exact_cpu_passed},
        "trials": trials,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
