#!/usr/bin/env python3
"""Evaluate a standalone kick-policy-v3 ONNX on untouched transition states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, kick_trial_success
from tools.generate_kick_switch_window_corpus import sha256_file


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def held_out_rows(split: np.ndarray, rollout_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(split, dtype=np.uint8)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    if values.ndim != 1 or ids.shape != values.shape:
        raise ValueError("split and rollout IDs must be aligned vectors")
    if not set(values.tolist()) <= {0, 1}:
        raise ValueError("split must contain only zero and one")
    if set(ids[values == 0].tolist()) & set(ids[values == 1].tolist()):
        raise ValueError("transition corpus leaks rollout IDs")
    rows = np.flatnonzero(values == 1)
    if rows.size < 2:
        raise ValueError("transition corpus has fewer than two held-out rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--correction-onnx", type=Path)
    parser.add_argument("--correction-scale", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3":
        raise ValueError("transition ONNX evaluation requires kick_policy_v3")
    session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].shape != list(contract.input_shape):
        raise ValueError("ONNX input does not match the selected contract")
    if session.get_outputs()[0].shape != list(contract.output_shape):
        raise ValueError("ONNX output does not match the selected contract")
    correction_session = None
    if args.correction_onnx is not None:
        correction_session = ort.InferenceSession(
            str(args.correction_onnx.resolve()), providers=["CPUExecutionProvider"]
        )
        if (
            correction_session.get_inputs()[0].shape != list(contract.input_shape)
            or correction_session.get_outputs()[0].shape
            != list(contract.output_shape)
        ):
            raise ValueError("correction ONNX does not match the selected contract")
    if not 0.0 < args.correction_scale <= 0.5:
        raise ValueError("correction scale must be in (0, 0.5]")

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
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        rollout_ids = np.asarray(archive["rollout_id"], dtype=np.int32)
        phase_bucket = np.asarray(archive["phase_bucket"], dtype=np.int32)
        split = np.asarray(archive["split"], dtype=np.uint8)
    rows = held_out_rows(split, rollout_ids)

    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(
            target_distance_m=float(record["distance_m"]),
            target_angle_deg=float(record["angle_deg"]),
            requested_ball_speed_mps=float(record["requested_speed_mps"]),
            desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
            action_mode=str(record["mode"]),
            evaluation_duration_s=3.0,
        ),
        contract=contract,
    )
    trials: list[dict[str, object]] = []
    for trial_index, row in enumerate(rows):
        metrics = evaluator.rollout(
            None,
            initial_qpos=qpos[row],
            initial_qvel=qvel[row],
            kick_policy_session=session,
            kick_correction_session=correction_session,
            kick_correction_scale=args.correction_scale,
        )
        trials.append(
            {
                "trial": trial_index,
                "corpus_index": int(row),
                "rollout_id": int(rollout_ids[row]),
                "phase_bucket": int(phase_bucket[row]),
                "success": bool(kick_trial_success(metrics)),
                "metrics": metrics,
            }
        )
        if (trial_index + 1) % 20 == 0 or trial_index + 1 == rows.size:
            print(f"evaluated {trial_index + 1}/{rows.size} held-out states", flush=True)

    successful = sum(bool(trial["success"]) for trial in trials)
    required = int(np.ceil(0.9 * len(trials)))
    report = {
        "schema_version": 1,
        "purpose": "kick_policy_v3_untouched_transition_closed_loop_evaluation",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "requires three seeds, independent corpus and server gates",
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "correction_onnx": (
            str(args.correction_onnx.resolve())
            if args.correction_onnx is not None
            else None
        ),
        "correction_onnx_sha256": (
            sha256_file(args.correction_onnx)
            if args.correction_onnx is not None
            else None
        ),
        "correction_scale": args.correction_scale,
        "transition_corpus": str(args.transition_corpus.resolve()),
        "transition_corpus_sha256": sha256_file(args.transition_corpus),
        "transition_corpus_manifest": str(corpus_manifest_path.resolve()),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "trial_count": len(trials),
        "successful_trials": successful,
        "success_rate": successful / len(trials),
        "contact_trials": sum(bool(trial["metrics"]["contact"]) for trial in trials),
        "fall_trials": sum(bool(trial["metrics"]["fell"]) for trial in trials),
        "phase_buckets": sorted(set(phase_bucket[rows].tolist())),
        "gate": {"required_successes": required, "passed": successful >= required},
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("trial_count", "successful_trials", "success_rate", "contact_trials", "fall_trials", "gate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
