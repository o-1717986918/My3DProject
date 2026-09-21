#!/usr/bin/env python3
"""Blind-evaluate one frozen release-time kick expert gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from my3d_rl.kick_expert_gate import kick_forward_drive_success
from tools.generate_kick_switch_window_corpus import sha256_file


def paired_choice_counts(
    gate_success: np.ndarray, baseline_success: np.ndarray
) -> dict[str, int | float]:
    """Report paired outcomes on the exact same release states."""
    gate = np.asarray(gate_success, dtype=bool)
    baseline = np.asarray(baseline_success, dtype=bool)
    if gate.ndim != 1 or baseline.shape != gate.shape:
        raise ValueError("paired kick outcomes must be aligned vectors")
    both = int(np.count_nonzero(gate & baseline))
    gate_only = int(np.count_nonzero(gate & ~baseline))
    baseline_only = int(np.count_nonzero(~gate & baseline))
    discordant = gate_only + baseline_only
    # Exact one-sided paired sign test under equal action quality.  This avoids
    # calling a 2-vs-1 result proof of improvement while adding no large-sample
    # or zero-fall assumption.
    one_sided_p = (
        sum(math.comb(discordant, wins) for wins in range(gate_only, discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "entries": int(gate.size),
        "both_succeed": both,
        "gate_only_succeeds": gate_only,
        "baseline_only_succeeds": baseline_only,
        "neither_succeeds": int(gate.size - both - gate_only - baseline_only),
        "net_success_advantage": gate_only - baseline_only,
        "one_sided_paired_sign_test_p": float(one_sided_p),
    }


def _manifest(path: Path, purpose: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("purpose") != purpose:
        raise ValueError(f"{path} is not a {purpose} manifest")
    if "status" in value and value["status"] != "complete":
        raise ValueError(f"{path} is incomplete")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("prototype_bank", type=Path)
    parser.add_argument("gate_manifest", type=Path)
    parser.add_argument("--baseline-prototype-rollout-id", type=int, default=164464)
    parser.add_argument(
        "--primary-contract",
        choices=("strict-pass", "forward-drive"),
        default="strict-pass",
    )
    parser.add_argument(
        "--row-selection", choices=("all", "train", "validation"), default="all"
    )
    parser.add_argument("--allow-source-corpus", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be absolute and outside the repository")
    if args.output.exists():
        raise FileExistsError("frozen gate evaluation output already exists")

    corpus = _manifest(
        args.transition_corpus.with_suffix(".json"),
        "kick_policy_v3_walk_to_kick_transition_corpus",
    )
    bank = _manifest(
        args.prototype_bank.with_suffix(".json"),
        "exact_cpu_kick_switch_prototype_bank_coverage",
    )
    gate = _manifest(
        args.gate_manifest,
        "match_grouped_release_time_kick_expert_gate",
    )
    corpus_hash = sha256_file(args.transition_corpus)
    bank_hash = sha256_file(args.prototype_bank)
    if corpus.get("npz_sha256") != corpus_hash:
        raise ValueError("transition corpus hash mismatch")
    if (
        bank.get("npz_sha256") != bank_hash
        or bank.get("switch_corpus_sha256") != corpus_hash
    ):
        raise ValueError("prototype bank does not match transition corpus")
    independent = corpus_hash != gate.get("transition_corpus_sha256")
    if not independent and not args.allow_source_corpus:
        raise ValueError("gate source corpus is not an independent evaluation")
    gate_onnx = Path(str(gate["onnx"]))
    if gate.get("onnx_sha256") != sha256_file(gate_onnx):
        raise ValueError("frozen gate ONNX hash mismatch")

    with np.load(args.transition_corpus, allow_pickle=False) as archive:
        rollout_ids = np.asarray(archive["rollout_id"], dtype=np.int32)
        split = np.asarray(archive["split"], dtype=np.uint8)
        source_match_ids = np.asarray(archive["source_match_id"], dtype=np.int32)
    with np.load(args.prototype_bank, allow_pickle=False) as archive:
        prototype_ids = np.asarray(archive["prototype_rollout_id"], dtype=np.int32)
        bank_rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        bank_split = np.asarray(archive["split"], dtype=np.uint8)
        observations = np.asarray(archive["actor_observation"], dtype=np.float32)
        success = np.asarray(archive["success"], dtype=np.uint8)
        fall = np.asarray(archive["fall"], dtype=np.uint8)
        physical_score = np.asarray(archive["score"], dtype=np.float32)
        contact = np.asarray(archive["contact"], dtype=bool)
        maximum_progress = np.asarray(
            archive["maximum_progress_m"], dtype=np.float32
        )
        lateral_error = np.asarray(archive["lateral_error_m"], dtype=np.float32)
        directional_speed = np.asarray(
            archive["maximum_directional_speed_mps"], dtype=np.float32
        )
    if (
        observations.shape != (rollout_ids.size, 98)
        or split.shape != rollout_ids.shape
        or source_match_ids.shape != rollout_ids.shape
        or not np.array_equal(rollout_ids, bank_rollout_ids)
        or not np.array_equal(split, bank_split)
        or success.shape != fall.shape
        or success.shape != physical_score.shape
        or success.shape != (prototype_ids.size, rollout_ids.size)
    ):
        raise ValueError("frozen gate evaluation arrays are misaligned")

    index_by_id = {int(value): index for index, value in enumerate(prototype_ids)}
    selected_ids = [int(value) for value in gate["prototype_rollout_ids"]]
    if not set(selected_ids) <= set(index_by_id):
        raise ValueError("new bank is missing a frozen gate expert")
    if args.baseline_prototype_rollout_id not in index_by_id:
        raise ValueError("new bank is missing the fixed baseline expert")
    selected_indices = np.asarray(
        [index_by_id[value] for value in selected_ids], dtype=np.int32
    )
    baseline_index = index_by_id[args.baseline_prototype_rollout_id]

    rows = np.ones(rollout_ids.size, dtype=bool)
    if args.row_selection == "train":
        rows = split == 0
    elif args.row_selection == "validation":
        rows = split == 1
    if not np.any(rows):
        raise ValueError("frozen gate row selection is empty")

    session = ort.InferenceSession(
        str(gate_onnx), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    predicted = np.vstack(
        [session.run(None, {input_name: row[None, :]})[0][0] for row in observations]
    ).astype(np.float32)
    if predicted.shape != (rollout_ids.size, selected_indices.size):
        raise ValueError("frozen gate ONNX output is incompatible with its manifest")
    row_indices = np.flatnonzero(rows)
    choices = selected_indices[np.argmax(predicted[rows], axis=1)]
    gate_success = success[choices, row_indices].astype(bool)
    gate_fall = fall[choices, row_indices].astype(bool)
    gate_score = physical_score[choices, row_indices]
    baseline_success = success[baseline_index, row_indices].astype(bool)
    baseline_fall = fall[baseline_index, row_indices].astype(bool)
    baseline_score = physical_score[baseline_index, row_indices]
    forward_drive = kick_forward_drive_success(
        contact, maximum_progress, lateral_error, directional_speed
    )
    gate_forward_drive = forward_drive[choices, row_indices]
    baseline_forward_drive = forward_drive[baseline_index, row_indices]
    strict_paired = paired_choice_counts(gate_success, baseline_success)
    forward_drive_paired = paired_choice_counts(
        gate_forward_drive, baseline_forward_drive
    )
    paired = (
        strict_paired
        if args.primary_contract == "strict-pass"
        else forward_drive_paired
    )
    gate_metrics = {
        "entries": int(row_indices.size),
        "successes": int(gate_success.sum()),
        "success_rate": float(gate_success.mean()),
        "falls": int(gate_fall.sum()),
        "fall_rate": float(gate_fall.mean()),
        "mean_physical_score": float(gate_score.mean()),
        "used_experts": int(np.unique(choices).size),
    }
    baseline_metrics = {
        "entries": int(row_indices.size),
        "successes": int(baseline_success.sum()),
        "success_rate": float(baseline_success.mean()),
        "falls": int(baseline_fall.sum()),
        "fall_rate": float(baseline_fall.mean()),
        "mean_physical_score": float(baseline_score.mean()),
    }
    forward_drive_metrics = {
        "contract": (
            "contact, maximum_progress>=2.5m, lateral<=1.0m, "
            "directional_speed>=1.5m/s; falls reported separately"
        ),
        "gate_successes": int(gate_forward_drive.sum()),
        "gate_success_rate": float(gate_forward_drive.mean()),
        "baseline_successes": int(baseline_forward_drive.sum()),
        "baseline_success_rate": float(baseline_forward_drive.mean()),
        "paired_comparison": forward_drive_paired,
    }
    passed = bool(
        independent
        and paired["net_success_advantage"] > 0
        and paired["one_sided_paired_sign_test_p"] <= 0.10
        and gate_metrics["mean_physical_score"] > baseline_metrics["mean_physical_score"]
        and gate_metrics["fall_rate"] <= baseline_metrics["fall_rate"] + 0.10
    )
    decisions = [
        {
            "rollout_id": int(rollout_ids[row]),
            "source_match_id": int(source_match_ids[row]),
            "prototype_index": int(choice),
            "prototype_rollout_id": int(prototype_ids[choice]),
            "success": bool(gate_success[index]),
            "forward_drive_success": bool(gate_forward_drive[index]),
            "fall": bool(gate_fall[index]),
            "physical_score": float(gate_score[index]),
        }
        for index, (row, choice) in enumerate(zip(row_indices, choices, strict=True))
    ]
    report = {
        "schema_version": 1,
        "purpose": "independent_frozen_kick_expert_gate_evaluation",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": (
            "RCSS shadow replay and repeated independent seeds remain required"
            if passed
            else "frozen gate did not beat the fixed expert on independent exact CPU states"
        ),
        "independent_exact_cpu_gate_passed": passed,
        "independent_corpus": independent,
        "row_selection": args.row_selection,
        "primary_contract": args.primary_contract,
        "source_match_ids": np.unique(source_match_ids[rows]).astype(int).tolist(),
        "comparison_gate": {
            "strictly_more_paired_successes": True,
            "one_sided_paired_sign_test_max_p": 0.10,
            "higher_mean_finite_cost_physical_score": True,
            "maximum_added_fall_rate": 0.10,
        },
        "gate_metrics": gate_metrics,
        "forward_drive_metrics": forward_drive_metrics,
        "baseline": {
            "prototype_rollout_id": args.baseline_prototype_rollout_id,
            "metrics": baseline_metrics,
        },
        "paired_comparison": paired,
        "strict_pass_paired_comparison": strict_paired,
        "decisions": decisions,
        "inputs": {
            "transition_corpus": str(args.transition_corpus.resolve()),
            "transition_corpus_sha256": corpus_hash,
            "prototype_bank": str(args.prototype_bank.resolve()),
            "prototype_bank_sha256": bank_hash,
            "gate_manifest": str(args.gate_manifest.resolve()),
            "gate_onnx": str(gate_onnx.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
