#!/usr/bin/env python3
"""Evaluate one frozen kick selector on a new exact-CPU outcome bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from my3d_rl.kick_switch_selector import sequential_policy_metrics
from tools.generate_kick_switch_window_corpus import sha256_file


def fixed_cycle_policy_metrics(
    success: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    confirmation_cycles: np.ndarray,
    rows: np.ndarray,
    *,
    prototype_index: int,
    release_cycle: int,
) -> dict[str, Any]:
    """Evaluate a fixed prototype at one predeclared confirmation cycle."""
    labels = np.asarray(success, dtype=bool)
    unsafe = np.asarray(fall, dtype=bool)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    cycles = np.asarray(confirmation_cycles, dtype=np.int64)
    selected_rows = np.asarray(rows, dtype=bool)
    if (
        labels.ndim != 2
        or unsafe.shape != labels.shape
        or ids.shape != cycles.shape
        or selected_rows.shape != ids.shape
        or labels.shape[1] != ids.size
        or prototype_index not in range(labels.shape[0])
        or release_cycle < 1
    ):
        raise ValueError("fixed-cycle policy inputs are invalid or misaligned")

    decisions: list[dict[str, int | bool]] = []
    for rollout_id in np.unique(ids[selected_rows]):
        matches = np.flatnonzero(
            selected_rows & (ids == rollout_id) & (cycles == release_cycle)
        )
        if matches.size == 0:
            continue
        row = int(matches[0])
        decisions.append(
            {
                "rollout_id": int(rollout_id),
                "row": row,
                "confirmation_cycles": int(cycles[row]),
                "prototype_index": prototype_index,
                "success": bool(labels[prototype_index, row]),
                "fall": bool(unsafe[prototype_index, row]),
            }
        )
    rollouts = int(np.unique(ids[selected_rows]).size)
    releases = len(decisions)
    successes = sum(int(node["success"]) for node in decisions)
    falls = sum(int(node["fall"]) for node in decisions)
    return {
        "rollouts": rollouts,
        "releases": releases,
        "successes": successes,
        "falls": falls,
        "release_rate": releases / rollouts if rollouts else 0.0,
        "success_rate": successes / rollouts if rollouts else 0.0,
        "release_precision": successes / releases if releases else 0.0,
        "decisions": decisions,
    }


def paired_success_counts(
    selector_decisions: list[dict[str, Any]],
    baseline_decisions: list[dict[str, Any]],
    rollout_ids: np.ndarray | None = None,
) -> dict[str, int]:
    """Compare success on matching rollouts; abstention is not success."""
    selector = {
        int(node["rollout_id"]): bool(node["success"]) for node in selector_decisions
    }
    baseline = {
        int(node["rollout_id"]): bool(node["success"]) for node in baseline_decisions
    }
    universe = set(selector) | set(baseline)
    if rollout_ids is not None:
        universe |= set(np.asarray(rollout_ids, dtype=np.int64).tolist())
    selector_only = sum(
        selector.get(value, False) and not baseline.get(value, False)
        for value in universe
    )
    baseline_only = sum(
        baseline.get(value, False) and not selector.get(value, False)
        for value in universe
    )
    both = sum(
        selector.get(value, False) and baseline.get(value, False)
        for value in universe
    )
    return {
        "compared_rollouts": len(universe),
        "both_succeed": both,
        "selector_only_succeeds": selector_only,
        "baseline_only_succeeds": baseline_only,
        "neither_succeeds": len(universe) - selector_only - baseline_only - both,
        "net_success_advantage": selector_only - baseline_only,
    }


def _manifest(path: Path, purpose: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("purpose") != purpose or value.get("status") != "complete":
        raise ValueError(f"{path} is not a complete {purpose} manifest")
    return value


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "decisions"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("switch_corpus", type=Path)
    parser.add_argument("prototype_bank", type=Path)
    parser.add_argument("selector_manifest", type=Path)
    parser.add_argument("--baseline-prototype-rollout-id", type=int, default=65)
    parser.add_argument("--baseline-confirmation-cycles", type=int, default=39)
    parser.add_argument(
        "--row-selection", choices=("all", "validation"), default="all"
    )
    parser.add_argument("--allow-source-corpus", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.baseline_confirmation_cycles < 1
        or not args.output.is_absolute()
        or args.output.is_relative_to(Path.cwd())
    ):
        raise ValueError("evaluation arguments are invalid")
    if args.output.exists():
        raise FileExistsError("evaluation output already exists")

    corpus = _manifest(
        args.switch_corpus.with_suffix(".json"),
        "exact_cpu_walk_to_kick_switch_window_corpus",
    )
    bank = _manifest(
        args.prototype_bank.with_suffix(".json"),
        "exact_cpu_kick_switch_prototype_bank_coverage",
    )
    selector = _manifest(
        args.selector_manifest, "causal_exact_cpu_kick_switch_prototype_selector"
    )
    corpus_hash = sha256_file(args.switch_corpus)
    bank_hash = sha256_file(args.prototype_bank)
    if corpus.get("npz_sha256") != corpus_hash:
        raise ValueError("switch corpus hash mismatch")
    if bank.get("npz_sha256") != bank_hash or bank.get(
        "switch_corpus_sha256"
    ) != corpus_hash:
        raise ValueError("prototype bank does not match the switch corpus")
    independent = corpus_hash != selector.get("switch_corpus_sha256")
    if not independent and not args.allow_source_corpus:
        raise ValueError(
            "selector source corpus is not independent; use --allow-source-corpus "
            "only to reproduce its original validation split"
        )
    if selector.get("feature_profile") != "current_state_v1" or selector[
        "calibration"
    ].get("selected_fallback") is not None:
        raise ValueError("this evaluator supports the frozen current-state gate only")
    selector_onnx = Path(str(selector["onnx"]))
    if selector.get("onnx_sha256") != sha256_file(selector_onnx):
        raise ValueError("selector ONNX hash mismatch")

    with np.load(args.switch_corpus, allow_pickle=False) as archive:
        observations = np.asarray(archive["actor_observation"], dtype=np.float32)
        rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        cycles = np.asarray(archive["confirmation_cycles"], dtype=np.int32)
        split = np.asarray(archive["split"], dtype=np.uint8)
    with np.load(args.prototype_bank, allow_pickle=False) as archive:
        prototype_ids = np.asarray(archive["prototype_rollout_id"], dtype=np.int32)
        bank_rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        success = np.asarray(archive["success"], dtype=np.uint8)
        fall = np.asarray(archive["fall"], dtype=np.uint8)
    if (
        observations.shape != (rollout_ids.size, 98)
        or cycles.shape != rollout_ids.shape
        or split.shape != rollout_ids.shape
        or not np.array_equal(rollout_ids, bank_rollout_ids)
        or success.shape != fall.shape
        or success.shape != (prototype_ids.size, rollout_ids.size)
    ):
        raise ValueError("corpus and prototype bank arrays are misaligned")

    index_by_id = {int(value): index for index, value in enumerate(prototype_ids)}
    selected_ids = [int(value) for value in selector["selected_prototype_rollout_ids"]]
    if not set(selected_ids) <= set(index_by_id):
        raise ValueError("prototype bank is missing a selector action")
    selected_indices = tuple(index_by_id[value] for value in selected_ids)
    if args.baseline_prototype_rollout_id not in index_by_id:
        raise ValueError("prototype bank is missing the fixed baseline")

    session = ort.InferenceSession(
        str(selector_onnx), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    probabilities = np.vstack(
        [session.run(None, {input_name: row[None, :]})[0][0] for row in observations]
    ).astype(np.float32)
    rows = np.ones(rollout_ids.size, dtype=bool)
    if args.row_selection == "validation":
        rows = split == 1
    if not np.any(rows):
        raise ValueError("row selection is empty")

    calibration = selector["calibration"]
    selector_metrics = sequential_policy_metrics(
        success,
        fall,
        rollout_ids,
        cycles,
        probabilities,
        rows,
        prototype_indices=selected_indices,
        threshold=float(calibration["selected_threshold"]),
        consecutive_frames=int(calibration["selected_consecutive_frames"]),
    )
    baseline_metrics = fixed_cycle_policy_metrics(
        success,
        fall,
        rollout_ids,
        cycles,
        rows,
        prototype_index=index_by_id[args.baseline_prototype_rollout_id],
        release_cycle=args.baseline_confirmation_cycles,
    )
    comparison = paired_success_counts(
        selector_metrics["decisions"],
        baseline_metrics["decisions"],
        np.unique(rollout_ids[rows]),
    )
    gate_passed = bool(
        independent
        and selector_metrics["falls"] <= baseline_metrics["falls"]
        and comparison["net_success_advantage"] > 0
    )
    report = {
        "schema_version": 1,
        "purpose": "independent_frozen_kick_switch_selector_evaluation",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": (
            "multiple independent seeds and RCSSServerMJ replay remain required"
            if gate_passed
            else (
                "source-corpus reproduction is not an independent gate"
                if not independent
                else "the independent exact-CPU repeatability gate did not pass"
            )
        ),
        "exact_cpu_repeatability_gate_passed": gate_passed,
        "independent_corpus": independent,
        "row_selection": args.row_selection,
        "inputs": {
            "switch_corpus": str(args.switch_corpus.resolve()),
            "switch_corpus_sha256": corpus_hash,
            "prototype_bank": str(args.prototype_bank.resolve()),
            "prototype_bank_sha256": bank_hash,
            "selector_manifest": str(args.selector_manifest.resolve()),
            "selector_onnx": str(selector_onnx.resolve()),
        },
        "selector_gate": {
            "threshold": float(calibration["selected_threshold"]),
            "consecutive_frames": int(calibration["selected_consecutive_frames"]),
        },
        "comparison_gate": {
            "strictly_more_target_successes": True,
            "no_more_falls_than_fixed_baseline": True,
        },
        "selector_metrics": _summary(selector_metrics),
        "baseline": {
            "prototype_rollout_id": args.baseline_prototype_rollout_id,
            "confirmation_cycles": args.baseline_confirmation_cycles,
            "metrics": _summary(baseline_metrics),
        },
        "paired_comparison": comparison,
        "selector_decisions": selector_metrics["decisions"],
        "baseline_decisions": baseline_metrics["decisions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "independent_corpus": independent,
                "exact_cpu_repeatability_gate_passed": gate_passed,
                "selector_metrics": report["selector_metrics"],
                "baseline": report["baseline"],
                "paired_comparison": comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
