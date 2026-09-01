#!/usr/bin/env python3
"""Train and blind-evaluate a causal kick switch/prototype selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax
import numpy as np
import onnxruntime as ort

from my3d_rl.kick_switch_selector import (
    apply_switch_selector_numpy,
    build_causal_sequence_features,
    export_switch_selector_onnx,
    grouped_fit_calibration_split,
    sequential_policy_metrics,
    train_switch_selector,
    verify_switch_selector_onnx,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _without_decisions(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "decisions"}


def _select_calibration_gate(
    success: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    confirmation_cycles: np.ndarray,
    probabilities: np.ndarray,
    rows: np.ndarray,
    *,
    prototype_indices: tuple[int, ...],
    minimum_precision: float,
    maximum_consecutive_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    grid: list[dict[str, Any]] = []
    for consecutive_frames in range(1, maximum_consecutive_frames + 1):
        for threshold in np.linspace(0.10, 0.99, 90):
            metrics = sequential_policy_metrics(
                success,
                fall,
                rollout_ids,
                confirmation_cycles,
                probabilities,
                rows,
                prototype_indices=prototype_indices,
                threshold=float(threshold),
                consecutive_frames=consecutive_frames,
            )
            grid.append(
                {
                    "threshold": float(threshold),
                    "consecutive_frames": consecutive_frames,
                    **_without_decisions(metrics),
                }
            )
    safe = [
        node
        for node in grid
        if node["falls"] == 0 and node["release_precision"] >= minimum_precision
    ]
    candidates = safe or [node for node in grid if node["falls"] == 0] or grid
    selected = max(
        candidates,
        key=lambda node: (
            node["successes"],
            node["release_precision"],
            -node["falls"],
            node["releases"],
            node["threshold"],
            -node["consecutive_frames"],
        ),
    )
    return selected, grid, bool(safe)


def _select_calibration_fallback(
    success: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    confirmation_cycles: np.ndarray,
    probabilities: np.ndarray,
    rows: np.ndarray,
    *,
    prototype_indices: tuple[int, ...],
    threshold: float,
    consecutive_frames: int,
    minimum_precision: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    baseline = sequential_policy_metrics(
        success,
        fall,
        rollout_ids,
        confirmation_cycles,
        probabilities,
        rows,
        prototype_indices=prototype_indices,
        threshold=threshold,
        consecutive_frames=consecutive_frames,
    )
    candidates: list[tuple[dict[str, int], dict[str, Any]]] = []
    for prototype_index in range(success.shape[0]):
        for cycle in np.unique(confirmation_cycles[rows]):
            config = {
                "fallback_prototype_index": prototype_index,
                "fallback_confirmation_cycles": int(cycle),
            }
            metrics = sequential_policy_metrics(
                success,
                fall,
                rollout_ids,
                confirmation_cycles,
                probabilities,
                rows,
                prototype_indices=prototype_indices,
                threshold=threshold,
                consecutive_frames=consecutive_frames,
                **config,
            )
            candidates.append((config, metrics))
    safe = [
        node
        for node in candidates
        if node[1]["falls"] == 0
        and node[1]["release_precision"] >= minimum_precision
    ]
    if not safe:
        return None, baseline, False
    config, metrics = max(
        safe,
        key=lambda node: (
            node[1]["successes"],
            node[1]["release_precision"],
            node[1]["releases"],
            node[0]["fallback_confirmation_cycles"],
            -node[0]["fallback_prototype_index"],
        ),
    )
    if metrics["successes"] <= baseline["successes"]:
        return None, baseline, True
    return config, metrics, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("switch_corpus", type=Path)
    parser.add_argument("prototype_bank", type=Path)
    parser.add_argument("--seed", type=int, default=10801)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--fall-weight", type=float, default=8.0)
    parser.add_argument("--minimum-calibration-precision", type=float, default=0.95)
    parser.add_argument("--maximum-consecutive-frames", type=int, default=3)
    parser.add_argument(
        "--feature-profile",
        choices=("current_state_v1", "anchor_context_v2"),
        default="anchor_context_v2",
    )
    parser.add_argument("--cycle-normalizer", type=float, default=60.0)
    parser.add_argument(
        "--prototype-set", choices=("selected", "all"), default="selected"
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.learning_rate <= 0.0
        or args.fall_weight < 0.0
        or not 0.5 <= args.minimum_calibration_precision <= 1.0
        or args.maximum_consecutive_frames < 1
        or not np.isfinite(args.cycle_normalizer)
        or args.cycle_normalizer <= 0.0
    ):
        raise ValueError("selector training or calibration settings are invalid")
    json_path = args.output_prefix.with_suffix(".json")
    npz_path = args.output_prefix.with_suffix(".npz")
    onnx_path = args.output_prefix.with_suffix(".onnx")
    if not json_path.is_absolute() or json_path.is_relative_to(Path.cwd()):
        raise ValueError("output prefix must be absolute and outside the repository")
    if any(path.exists() for path in (json_path, npz_path, onnx_path)):
        raise FileExistsError("selector outputs already exist")

    corpus_manifest_path = args.switch_corpus.with_suffix(".json")
    bank_manifest_path = args.prototype_bank.with_suffix(".json")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("purpose")
        != "exact_cpu_walk_to_kick_switch_window_corpus"
        or corpus_manifest.get("npz_sha256") != sha256_file(args.switch_corpus)
    ):
        raise ValueError("switch corpus is invalid or has a hash mismatch")
    if (
        bank_manifest.get("purpose")
        != "exact_cpu_kick_switch_prototype_bank_coverage"
        or bank_manifest.get("npz_sha256") != sha256_file(args.prototype_bank)
        or bank_manifest.get("switch_corpus_sha256")
        != corpus_manifest["npz_sha256"]
    ):
        raise ValueError("prototype bank is invalid or does not match the corpus")

    with np.load(args.switch_corpus, allow_pickle=False) as archive:
        required = {
            "actor_observation",
            "approach_rollout_id",
            "confirmation_cycles",
            "split",
        }
        if not required <= set(archive.files):
            raise ValueError("switch corpus is missing selector arrays")
        observations = np.asarray(archive["actor_observation"], dtype=np.float32)
        rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        confirmation_cycles = np.asarray(
            archive["confirmation_cycles"], dtype=np.int32
        )
        split = np.asarray(archive["split"], dtype=np.uint8)
    with np.load(args.prototype_bank, allow_pickle=False) as archive:
        required = {"prototype_rollout_id", "approach_rollout_id", "split", "success", "fall"}
        if not required <= set(archive.files):
            raise ValueError("prototype bank is missing selector labels")
        prototype_rollout_ids = np.asarray(
            archive["prototype_rollout_id"], dtype=np.int32
        )
        bank_rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        bank_split = np.asarray(archive["split"], dtype=np.uint8)
        success = np.asarray(archive["success"], dtype=np.uint8)
        fall = np.asarray(archive["fall"], dtype=np.uint8)
    if (
        observations.shape[0] != rollout_ids.size
        or confirmation_cycles.shape != rollout_ids.shape
        or split.shape != rollout_ids.shape
        or not np.array_equal(rollout_ids, bank_rollout_ids)
        or not np.array_equal(split, bank_split)
        or success.shape != fall.shape
        or success.shape != (prototype_rollout_ids.size, rollout_ids.size)
    ):
        raise ValueError("corpus and prototype-bank arrays are misaligned")
    model_observations = (
        build_causal_sequence_features(
            observations,
            rollout_ids,
            confirmation_cycles,
            cycle_normalizer=args.cycle_normalizer,
        )
        if args.feature_profile == "anchor_context_v2"
        else observations
    )
    prototype_indices = (
        tuple(range(prototype_rollout_ids.size))
        if args.prototype_set == "all"
        else tuple(
            int(value) for value in bank_manifest["selected_prototype_indices"]
        )
    )
    if not prototype_indices:
        raise ValueError("prototype bank selected no training action set")

    train_rows = split == 0
    validation_rows = split == 1
    fit_ids, calibration_ids = grouped_fit_calibration_split(
        rollout_ids,
        train_rows,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    result = train_switch_selector(
        model_observations,
        success,
        fall,
        rollout_ids,
        prototype_indices=prototype_indices,
        fit_rollout_ids=fit_ids,
        calibration_rollout_ids=calibration_ids,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fall_weight=args.fall_weight,
    )
    probabilities = apply_switch_selector_numpy(result, model_observations)
    calibration_rows = np.isin(rollout_ids, calibration_ids)
    fit_rows = np.isin(rollout_ids, fit_ids)
    selected_gate, calibration_grid, precision_gate_available = (
        _select_calibration_gate(
            success,
            fall,
            rollout_ids,
            confirmation_cycles,
            probabilities,
            calibration_rows,
            prototype_indices=prototype_indices,
            minimum_precision=args.minimum_calibration_precision,
            maximum_consecutive_frames=args.maximum_consecutive_frames,
        )
    )
    fallback, _, fallback_precision_gate_available = _select_calibration_fallback(
        success,
        fall,
        rollout_ids,
        confirmation_cycles,
        probabilities,
        calibration_rows,
        prototype_indices=prototype_indices,
        threshold=float(selected_gate["threshold"]),
        consecutive_frames=int(selected_gate["consecutive_frames"]),
        minimum_precision=args.minimum_calibration_precision,
    )
    gate_kwargs = {
        "prototype_indices": prototype_indices,
        "threshold": float(selected_gate["threshold"]),
        "consecutive_frames": int(selected_gate["consecutive_frames"]),
    }
    if fallback is not None:
        gate_kwargs.update(fallback)
    fit_metrics = sequential_policy_metrics(
        success,
        fall,
        rollout_ids,
        confirmation_cycles,
        probabilities,
        fit_rows,
        **gate_kwargs,
    )
    calibration_metrics = sequential_policy_metrics(
        success,
        fall,
        rollout_ids,
        confirmation_cycles,
        probabilities,
        calibration_rows,
        **gate_kwargs,
    )
    validation_metrics = sequential_policy_metrics(
        success,
        fall,
        rollout_ids,
        confirmation_cycles,
        probabilities,
        validation_rows,
        **gate_kwargs,
    )
    export_switch_selector_onnx(result, onnx_path)
    parity = verify_switch_selector_onnx(result, onnx_path, model_observations)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_shape = session.get_outputs()[0].shape
    if output_shape != [1, len(prototype_indices)]:
        raise RuntimeError("exported selector has an unexpected output shape")

    offline_gate_passed = bool(
        validation_metrics["success_rate"] >= 0.90
        and validation_metrics["falls"] == 0
        and parity["maximum_absolute_error"] <= 5.0e-6
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        probabilities=probabilities.astype(np.float32),
        prototype_indices=np.asarray(prototype_indices, dtype=np.int32),
        prototype_rollout_id=prototype_rollout_ids[np.asarray(prototype_indices)],
        fit_rows=fit_rows.astype(np.uint8),
        calibration_rows=calibration_rows.astype(np.uint8),
        validation_rows=validation_rows.astype(np.uint8),
        threshold=np.asarray([selected_gate["threshold"]], dtype=np.float32),
        consecutive_frames=np.asarray(
            [selected_gate["consecutive_frames"]], dtype=np.int32
        ),
    )
    report = {
        "schema_version": 1,
        "purpose": "causal_exact_cpu_kick_switch_prototype_selector",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": (
            "online closed-loop and server replay gates remain required"
            if offline_gate_passed
            else "blind offline approach gate did not pass"
        ),
        "offline_release_gate_passed": offline_gate_passed,
        "switch_corpus": str(args.switch_corpus.resolve()),
        "switch_corpus_sha256": sha256_file(args.switch_corpus),
        "prototype_bank": str(args.prototype_bank.resolve()),
        "prototype_bank_sha256": sha256_file(args.prototype_bank),
        "selected_prototype_indices": list(prototype_indices),
        "selected_prototype_rollout_ids": prototype_rollout_ids[
            np.asarray(prototype_indices)
        ].tolist(),
        "seed": args.seed,
        "prototype_set": args.prototype_set,
        "feature_profile": args.feature_profile,
        "cycle_normalizer": args.cycle_normalizer,
        "model_observation_size": int(model_observations.shape[1]),
        "fit_rollout_ids": list(fit_ids),
        "calibration_rollout_ids": list(calibration_ids),
        "validation_rollout_ids": np.unique(
            rollout_ids[validation_rows]
        ).astype(int).tolist(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "fall_weight": args.fall_weight,
            "positive_weights": result.positive_weights.tolist(),
            "fit_binary_cross_entropy": result.fit_loss,
            "calibration_binary_cross_entropy": result.calibration_loss,
            "history": list(result.history),
        },
        "calibration": {
            "minimum_precision": args.minimum_calibration_precision,
            "precision_gate_available": precision_gate_available,
            "selected_threshold": selected_gate["threshold"],
            "selected_consecutive_frames": selected_gate["consecutive_frames"],
            "fallback_precision_gate_available": fallback_precision_gate_available,
            "selected_fallback": fallback,
            "evaluated_configurations": len(calibration_grid),
            "selected_metrics": _without_decisions(calibration_metrics),
        },
        "fit_metrics": _without_decisions(fit_metrics),
        "validation_metrics": _without_decisions(validation_metrics),
        "validation_decisions": validation_metrics["decisions"],
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_parity": parity,
        "npz": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "onnxruntime": ort.__version__,
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
