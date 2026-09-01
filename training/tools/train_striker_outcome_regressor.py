#!/usr/bin/env python3
"""Train a continuous exact-CPU outcome model for striker action selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_switch_selector import grouped_fit_calibration_split
from my3d_rl.striker_outcome_regressor import (
    apply_outcome_regressor_numpy,
    export_outcome_regressor_onnx,
    train_outcome_regressor,
    verify_outcome_regressor_onnx,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_observations(
    archive: np.lib.npyio.NpzFile, profile: str
) -> np.ndarray:
    if profile in {"history_summary", "history_10frame"}:
        history = np.asarray(archive["actor_history"], dtype=np.float32)
        selector_state = np.asarray(archive["selector_observation"], dtype=np.float32)
        if history.shape != (selector_state.shape[0], 50, 102):
            raise ValueError("selector history has an invalid shape")
        if profile == "history_summary":
            return np.concatenate(
                [
                    selector_state,
                    history.mean(axis=1),
                    history.std(axis=1),
                    history[:, -1] - history[:, 0],
                ],
                axis=1,
            ).astype(np.float32)
        return np.concatenate(
            [
                history[:, np.linspace(0, 49, 10, dtype=np.int32)].reshape(
                    history.shape[0], -1
                ),
                selector_state[:, 102:125],
            ],
            axis=1,
        ).astype(np.float32)
    observation_key = {
        "actor": "actor_observation",
        "selector": "selector_observation",
        "privileged": "privileged_observation",
    }[profile]
    return np.asarray(archive[observation_key], dtype=np.float32)


def _selection_metrics(
    success: np.ndarray,
    fall: np.ndarray,
    predicted_log_distance: np.ndarray,
    rows: np.ndarray,
    *,
    fallback_local_index: int,
    margin_threshold: float,
) -> dict[str, float | int]:
    selected_rows = np.flatnonzero(rows)
    scores = predicted_log_distance[selected_rows]
    order = np.argsort(scores, axis=1, kind="stable")
    learned = order[:, 0]
    margins = scores[np.arange(scores.shape[0]), order[:, 1]] - scores[
        np.arange(scores.shape[0]), order[:, 0]
    ]
    learned_mask = margins >= margin_threshold
    chosen = np.where(learned_mask, learned, fallback_local_index)
    succeeded = success[chosen, selected_rows]
    fallen = fall[chosen, selected_rows]
    return {
        "rollouts": int(selected_rows.size),
        "successes": int(succeeded.sum()),
        "falls": int(fallen.sum()),
        "success_rate": float(succeeded.mean()),
        "learned_decisions": int(learned_mask.sum()),
        "fallback_decisions": int((~learned_mask).sum()),
        "mean_predicted_margin": float(margins.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--seed", type=int, default=14_401)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument(
        "--observation-profile",
        choices=(
            "actor",
            "selector",
            "privileged",
            "history_summary",
            "history_10frame",
        ),
        default="selector",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0.0:
        raise ValueError("regressor optimization settings are invalid")
    if not args.output_prefix.is_absolute() or args.output_prefix.is_relative_to(
        Path.cwd()
    ):
        raise ValueError("output prefix must be absolute and outside the repository")
    manifest_path = args.corpus.with_suffix(".json")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        source.get("purpose") != "exact_cpu_striker_trigger_action_bank_corpus"
        or source.get("schema_version", 0) < 2
        or source.get("npz_sha256") != _sha256(args.corpus)
    ):
        raise ValueError("continuous striker action-bank corpus is invalid")
    with np.load(args.corpus, allow_pickle=False) as archive:
        observations = _load_observations(archive, args.observation_profile)
        success = np.asarray(archive["success"], dtype=np.uint8)
        fall = np.asarray(archive["fall"], dtype=np.uint8)
        final_goal_distance = np.asarray(
            archive["final_goal_distance_m"], dtype=np.float32
        )
        rollout_ids = np.asarray(archive["rollout_id"], dtype=np.int64)
        action_indices = np.asarray(archive["action_prior_index"], dtype=np.int32)
        split = np.asarray(archive["split"], dtype=np.uint8)
    if (
        observations.shape[0] != rollout_ids.size
        or observations.shape[1] not in (102, 125, 138, 431, 1043)
        or success.shape != fall.shape
        or success.shape != final_goal_distance.shape
        or success.shape != (action_indices.size, rollout_ids.size)
        or split.shape != rollout_ids.shape
    ):
        raise ValueError("continuous striker corpus arrays are misaligned")

    train_rows = split == 0
    validation_rows = split == 1
    fit_ids, calibration_ids = grouped_fit_calibration_split(
        rollout_ids,
        train_rows,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    result = train_outcome_regressor(
        observations,
        final_goal_distance,
        fall,
        rollout_ids,
        action_prior_indices=tuple(int(value) for value in action_indices),
        fit_rollout_ids=fit_ids,
        calibration_rollout_ids=calibration_ids,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    predicted_log_distance = apply_outcome_regressor_numpy(result, observations)
    predicted_distance = np.maximum(np.expm1(predicted_log_distance), 0.0)
    fit_rows = np.isin(rollout_ids, fit_ids)
    calibration_rows = np.isin(rollout_ids, calibration_ids)
    fallback_local_index = int(np.argmax(success[:, fit_rows].sum(axis=1)))
    calibration_scores = predicted_log_distance[calibration_rows]
    sorted_scores = np.sort(calibration_scores, axis=1)
    calibration_margins = sorted_scores[:, 1] - sorted_scores[:, 0]
    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0], dtype=np.float32),
                np.quantile(calibration_margins, np.linspace(0.05, 0.95, 19)),
                np.array([np.inf], dtype=np.float32),
            ]
        )
    )
    calibration_grid = [
        {
            "margin_threshold": None if np.isinf(threshold) else float(threshold),
            **_selection_metrics(
                success,
                fall,
                predicted_log_distance,
                calibration_rows,
                fallback_local_index=fallback_local_index,
                margin_threshold=float(threshold),
            ),
        }
        for threshold in thresholds
    ]
    selected_gate_index = max(
        range(len(calibration_grid)),
        key=lambda index: (
            -int(calibration_grid[index]["falls"]),
            int(calibration_grid[index]["successes"]),
            int(calibration_grid[index]["learned_decisions"]),
        ),
    )
    threshold = float(thresholds[selected_gate_index])
    fit_metrics = _selection_metrics(
        success,
        fall,
        predicted_log_distance,
        fit_rows,
        fallback_local_index=fallback_local_index,
        margin_threshold=threshold,
    )
    calibration_metrics = _selection_metrics(
        success,
        fall,
        predicted_log_distance,
        calibration_rows,
        fallback_local_index=fallback_local_index,
        margin_threshold=threshold,
    )
    validation_metrics = _selection_metrics(
        success,
        fall,
        predicted_log_distance,
        validation_rows,
        fallback_local_index=fallback_local_index,
        margin_threshold=threshold,
    )
    validation_target = final_goal_distance[:, validation_rows].T
    validation_prediction = predicted_distance[validation_rows]

    output_json = args.output_prefix.with_suffix(".json")
    output_npz = args.output_prefix.with_suffix(".npz")
    output_onnx = args.output_prefix.with_suffix(".onnx")
    if any(path.exists() for path in (output_json, output_npz, output_onnx)):
        raise FileExistsError("regressor outputs already exist")
    export_outcome_regressor_onnx(result, output_onnx)
    parity = verify_outcome_regressor_onnx(result, output_onnx, observations)
    np.savez_compressed(
        output_npz,
        predicted_log1p_goal_distance=predicted_log_distance.astype(np.float32),
        predicted_goal_distance_m=predicted_distance.astype(np.float32),
        action_prior_index=action_indices,
        fit_rows=fit_rows.astype(np.uint8),
        calibration_rows=calibration_rows.astype(np.uint8),
        validation_rows=validation_rows.astype(np.uint8),
        fallback_local_index=np.asarray([fallback_local_index], dtype=np.int32),
        margin_threshold=np.asarray([threshold], dtype=np.float32),
    )
    gate_passed = bool(
        validation_metrics["success_rate"] >= 0.90
        and validation_metrics["falls"] == 0
        and parity["maximum_absolute_error"] <= 5.0e-6
    )
    report = {
        "schema_version": 1,
        "purpose": "exact_cpu_striker_action_outcome_regressor",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": (
            "requires a second frozen corpus and online server replay"
            if gate_passed
            else "untouched exact CPU outcome-regression gate did not pass"
        ),
        "offline_gate_passed": gate_passed,
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": _sha256(args.corpus),
        "seed": args.seed,
        "observation_profile": args.observation_profile,
        "observation_size": int(observations.shape[1]),
        "action_prior_indices": action_indices.tolist(),
        "fit_rollout_ids": list(fit_ids),
        "calibration_rollout_ids": list(calibration_ids),
        "validation_rollout_ids": rollout_ids[validation_rows].astype(int).tolist(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "fit_huber_loss": result.fit_loss,
            "calibration_huber_loss": result.calibration_loss,
        },
        "selection": {
            "fallback_local_index": fallback_local_index,
            "fallback_prior_index": int(action_indices[fallback_local_index]),
            "margin_threshold": None if np.isinf(threshold) else threshold,
            "calibration_grid": calibration_grid,
        },
        "fit_metrics": fit_metrics,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "validation_outcome_error": {
            "mean_absolute_error_m": float(
                np.mean(np.abs(validation_prediction - validation_target))
            ),
            "root_mean_squared_error_m": float(
                np.sqrt(np.mean(np.square(validation_prediction - validation_target)))
            ),
        },
        "onnx": str(output_onnx.resolve()),
        "onnx_sha256": _sha256(output_onnx),
        "onnx_parity": parity,
        "npz": str(output_npz.resolve()),
        "npz_sha256": _sha256(output_npz),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
