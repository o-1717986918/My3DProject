#!/usr/bin/env python3
"""Train a match-grouped, release-time gate over fixed kick experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import onnxruntime as ort

from my3d_rl.kick_expert_gate import (
    apply_kick_expert_gate,
    expert_choice_metrics,
    export_kick_expert_gate_onnx,
    feature_indices,
    fit_kick_expert_gate,
    kick_forward_drive_success,
    kick_strict_success_margin,
    verify_kick_expert_gate_onnx,
)
from tools.evaluate_kick_switch_prototype_bank import greedy_utility_bank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _static_metrics(
    prototype_index: int,
    success: np.ndarray,
    fall: np.ndarray,
    score: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(rows, dtype=bool)
    return {
        "prototype_index": int(prototype_index),
        "entries": int(np.count_nonzero(selected)),
        "successes": int(np.count_nonzero(success[prototype_index, selected])),
        "falls": int(np.count_nonzero(fall[prototype_index, selected])),
        "mean_physical_score": float(np.mean(score[prototype_index, selected])),
    }


def _best_static_expert(
    success: np.ndarray,
    fall: np.ndarray,
    score: np.ndarray,
    rows: np.ndarray,
) -> int:
    """Choose a static expert on the supplied rows with deterministic ties."""
    selected = np.asarray(rows, dtype=bool)
    return max(
        range(success.shape[0]),
        key=lambda index: (
            int(np.count_nonzero(success[index, selected])),
            float(np.mean(score[index, selected])),
            -int(np.count_nonzero(fall[index, selected])),
            -index,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("prototype_bank", type=Path)
    parser.add_argument("--maximum-prototypes", type=int, default=30)
    parser.add_argument(
        "--feature-profile",
        choices=("physical_v1", "proprioception_v1"),
        default="physical_v1",
    )
    parser.add_argument(
        "--target-profile",
        choices=("physical-score", "strict-success-margin"),
        default="physical-score",
        help=(
            "physical-score optimizes general ball progress; strict-success-margin "
            "models the nearest distance/direction/speed contract boundary"
        ),
    )
    parser.add_argument("--latent-rank", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=30.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_prototypes < 2:
        raise ValueError("expert gate requires at least two prototypes")
    if not args.output_prefix.is_absolute() or args.output_prefix.is_relative_to(
        Path.cwd()
    ):
        raise ValueError("output prefix must be absolute and outside the repository")
    output_json = args.output_prefix.with_suffix(".json")
    output_npz = args.output_prefix.with_suffix(".npz")
    output_onnx = args.output_prefix.with_suffix(".onnx")
    if any(path.exists() for path in (output_json, output_npz, output_onnx)):
        raise FileExistsError("kick expert gate outputs already exist")

    corpus_manifest_path = args.transition_corpus.with_suffix(".json")
    bank_manifest_path = args.prototype_bank.with_suffix(".json")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("purpose")
        != "kick_policy_v3_walk_to_kick_transition_corpus"
        or corpus_manifest.get("npz_sha256") != _sha256(args.transition_corpus)
    ):
        raise ValueError("transition corpus is invalid or has a hash mismatch")
    if (
        bank_manifest.get("purpose")
        != "exact_cpu_kick_switch_prototype_bank_coverage"
        or bank_manifest.get("npz_sha256") != _sha256(args.prototype_bank)
        or bank_manifest.get("switch_corpus_sha256")
        != corpus_manifest["npz_sha256"]
    ):
        raise ValueError("prototype bank is invalid or does not match the corpus")

    with np.load(args.prototype_bank, allow_pickle=False) as archive:
        required = {
            "prototype_rollout_id",
            "approach_rollout_id",
            "split",
            "success",
            "fall",
            "score",
            "actor_observation",
            "contact",
            "maximum_progress_m",
            "lateral_error_m",
            "maximum_directional_speed_mps",
        }
        if args.target_profile == "strict-success-margin":
            required |= {
                "contact",
                "range_error_m",
                "lateral_error_m",
                "speed_error_mps",
            }
        if not required <= set(archive.files):
            raise ValueError("prototype bank lacks expert-gate arrays")
        prototype_rollout_ids = np.asarray(
            archive["prototype_rollout_id"], dtype=np.int32
        )
        rollout_ids = np.asarray(archive["approach_rollout_id"], dtype=np.int32)
        split = np.asarray(archive["split"], dtype=np.uint8)
        success = np.asarray(archive["success"], dtype=np.uint8)
        fall = np.asarray(archive["fall"], dtype=np.uint8)
        score = np.asarray(archive["score"], dtype=np.float32)
        observations = np.asarray(archive["actor_observation"], dtype=np.float32)
        contact = np.asarray(archive["contact"], dtype=bool)
        maximum_progress = np.asarray(
            archive["maximum_progress_m"], dtype=np.float32
        )
        lateral_error = np.asarray(archive["lateral_error_m"], dtype=np.float32)
        directional_speed = np.asarray(
            archive["maximum_directional_speed_mps"], dtype=np.float32
        )
        if args.target_profile == "strict-success-margin":
            range_error = np.asarray(archive["range_error_m"], dtype=np.float32)
            speed_error = np.asarray(archive["speed_error_mps"], dtype=np.float32)
    with np.load(args.transition_corpus, allow_pickle=False) as archive:
        required = {"rollout_id", "split", "source_match_id"}
        if not required <= set(archive.files):
            raise ValueError("transition corpus lacks match-group metadata")
        corpus_rollout_ids = np.asarray(archive["rollout_id"], dtype=np.int32)
        corpus_split = np.asarray(archive["split"], dtype=np.uint8)
        source_match_ids = np.asarray(archive["source_match_id"], dtype=np.int32)
    if (
        observations.ndim != 2
        or observations.shape[0] != rollout_ids.size
        or split.shape != rollout_ids.shape
        or source_match_ids.shape != rollout_ids.shape
        or not np.array_equal(rollout_ids, corpus_rollout_ids)
        or not np.array_equal(split, corpus_split)
        or success.shape != fall.shape
        or success.shape != score.shape
        or success.shape != (prototype_rollout_ids.size, rollout_ids.size)
    ):
        raise ValueError("kick expert gate arrays are misaligned")

    train_rows = split == 0
    development_rows = split == 1
    train_matches = np.unique(source_match_ids[train_rows])
    development_matches = np.unique(source_match_ids[development_rows])
    if (
        train_matches.size < 3
        or development_matches.size < 1
        or np.intersect1d(train_matches, development_matches).size
    ):
        raise ValueError("kick expert gate requires disjoint whole-match splits")

    selected = tuple(
        greedy_utility_bank(
            score,
            fall,
            rollout_ids,
            train_rows,
            maximum_prototypes=args.maximum_prototypes,
        )
    )
    target = score
    if args.target_profile == "strict-success-margin":
        target = kick_strict_success_margin(
            contact,
            fall,
            range_error,
            lateral_error,
            speed_error,
        )
    features = feature_indices(args.feature_profile)
    result = fit_kick_expert_gate(
        observations,
        target,
        train_rows,
        prototype_indices=selected,
        feature_indices=features,
        latent_rank=args.latent_rank,
        ridge=args.ridge,
    )
    predictions = apply_kick_expert_gate(result, observations)

    # Honest cross-match estimate: the expert bank and model are both rebuilt
    # without the held-out match.  Per-frame rollout IDs are never used as a
    # statistical partition.
    oof_choices = np.full(rollout_ids.shape, -1, dtype=np.int32)
    for match_id in train_matches:
        held_out = train_rows & (source_match_ids == match_id)
        fold_fit = train_rows & ~held_out
        fold_selected = tuple(
            greedy_utility_bank(
                score,
                fall,
                rollout_ids,
                fold_fit,
                maximum_prototypes=args.maximum_prototypes,
            )
        )
        fold_result = fit_kick_expert_gate(
            observations,
            target,
            fold_fit,
            prototype_indices=fold_selected,
            feature_indices=features,
            latent_rank=args.latent_rank,
            ridge=args.ridge,
        )
        fold_predictions = apply_kick_expert_gate(
            fold_result, observations[held_out]
        )
        oof_choices[held_out] = np.asarray(fold_selected, dtype=np.int32)[
            np.argmax(fold_predictions, axis=1)
        ]
    train_indices = np.flatnonzero(train_rows)
    if np.any(oof_choices[train_rows] < 0):
        raise RuntimeError("whole-match cross-validation left unscored training rows")
    oof_success = success[oof_choices[train_rows], train_indices]
    oof_fall = fall[oof_choices[train_rows], train_indices]
    oof_score = score[oof_choices[train_rows], train_indices]
    cross_match_metrics = {
        "entries": int(train_indices.size),
        "successes": int(oof_success.sum()),
        "success_rate": float(oof_success.mean()),
        "falls": int(oof_fall.sum()),
        "fall_rate": float(oof_fall.mean()),
        "mean_physical_score": float(oof_score.mean()),
        "used_experts": int(np.unique(oof_choices[train_rows]).size),
    }

    train_static = _best_static_expert(success, fall, score, train_rows)
    # This development-only oracle is reported as a comparator, never exported
    # as a deployable choice.  It makes selection bias explicit.
    development_best_static = _best_static_expert(
        success, fall, score, development_rows
    )
    train_metrics = expert_choice_metrics(
        result, predictions, success, fall, score, train_rows
    )
    development_metrics = expert_choice_metrics(
        result, predictions, success, fall, score, development_rows
    )
    forward_drive = kick_forward_drive_success(
        contact, maximum_progress, lateral_error, directional_speed
    )
    development_indices = np.flatnonzero(development_rows)
    development_choices = np.asarray(result.prototype_indices, dtype=np.int32)[
        np.argmax(predictions[development_rows], axis=1)
    ]
    development_forward_success = forward_drive[
        development_choices, development_indices
    ]
    train_static_development = _static_metrics(
        train_static, success, fall, score, development_rows
    )
    best_static_development = _static_metrics(
        development_best_static, success, fall, score, development_rows
    )
    strict_candidate_advanced = bool(
        development_metrics["successes"]
        > best_static_development["successes"]
        and development_metrics["fall_rate"]
        <= best_static_development["falls"]
        / max(1, best_static_development["entries"])
        + 0.10
    )
    development_best_forward_static = max(
        range(forward_drive.shape[0]),
        key=lambda index: (
            int(np.count_nonzero(forward_drive[index, development_rows])),
            float(np.mean(score[index, development_rows])),
            -int(np.count_nonzero(fall[index, development_rows])),
            -index,
        ),
    )
    best_forward_static_successes = int(
        np.count_nonzero(
            forward_drive[development_best_forward_static, development_rows]
        )
    )
    forward_drive_candidate_advanced = bool(
        np.count_nonzero(development_forward_success)
        > best_forward_static_successes
        and np.mean(fall[development_choices, development_indices])
        <= np.mean(fall[development_best_forward_static, development_rows]) + 0.10
    )
    candidate_advanced = strict_candidate_advanced or forward_drive_candidate_advanced

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    export_kick_expert_gate_onnx(
        result, output_onnx, observation_size=observations.shape[1]
    )
    parity = verify_kick_expert_gate_onnx(result, output_onnx, observations)
    session = ort.InferenceSession(
        str(output_onnx), providers=["CPUExecutionProvider"]
    )
    if session.get_outputs()[0].shape != [1, len(selected)]:
        raise RuntimeError("kick expert gate ONNX output shape is invalid")
    np.savez_compressed(
        output_npz,
        feature_indices=result.feature_indices,
        prototype_indices=np.asarray(result.prototype_indices, dtype=np.int32),
        prototype_rollout_id=prototype_rollout_ids[
            np.asarray(result.prototype_indices)
        ],
        observation_mean=result.observation_mean,
        observation_std=result.observation_std,
        target_mean=result.target_mean,
        latent_basis=result.latent_basis,
        latent_weights=result.latent_weights,
        predicted_target=predictions.astype(np.float32),
        oof_prototype_indices=oof_choices,
        train_rows=train_rows.astype(np.uint8),
        development_rows=development_rows.astype(np.uint8),
    )
    report = {
        "schema_version": 1,
        "purpose": "match_grouped_release_time_kick_expert_gate",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": (
            "candidate must pass a newly collected post-freeze whole-match corpus and RCSS shadow replay"
            if candidate_advanced
            else "development corpus did not beat the strongest static expert"
        ),
        "candidate_advanced_to_fresh_blind_check": candidate_advanced,
        "strict_pass_candidate_advanced": strict_candidate_advanced,
        "forward_drive_candidate_advanced": forward_drive_candidate_advanced,
        "decision_semantics": "choose_one_expert_at_release_and_hold_for_complete_motion",
        "fall_policy": "finite target cost; occasional falls are allowed and reported",
        "transition_corpus": str(args.transition_corpus.resolve()),
        "transition_corpus_sha256": _sha256(args.transition_corpus),
        "prototype_bank": str(args.prototype_bank.resolve()),
        "prototype_bank_sha256": _sha256(args.prototype_bank),
        "feature_profile": args.feature_profile,
        "target_profile": args.target_profile,
        "target_contract": (
            "minimum normalized margin to range<=0.5m, lateral<=0.5m, speed_error<=1.0m/s; "
            "no-contact=-2, fall subtracts finite 4"
            if args.target_profile == "strict-success-margin"
            else "kick teacher continuous physical score with finite fall cost"
        ),
        "feature_count": int(features.size),
        "latent_rank": args.latent_rank,
        "ridge": args.ridge,
        "maximum_prototypes": args.maximum_prototypes,
        "prototype_indices": list(result.prototype_indices),
        "prototype_rollout_ids": prototype_rollout_ids[
            np.asarray(result.prototype_indices)
        ].astype(int).tolist(),
        "train_source_match_ids": train_matches.astype(int).tolist(),
        "development_source_match_ids": development_matches.astype(int).tolist(),
        "cross_match_training_metrics": cross_match_metrics,
        "fit_metrics": {k: v for k, v in train_metrics.items() if k != "prototype_indices"},
        "development_metrics": {
            k: v for k, v in development_metrics.items() if k != "prototype_indices"
        },
        "development_forward_drive_metrics": {
            "contract": (
                "contact, maximum_progress>=2.5m, lateral<=1.0m, "
                "directional_speed>=1.5m/s; falls reported separately"
            ),
            "entries": int(development_indices.size),
            "successes": int(np.count_nonzero(development_forward_success)),
            "success_rate": float(np.mean(development_forward_success)),
            "falls": int(
                np.count_nonzero(fall[development_choices, development_indices])
            ),
        },
        "development_oracle_static_forward_drive_metrics": {
            "prototype_index": development_best_forward_static,
            "prototype_rollout_id": int(
                prototype_rollout_ids[development_best_forward_static]
            ),
            "entries": int(development_indices.size),
            "successes": best_forward_static_successes,
            "falls": int(
                np.count_nonzero(
                    fall[development_best_forward_static, development_rows]
                )
            ),
            "warning": "selected with development outcomes; comparator only",
        },
        "training_selected_static_development_metrics": {
            **train_static_development,
            "prototype_rollout_id": int(prototype_rollout_ids[train_static]),
        },
        "development_oracle_static_metrics": {
            **best_static_development,
            "prototype_rollout_id": int(
                prototype_rollout_ids[development_best_static]
            ),
            "warning": "selected with development outcomes; comparator only",
        },
        "onnx": str(output_onnx.resolve()),
        "onnx_sha256": _sha256(output_onnx),
        "onnx_parity": parity,
        "npz": str(output_npz.resolve()),
        "npz_sha256": _sha256(output_npz),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
    }
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
