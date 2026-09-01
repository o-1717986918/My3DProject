"""Validated K1 soccer-motion teacher data and balanced BC utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SOCCER_MOTION_DAGGER_PURPOSES = {
    "k1_b_exact_cpu_soccer_motion_dagger_collection",
    "k1_d_cross_fitted_soccer_motion_dagger_collection",
}


def validate_bc_data_manifest(
    dataset: Path,
    *,
    base_checkpoint: Path,
    selection_manifest: Path | None = None,
    dagger_manifest: Path | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    """Bind BC input to either the selected teacher or a DAgger aggregate."""
    if (selection_manifest is None) == (dagger_manifest is None):
        raise ValueError("select exactly one BC data manifest")
    manifest_path = selection_manifest or dagger_manifest
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if selection_manifest is not None:
        if (
            manifest.get("status") != "complete_teacher_corpus_selected"
            or manifest.get("teacher_gates_passed") != manifest.get("motion_count")
            or manifest.get("combined_dataset", {}).get("sha256")
            != dataset_sha256
        ):
            raise ValueError("teacher selection manifest is incomplete or mismatched")
        return "selected_teacher_corpus", manifest_path, manifest
    if (
        manifest.get("status") != "complete"
        or manifest.get("purpose") not in SOCCER_MOTION_DAGGER_PURPOSES
        or manifest.get("output_dataset_sha256") != dataset_sha256
        or Path(manifest.get("student_checkpoint", "")).resolve()
        != base_checkpoint.resolve()
    ):
        raise ValueError("DAgger manifest is incomplete or mismatched")
    return "dagger_aggregate", manifest_path, manifest


def dagger_row_mask(
    motion: np.ndarray,
    start_frame: np.ndarray,
    manifest: dict[str, Any],
) -> np.ndarray:
    """Identify the appended on-policy rows in a DAgger aggregate."""
    motion = np.asarray(motion)
    start_frame = np.asarray(start_frame)
    if motion.ndim != 1 or motion.shape != start_frame.shape:
        raise ValueError("DAgger row keys must be equal one-dimensional arrays")
    purpose = manifest.get("purpose")
    if purpose not in SOCCER_MOTION_DAGGER_PURPOSES:
        raise ValueError("manifest is not a soccer-motion DAgger collection")
    if purpose == "k1_d_cross_fitted_soccer_motion_dagger_collection":
        source_frames = int(manifest.get("source_frames", -1))
        dagger_frames = int(manifest.get("dagger_frames", -1))
        if min(source_frames, dagger_frames) < 0 or (
            source_frames + dagger_frames != motion.shape[0]
        ):
            raise ValueError("cross-fitted DAgger frame counts are invalid")
        result = np.zeros(motion.shape, dtype=bool)
        result[source_frames:] = True
        return result
    result = np.zeros(motion.shape, dtype=bool)
    nodes = manifest.get("per_motion")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("DAgger manifest lacks per-motion start frames")
    seen: set[int] = set()
    for node in nodes:
        value = int(node["motion"])
        if value in seen:
            raise ValueError("DAgger manifest repeats a motion")
        seen.add(value)
        starts = np.asarray(node["start_frames"], dtype=start_frame.dtype)
        result |= (motion == value) & np.isin(start_frame, starts)
    if int(np.sum(result)) != int(manifest.get("dagger_frames", -1)):
        raise ValueError("DAgger manifest frame count differs from the aggregate")
    return result


def load_soccer_motion_teacher_dataset(
    path: Path, *, observation_size: int, action_size: int
) -> dict[str, np.ndarray]:
    required = {
        "observation",
        "base_action",
        "teacher_action",
        "split",
        "motion",
        "start_frame",
        "reference_frame",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"soccer-motion teacher data is missing {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    count = result["observation"].shape[0]
    if result["observation"].shape != (count, observation_size):
        raise ValueError("soccer-motion teacher observation shape is invalid")
    for name in ("base_action", "teacher_action"):
        if result[name].shape != (count, action_size):
            raise ValueError(f"soccer-motion teacher {name} shape is invalid")
    for name in ("split", "motion", "start_frame", "reference_frame"):
        if result[name].shape != (count,):
            raise ValueError(f"soccer-motion teacher {name} shape is invalid")
    if count < 2 or not np.isfinite(result["observation"]).all():
        raise ValueError("soccer-motion teacher observations are empty or non-finite")
    if not np.isfinite(result["base_action"]).all() or not np.isfinite(
        result["teacher_action"]
    ).all():
        raise ValueError("soccer-motion teacher actions are non-finite")
    if np.max(np.abs(result["teacher_action"])) > 1.0 + 1.0e-6:
        raise ValueError("soccer-motion teacher actions exceed the policy contract")
    if not set(result["split"].tolist()) <= {0, 1}:
        raise ValueError("soccer-motion teacher split must be binary")
    if not np.any(result["split"] == 0) or not np.any(result["split"] == 1):
        raise ValueError("soccer-motion teacher needs train and validation samples")
    motions = sorted(set(int(value) for value in result["motion"]))
    if motions != list(range(len(motions))):
        raise ValueError("soccer-motion teacher motion IDs must be contiguous from zero")
    for motion in motions:
        values = result["split"][result["motion"] == motion]
        if not np.any(values == 0) or not np.any(values == 1):
            raise ValueError(f"motion {motion} lacks a train or validation split")
    return {
        "observation": result["observation"].astype(np.float32),
        "base_action": result["base_action"].astype(np.float32),
        "teacher_action": result["teacher_action"].astype(np.float32),
        "split": result["split"].astype(np.int8),
        "motion": result["motion"].astype(np.int16),
        "start_frame": result["start_frame"].astype(np.int32),
        "reference_frame": result["reference_frame"].astype(np.int32),
    }


def motion_balanced_indices(
    rng: np.random.Generator,
    motion: np.ndarray,
    split: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1 or motion.shape != split.shape or motion.ndim != 1:
        raise ValueError("balanced sampler inputs are invalid")
    train_motions = sorted(set(int(value) for value in motion[split == 0]))
    if not train_motions:
        raise ValueError("balanced sampler has no training motions")
    groups = {value: np.flatnonzero((motion == value) & (split == 0)) for value in train_motions}
    selected_motion = rng.choice(train_motions, size=batch_size, replace=True)
    result = np.empty(batch_size, dtype=np.int64)
    for value, indices in groups.items():
        mask = selected_motion == value
        result[mask] = rng.choice(indices, size=int(np.sum(mask)), replace=True)
    return result


def action_error_metrics(
    prediction: np.ndarray,
    teacher: np.ndarray,
    base: np.ndarray,
    motion: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if prediction.shape != teacher.shape or prediction.shape != base.shape:
        raise ValueError("action error arrays must have identical shapes")
    error = prediction - teacher
    per_motion: list[dict[str, float | int]] = []
    for value in sorted(set(int(node) for node in motion)):
        mask = motion == value
        per_motion.append(
            {
                "motion": value,
                "samples": int(np.sum(mask)),
                "teacher_mse": float(np.mean(np.square(error[mask]))),
                "teacher_max_abs_error": float(np.max(np.abs(error[mask]))),
                "base_mse": float(np.mean(np.square(prediction[mask] - base[mask]))),
            }
        )
    return {
        "samples": int(prediction.shape[0]),
        "teacher_mse": float(np.mean(np.square(error))),
        "teacher_mae": float(np.mean(np.abs(error))),
        "teacher_max_abs_error": float(np.max(np.abs(error))),
        "base_mse": float(np.mean(np.square(prediction - base))),
        "maximum_abs_action": float(np.max(np.abs(prediction))),
        "per_motion": per_motion,
    }
