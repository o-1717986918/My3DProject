"""Validation for retargeted T1 motion-prior inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_SHAPES = {
    "root_position": 3,
    "root_quaternion_xyzw": 4,
    "root_linear_velocity": 3,
    "root_angular_velocity": 3,
    "joint_position": 23,
    "joint_velocity": 23,
    "foot_contact": 2,
}
PROVENANCE_FIELDS = (
    "source_url",
    "source_version",
    "source_license",
    "conversion_command",
    "source_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_consecutive(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def validate_motion_reference(path: Path) -> dict[str, Any]:
    """Validate shapes, numerics, provenance and the minimum running morphology."""
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(ARRAY_SHAPES) - set(archive.files))
        if missing:
            errors.append(f"missing arrays: {missing}")
            frame_count = 0
            arrays: dict[str, np.ndarray] = {}
        else:
            arrays = {name: archive[name] for name in ARRAY_SHAPES}
            frame_count = int(arrays["joint_position"].shape[0])

        for name, width in ARRAY_SHAPES.items():
            if name not in arrays:
                continue
            value = arrays[name]
            if value.ndim != 2 or value.shape != (frame_count, width):
                errors.append(f"{name} shape {value.shape} != ({frame_count}, {width})")
            if name != "foot_contact" and not np.isfinite(value).all():
                errors.append(f"{name} contains non-finite values")

        if frame_count < 25:
            errors.append(f"frame count {frame_count} is below 25")

        if "root_quaternion_xyzw" in arrays:
            norms = np.linalg.norm(arrays["root_quaternion_xyzw"], axis=1)
            if not np.allclose(norms, 1.0, atol=1.0e-3):
                errors.append("root quaternions are not normalized within 1e-3")

        longest_airborne = 0
        if "foot_contact" in arrays:
            contact = arrays["foot_contact"].astype(bool)
            longest_airborne = _max_consecutive(~contact.any(axis=1))
            if longest_airborne < 1:
                errors.append("reference contains no 50 Hz two-foot aerial frame")

        metadata: dict[str, Any] = {}
        if "metadata_json" not in archive.files:
            errors.append("missing metadata_json")
        else:
            try:
                metadata = json.loads(str(archive["metadata_json"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid metadata_json: {exc}")
            for field in PROVENANCE_FIELDS:
                if not metadata.get(field):
                    errors.append(f"missing provenance field {field}")

    actual_sha = sha256(path)
    return {
        "schema_version": 1,
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "frame_count": frame_count,
        "frequency_hz": 50,
        "duration_seconds": frame_count / 50.0,
        "longest_airborne_frames": longest_airborne,
        "longest_airborne_seconds": longest_airborne / 50.0,
        "provenance": metadata,
        "errors": errors,
        "passed": not errors,
    }
