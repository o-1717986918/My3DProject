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
ARRAY_DTYPES = {
    "root_position": np.float32,
    "root_quaternion_xyzw": np.float32,
    "root_linear_velocity": np.float32,
    "root_angular_velocity": np.float32,
    "joint_position": np.float32,
    "joint_velocity": np.float32,
    "foot_contact": np.bool_,
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
            if value.dtype != ARRAY_DTYPES[name]:
                errors.append(
                    f"{name} dtype {value.dtype} != {np.dtype(ARRAY_DTYPES[name])}"
                )

        if frame_count < 25:
            errors.append(f"frame count {frame_count} is below 25")

        if "root_quaternion_xyzw" in arrays:
            norms = np.linalg.norm(arrays["root_quaternion_xyzw"], axis=1)
            if not np.allclose(norms, 1.0, atol=1.0e-3):
                errors.append("root quaternions are not normalized within 1e-3")

        longest_airborne = 0
        contact_counts = [0, 0]
        if "foot_contact" in arrays:
            contact = arrays["foot_contact"].astype(bool)
            contact_counts = contact.sum(axis=0).astype(int).tolist()
            longest_airborne = _max_consecutive(~contact.any(axis=1))
            if longest_airborne < 1:
                errors.append("reference contains no 50 Hz two-foot aerial frame")
            if longest_airborne > 15:
                errors.append(
                    f"longest aerial interval {longest_airborne} exceeds 15 frames"
                )
            minimum_contact_frames = max(2, int(np.ceil(0.05 * frame_count)))
            for side, count in zip(("left", "right"), contact_counts):
                if count < minimum_contact_frames:
                    errors.append(
                        f"{side} contact count {count} is below "
                        f"{minimum_contact_frames}"
                    )

        average_horizontal_speed = 0.0
        if "root_position" in arrays and frame_count >= 2:
            duration = (frame_count - 1) / 50.0
            displacement = (
                arrays["root_position"][-1, :2] - arrays["root_position"][0, :2]
            )
            average_horizontal_speed = float(np.linalg.norm(displacement) / duration)
            if not 1.0 <= average_horizontal_speed <= 4.5:
                errors.append(
                    "average horizontal speed "
                    f"{average_horizontal_speed:.3f} m/s is outside [1.0, 4.5]"
                )
            if float(np.min(arrays["root_position"][:, 2])) < 0.30:
                errors.append("root height falls below 0.30 m")

        if "joint_position" in arrays:
            lower_body_excursion = float(
                np.max(np.ptp(arrays["joint_position"][:, 11:], axis=0))
            )
            if lower_body_excursion < 0.20:
                errors.append(
                    f"lower-body joint excursion {lower_body_excursion:.3f} rad "
                    "is below 0.20"
                )

        if "root_linear_velocity" in arrays:
            maximum_vertical_speed = float(
                np.max(np.abs(arrays["root_linear_velocity"][:, 2]))
            )
            if maximum_vertical_speed > 4.0:
                errors.append(
                    f"root vertical speed {maximum_vertical_speed:.3f} m/s "
                    "exceeds 4.0"
                )

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
            if metadata.get("output_frequency_hz") not in (None, 50, 50.0):
                errors.append("metadata output_frequency_hz is not 50")
            replay = metadata.get("rcss_replay", {})
            if replay:
                if replay.get("non_foot_pitch_contact_frames", 0) != 0:
                    errors.append("RCSS replay contains non-foot pitch contact")
                if replay.get("minimum_contact_distance_m", 0.0) < -0.015:
                    errors.append("RCSS replay penetration exceeds 15 mm")

    actual_sha = sha256(path)
    return {
        "schema_version": 1,
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "frame_count": frame_count,
        "frequency_hz": 50,
        "duration_seconds": max(0, frame_count - 1) / 50.0,
        "average_horizontal_speed_m_s": average_horizontal_speed,
        "foot_contact_frames": contact_counts,
        "longest_airborne_frames": longest_airborne,
        "longest_airborne_seconds": longest_airborne / 50.0,
        "provenance": metadata,
        "errors": errors,
        "passed": not errors,
    }
