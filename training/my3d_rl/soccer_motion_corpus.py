"""Validated padded corpus for finite multi-motion soccer training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .soccer_motion_dynamics import load_soccer_motion_arrays
from .soccer_motion_reference import validate_soccer_motion_reference


@dataclass(frozen=True)
class SoccerMotionCorpus:
    relative_paths: tuple[str, ...]
    sha256: tuple[str, ...]
    lengths: np.ndarray
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    root_linear_velocity: np.ndarray
    root_angular_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    foot_contact: np.ndarray
    kick_leg_one_hot: np.ndarray
    reset_weights: np.ndarray

    @property
    def motion_count(self) -> int:
        return len(self.relative_paths)

    @property
    def maximum_frames(self) -> int:
        return int(self.joint_position.shape[1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_soccer_motion_corpus(
    root: Path,
    *,
    failure_report: Path | None = None,
    validate: bool = True,
) -> SoccerMotionCorpus:
    """Load, provenance-check and last-frame-pad a local motion corpus."""
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no soccer motions found below {root}")
    arrays: list[dict[str, np.ndarray]] = []
    legs: list[str] = []
    hashes: list[str] = []
    relative_paths: list[str] = []
    for path in paths:
        if validate:
            result = validate_soccer_motion_reference(path)
            if not result["passed"]:
                raise ValueError(f"K0 validation failed for {path}: {result['errors']}")
        values = load_soccer_motion_arrays(path)
        with np.load(path, allow_pickle=False) as archive:
            leg = str(np.asarray(archive["kick_leg"]).item()).lower()
        if leg not in {"left", "right"}:
            raise ValueError(f"invalid kick leg {leg!r} in {path}")
        arrays.append(values)
        legs.append(leg)
        hashes.append(_sha256(path))
        relative_paths.append(str(path.relative_to(root)))

    lengths = np.asarray(
        [values["joint_position"].shape[0] for values in arrays], dtype=np.int32
    )
    maximum_frames = int(np.max(lengths))

    def pad(name: str, *, dtype: np.dtype | None = None) -> np.ndarray:
        values = []
        for motion in arrays:
            source = np.asarray(motion[name], dtype=dtype)
            padding = np.repeat(
                source[-1:], maximum_frames - source.shape[0], axis=0
            )
            values.append(np.concatenate([source, padding], axis=0))
        return np.stack(values)

    reset_weights = np.zeros((len(paths), maximum_frames), dtype=np.float64)
    report_records: dict[str, dict] = {}
    if failure_report is not None:
        payload = json.loads(failure_report.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError("failure report is incomplete")
        report_records = {
            record["relative_path"]: record for record in payload["records"]
        }
        if set(report_records) != set(relative_paths):
            raise ValueError("failure report paths differ from corpus")
    for index, (relative_path, digest, length) in enumerate(
        zip(relative_paths, hashes, lengths, strict=True)
    ):
        if report_records:
            record = report_records[relative_path]
            if record["sha256"] != digest:
                raise ValueError(f"failure report hash mismatch for {relative_path}")
            weights = np.asarray(
                record["failure_frame_sampling"]["weights"], dtype=np.float64
            )
            if weights.shape != (length,):
                raise ValueError(f"failure weights have wrong shape for {relative_path}")
        else:
            weights = np.ones(length, dtype=np.float64)
        weights = weights.copy()
        weights[-1] = 0.0
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError(f"invalid reset weights for {relative_path}")
        if np.sum(weights) <= 0.0:
            raise ValueError(f"reset weights have no support for {relative_path}")
        reset_weights[index, :length] = weights / np.sum(weights)

    kick_leg = np.asarray(
        [[1.0, 0.0] if leg == "left" else [0.0, 1.0] for leg in legs],
        dtype=np.float32,
    )
    quaternion_xyzw = pad("root_quaternion_xyzw", dtype=np.float32)
    return SoccerMotionCorpus(
        relative_paths=tuple(relative_paths),
        sha256=tuple(hashes),
        lengths=lengths,
        root_position=pad("root_position", dtype=np.float32),
        root_quaternion_wxyz=quaternion_xyzw[:, :, [3, 0, 1, 2]],
        root_linear_velocity=pad("root_linear_velocity", dtype=np.float32),
        root_angular_velocity=pad("root_angular_velocity", dtype=np.float32),
        joint_position=pad("joint_position", dtype=np.float32),
        joint_velocity=pad("joint_velocity", dtype=np.float32),
        foot_contact=pad("foot_contact", dtype=bool),
        kick_leg_one_hot=kick_leg,
        reset_weights=reset_weights.astype(np.float32),
    )
