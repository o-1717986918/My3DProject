"""Short-horizon state comparison helpers for CPU MuJoCo and MJX backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParityThresholds:
    """Tolerances for locating meaningful short-horizon solver divergence."""

    target_max_abs_rad: float = 1.0e-6
    joint_position_max_abs_rad: float = 0.05
    root_position_norm_m: float = 0.03
    root_yaw_abs_rad: float = 0.05
    torso_orientation_rad: float = 0.08
    foot_height_max_abs_m: float = 0.025
    contact_proxy_mismatch_frames: int = 2

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"parity threshold {name} must be non-negative")


ERROR_TO_THRESHOLD = {
    "target_max_abs_rad": "target_max_abs_rad",
    "joint_position_max_abs_rad": "joint_position_max_abs_rad",
    "root_position_norm_m": "root_position_norm_m",
    "root_yaw_abs_rad": "root_yaw_abs_rad",
    "torso_orientation_rad": "torso_orientation_rad",
    "foot_height_max_abs_m": "foot_height_max_abs_m",
}


def quaternion_angle_error(a_wxyz: Sequence[float], b_wxyz: Sequence[float]) -> float:
    """Return the sign-invariant shortest angle between two quaternions."""
    a = np.asarray(a_wxyz, dtype=np.float64)
    b = np.asarray(b_wxyz, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("quaternions must each have shape (4,)")
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1.0e-12 or b_norm < 1.0e-12:
        return float("inf")
    dot = np.clip(abs(float(np.dot(a / a_norm, b / b_norm))), 0.0, 1.0)
    return float(2.0 * np.arccos(dot))


def wrapped_angle_error(a: float, b: float) -> float:
    """Return an absolute angle difference in [0, pi]."""
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def generate_action_sequence(
    *,
    pattern: str,
    steps: int,
    action_size: int,
    amplitude: float,
    seed: int,
) -> np.ndarray:
    """Create a deterministic action trace for parity replay."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if action_size < 1:
        raise ValueError("action_size must be positive")
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be in [0, 1]")
    if pattern == "zero":
        return np.zeros((steps, action_size), dtype=np.float32)
    if pattern == "sine":
        step_phase = np.arange(steps, dtype=np.float64)[:, None] * 0.37
        joint_phase = np.linspace(0.0, 2.0 * np.pi, action_size, endpoint=False)[None]
        return (amplitude * np.sin(step_phase + joint_phase)).astype(np.float32)
    if pattern == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-amplitude, amplitude, size=(steps, action_size)).astype(
            np.float32
        )
    raise ValueError(f"unsupported action pattern: {pattern}")


def step_errors(
    cpu: Mapping[str, Any], accelerated: Mapping[str, Any]
) -> dict[str, float | int]:
    """Compute backend errors from two normalized state snapshots."""
    cpu_target = np.asarray(cpu["joint_target_rad"], dtype=np.float64)
    accelerated_target = np.asarray(accelerated["joint_target_rad"], dtype=np.float64)
    cpu_joint = np.asarray(cpu["joint_position_rad"], dtype=np.float64)
    accelerated_joint = np.asarray(accelerated["joint_position_rad"], dtype=np.float64)
    cpu_root = np.asarray(cpu["root_position_m"], dtype=np.float64)
    accelerated_root = np.asarray(accelerated["root_position_m"], dtype=np.float64)
    cpu_feet = np.asarray(cpu["foot_lowest_height_m"], dtype=np.float64)
    accelerated_feet = np.asarray(accelerated["foot_lowest_height_m"], dtype=np.float64)
    cpu_contact = np.asarray(cpu["contact_proxy"], dtype=bool)
    accelerated_contact = np.asarray(accelerated["contact_proxy"], dtype=bool)

    return {
        "target_max_abs_rad": float(np.max(np.abs(cpu_target - accelerated_target))),
        "joint_position_max_abs_rad": float(
            np.max(np.abs(cpu_joint - accelerated_joint))
        ),
        "root_position_norm_m": float(np.linalg.norm(cpu_root - accelerated_root)),
        "root_yaw_abs_rad": wrapped_angle_error(
            float(cpu["root_yaw_rad"]), float(accelerated["root_yaw_rad"])
        ),
        "torso_orientation_rad": quaternion_angle_error(
            cpu["torso_quaternion_wxyz"], accelerated["torso_quaternion_wxyz"]
        ),
        "foot_height_max_abs_m": float(np.max(np.abs(cpu_feet - accelerated_feet))),
        "contact_proxy_mismatch_count": int(
            np.count_nonzero(cpu_contact != accelerated_contact)
        ),
    }


def summarize_trace(
    trace: Sequence[Mapping[str, Any]], thresholds: ParityThresholds
) -> dict[str, Any]:
    """Summarize maxima and the first step exceeding every parity threshold."""
    thresholds.validate()
    if not trace:
        raise ValueError("parity trace cannot be empty")

    first_divergence: dict[str, int | None] = {
        metric: None for metric in ERROR_TO_THRESHOLD
    }
    maxima = {metric: 0.0 for metric in ERROR_TO_THRESHOLD}
    contact_mismatch_frames = 0
    finite = True

    for item in trace:
        step = int(item["step"])
        errors = item["errors"]
        for metric, threshold_name in ERROR_TO_THRESHOLD.items():
            value = float(errors[metric])
            finite &= bool(np.isfinite(value))
            maxima[metric] = max(maxima[metric], value)
            threshold = float(getattr(thresholds, threshold_name))
            if value > threshold and first_divergence[metric] is None:
                first_divergence[metric] = step
        contact_mismatch_frames += int(int(errors["contact_proxy_mismatch_count"]) > 0)

    contact_gate = contact_mismatch_frames <= thresholds.contact_proxy_mismatch_frames
    metric_gates = {
        metric: first_divergence[metric] is None for metric in ERROR_TO_THRESHOLD
    }
    return {
        "finite": finite,
        "max_errors": maxima,
        "first_divergence_step": first_divergence,
        "contact_proxy_mismatch_frames": contact_mismatch_frames,
        "gates": {
            **metric_gates,
            "contact_proxy": contact_gate,
            "finite": finite,
        },
        "parity_gate_passed": finite and contact_gate and all(metric_gates.values()),
    }
