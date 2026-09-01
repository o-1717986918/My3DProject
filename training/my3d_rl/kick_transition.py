"""Shared, deployable walk-to-kick transition-state features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jp
import numpy as np


NOMINAL_GAIT_FREQUENCY_HZ = 1.6
NEUTRAL_MAGNITUDE_RAD = 0.02
SUPPORT_SWITCH_SINE = 0.15


@dataclass(frozen=True)
class LocomotionPhase:
    """Continuous gait proxy and an estimated support state.

    ``sin_cos`` is derived only from joint positions and velocities available
    in the competition runtime.  ``support_hint`` uses the contract order
    ``left, double, right`` and is deliberately an estimate, not privileged
    simulator contact information.
    """

    sin_cos: np.ndarray
    support_hint: np.ndarray
    magnitude_rad: float


def hip_pitch_indices(joint_order: Sequence[str]) -> tuple[int, int]:
    """Return left/right hip-pitch indices with an explicit contract failure."""
    try:
        return joint_order.index("Left_Hip_Pitch"), joint_order.index(
            "Right_Hip_Pitch"
        )
    except ValueError as exc:
        raise ValueError("joint order must contain both hip-pitch joints") from exc


def estimate_locomotion_phase(
    joint_position_offset: np.ndarray,
    joint_velocity: np.ndarray,
    joint_order: Sequence[str],
    *,
    nominal_frequency_hz: float = NOMINAL_GAIT_FREQUENCY_HZ,
    neutral_magnitude_rad: float = NEUTRAL_MAGNITUDE_RAD,
    support_switch_sine: float = SUPPORT_SWITCH_SINE,
) -> LocomotionPhase:
    """Estimate a continuous phase from the left/right hip oscillator.

    The position difference and velocity divided by nominal angular frequency
    have matching radian units.  Normalizing that two-dimensional state avoids
    coupling phase to gait amplitude.  Near the origin the phase is undefined,
    so the stable neutral encoding ``[0, 1]`` and double support are returned.
    """
    positions = np.asarray(joint_position_offset, dtype=np.float64)
    velocities = np.asarray(joint_velocity, dtype=np.float64)
    expected_shape = (len(joint_order),)
    if positions.shape != expected_shape or velocities.shape != expected_shape:
        raise ValueError(
            "joint position/velocity must match the declared joint order"
        )
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("joint phase inputs must be finite")
    if nominal_frequency_hz <= 0.0 or neutral_magnitude_rad <= 0.0:
        raise ValueError("phase frequency and neutral magnitude must be positive")
    if not 0.0 <= support_switch_sine < 1.0:
        raise ValueError("support switch sine must be in [0, 1)")

    left, right = hip_pitch_indices(joint_order)
    position_signal = positions[right] - positions[left]
    angular_frequency = 2.0 * np.pi * nominal_frequency_hz
    velocity_signal = (velocities[right] - velocities[left]) / angular_frequency
    magnitude = float(np.hypot(position_signal, velocity_signal))
    if magnitude < neutral_magnitude_rad:
        return LocomotionPhase(
            sin_cos=np.array([0.0, 1.0], dtype=np.float32),
            support_hint=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            magnitude_rad=magnitude,
        )

    sin_cos = np.array(
        [position_signal / magnitude, velocity_signal / magnitude],
        dtype=np.float32,
    )
    if sin_cos[0] > support_switch_sine:
        support_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif sin_cos[0] < -support_switch_sine:
        support_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        support_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return LocomotionPhase(sin_cos, support_hint, magnitude)


def estimate_locomotion_phase_jax(
    joint_position_offset,
    joint_velocity,
    left_hip_pitch_index: int,
    right_hip_pitch_index: int,
):
    """JAX equivalent of :func:`estimate_locomotion_phase` for MJX rollouts."""
    position_signal = (
        joint_position_offset[right_hip_pitch_index]
        - joint_position_offset[left_hip_pitch_index]
    )
    velocity_signal = (
        joint_velocity[right_hip_pitch_index]
        - joint_velocity[left_hip_pitch_index]
    ) / (2.0 * jp.pi * NOMINAL_GAIT_FREQUENCY_HZ)
    magnitude = jp.hypot(position_signal, velocity_signal)
    neutral = magnitude < NEUTRAL_MAGNITUDE_RAD
    safe_magnitude = jp.maximum(magnitude, NEUTRAL_MAGNITUDE_RAD)
    sin_cos = jp.where(
        neutral,
        jp.array([0.0, 1.0]),
        jp.array(
            [position_signal / safe_magnitude, velocity_signal / safe_magnitude]
        ),
    )
    support_hint = jp.where(
        neutral | (jp.abs(sin_cos[0]) <= SUPPORT_SWITCH_SINE),
        jp.array([0.0, 1.0, 0.0]),
        jp.where(
            sin_cos[0] > 0.0,
            jp.array([1.0, 0.0, 0.0]),
            jp.array([0.0, 0.0, 1.0]),
        ),
    )
    return sin_cos, support_hint, magnitude
