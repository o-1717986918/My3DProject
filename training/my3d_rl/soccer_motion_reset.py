"""Shared deterministic reset perturbations for exact soccer-motion work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoccerMotionResetPerturbation:
    case_seed: int
    yaw: float
    joint_position_noise: np.ndarray
    root_velocity_noise: np.ndarray


def derive_case_seed(base_seed: int, *coordinates: int) -> int:
    values = (base_seed, *coordinates)
    if any(value < 0 for value in values):
        raise ValueError("reset seed coordinates must be non-negative")
    # Keep derived seeds inside signed int64 so provenance arrays can use -1 as
    # an explicit "not applicable" sentinel without overflow or wraparound.
    raw = np.random.SeedSequence(values).generate_state(1, dtype=np.uint64)[0]
    return int(raw & np.uint64(0x7FFF_FFFF_FFFF_FFFF))


def deterministic_reset_perturbation(
    *,
    base_seed: int,
    motion: int,
    start_frame: int,
    action_size: int,
    joint_noise: float,
    root_velocity_noise: float,
    yaw_range: float,
) -> SoccerMotionResetPerturbation:
    if action_size < 1 or min(joint_noise, root_velocity_noise, yaw_range) < 0.0:
        raise ValueError("reset perturbation envelope is invalid")
    case_seed = derive_case_seed(base_seed, motion, start_frame)
    rng = np.random.default_rng(case_seed)
    return SoccerMotionResetPerturbation(
        case_seed=case_seed,
        yaw=float(rng.uniform(-yaw_range, yaw_range)),
        joint_position_noise=rng.uniform(
            -joint_noise, joint_noise, size=action_size
        ),
        root_velocity_noise=rng.uniform(
            -root_velocity_noise, root_velocity_noise, size=6
        ),
    )


def yaw_quaternion_rotate(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("yaw rotation requires one finite wxyz quaternion")
    half = 0.5 * yaw
    yaw_w = np.cos(half)
    yaw_z = np.sin(half)
    w, x, y, z = quaternion
    return np.asarray(
        [
            yaw_w * w - yaw_z * z,
            yaw_w * x - yaw_z * y,
            yaw_w * y + yaw_z * x,
            yaw_w * z + yaw_z * w,
        ],
        dtype=np.float64,
    )


def yaw_vector_rotate(vector: np.ndarray, yaw: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("yaw rotation requires one finite xyz vector")
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            vector[2],
        ],
        dtype=np.float64,
    )
