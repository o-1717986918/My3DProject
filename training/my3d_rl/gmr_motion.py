"""Utilities for importing GMR Booster T1 motion into the 23-DoF contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


OPTIONAL_ZERO_JOINTS = frozenset({"AAHead_yaw", "Head_pitch"})


def map_named_qpos_to_contract(
    qpos: np.ndarray,
    source_qpos_addresses: Mapping[str, int],
    target_joint_order: Sequence[str],
    *,
    optional_zero_joints: frozenset[str] = OPTIONAL_ZERO_JOINTS,
) -> np.ndarray:
    """Map free-root qpos by joint name and zero explicitly unsupported joints.

    GMR's direct LAFAN1 configuration targets its extended Booster T1 model,
    which adds three wrist joints per arm and omits the two head joints.  The
    competition model uses the common shoulder/elbow, waist and leg names.
    Mapping by MuJoCo qpos address makes that compatibility boundary explicit.
    """
    source = np.asarray(qpos, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 8:
        raise ValueError(f"expected source qpos shape (T, nq), got {source.shape}")

    target = np.zeros((source.shape[0], 7 + len(target_joint_order)), dtype=np.float64)
    target[:, :7] = source[:, :7]
    missing: list[str] = []
    for target_index, name in enumerate(target_joint_order, start=7):
        address = source_qpos_addresses.get(name)
        if address is None:
            if name not in optional_zero_joints:
                missing.append(name)
            continue
        if address < 7 or address >= source.shape[1]:
            raise ValueError(f"invalid qpos address {address} for source joint {name}")
        target[:, target_index] = source[:, address]
    if missing:
        raise ValueError(f"source motion is missing required joints: {missing}")
    if not np.isfinite(target).all():
        raise ValueError("mapped qpos contains non-finite values")
    quaternion_norm = np.linalg.norm(target[:, 3:7], axis=1)
    if np.any(quaternion_norm < 1.0e-12):
        raise ValueError("mapped qpos contains an invalid root quaternion")
    target[:, 3:7] /= quaternion_norm[:, None]
    return target


def contact_only_human_joints(
    left_foot_positions: np.ndarray,
    right_foot_positions: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Build the minimal Holosoma-compatible contact channel used downstream.

    Only joint slots 8 (left toe) and 4 (right toe) are consumed by the shared
    importer.  Z is shifted by the common minimum solely for contact labelling;
    GMR robot qpos is not modified here.
    """
    left = np.asarray(left_foot_positions, dtype=np.float64)
    right = np.asarray(right_foot_positions, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError(
            "expected matching foot positions with shape (T, 3), got "
            f"{left.shape} and {right.shape}"
        )
    if left.shape[0] < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("foot positions must contain at least two finite frames")
    ground = float(min(left[:, 2].min(), right[:, 2].min()))
    human = np.zeros((left.shape[0], 22, 3), dtype=np.float64)
    human[:, 8] = left
    human[:, 4] = right
    human[:, [8, 4], 2] -= ground
    return human, ground


def clip_contract_joint_limits(
    qpos: np.ndarray, joint_lower: np.ndarray, joint_upper: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clip mapped joint positions to the exact competition-model limits."""
    values = np.asarray(qpos, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 + lower.size:
        raise ValueError("qpos and joint limits have incompatible shapes")
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("invalid joint limits")
    joints = values[:, 7:]
    clipped_joints = np.clip(joints, lower, upper)
    correction = clipped_joints - joints
    result = values.copy()
    result[:, 7:] = clipped_joints
    return result, {
        "clipped_value_count": int(np.count_nonzero(correction)),
        "maximum_abs_correction_rad": float(np.max(np.abs(correction))),
        "mean_abs_correction_rad": float(np.mean(np.abs(correction))),
    }
