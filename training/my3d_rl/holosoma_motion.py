"""Convert audited Holosoma T1 qpos output into the project motion schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contract import PolicyContract
from .rcss_scene import build_single_t1_soccer_model


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid zero or non-finite quaternion")
    return quaternion / norm


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_slerp_wxyz(
    start: np.ndarray, end: np.ndarray, fraction: float
) -> np.ndarray:
    """Shortest-arc spherical interpolation for scalar-first quaternions."""
    first = _normalize_quaternion(np.asarray(start, dtype=np.float64))
    second = _normalize_quaternion(np.asarray(end, dtype=np.float64))
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(first + fraction * (second - first))
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return _normalize_quaternion(
        np.sin((1.0 - fraction) * angle) / scale * first
        + np.sin(fraction * angle) / scale * second
    )


def resample_qpos(qpos: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    """Resample MuJoCo free-root plus 23 hinge positions without Euler angles."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 30 or values.shape[0] < 2:
        raise ValueError(f"expected qpos shape (T, 30) with T >= 2, got {values.shape}")
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError("input and output frequencies must be positive")

    source_time = np.arange(values.shape[0], dtype=np.float64) / input_fps
    duration = source_time[-1]
    frame_count = int(round(duration * output_fps)) + 1
    target_time = np.arange(frame_count, dtype=np.float64) / output_fps
    target_time[-1] = duration

    result = np.empty((frame_count, 30), dtype=np.float64)
    for column in [0, 1, 2, *range(7, 30)]:
        result[:, column] = np.interp(target_time, source_time, values[:, column])

    quaternions = values[:, 3:7].copy()
    for index in range(1, quaternions.shape[0]):
        quaternions[index] = _normalize_quaternion(quaternions[index])
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    quaternions[0] = _normalize_quaternion(quaternions[0])
    upper = np.searchsorted(source_time, target_time, side="right")
    upper = np.clip(upper, 1, len(source_time) - 1)
    lower = upper - 1
    interval = source_time[upper] - source_time[lower]
    fractions = np.divide(
        target_time - source_time[lower],
        interval,
        out=np.zeros_like(target_time),
        where=interval > 0.0,
    )
    result[:, 3:7] = np.stack(
        [
            quaternion_slerp_wxyz(quaternions[lo], quaternions[hi], float(alpha))
            for lo, hi, alpha in zip(lower, upper, fractions)
        ]
    )
    return result


def canonicalize_forward(qpos: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the horizontal trajectory to +X and put its first XY at zero."""
    result = np.asarray(qpos, dtype=np.float64).copy()
    displacement = result[-1, :2] - result[0, :2]
    distance = float(np.linalg.norm(displacement))
    if distance < 1.0e-4:
        raise ValueError("motion has less than 0.1 mm horizontal displacement")
    heading = float(np.arctan2(displacement[1], displacement[0]))
    angle = -heading
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    result[:, :2] = (result[:, :2] - result[0, :2]) @ rotation.T
    yaw_quaternion = np.array([np.cos(0.5 * angle), 0.0, 0.0, np.sin(0.5 * angle)])
    result[:, 3:7] = np.stack(
        [
            _normalize_quaternion(quaternion_multiply_wxyz(yaw_quaternion, value))
            for value in result[:, 3:7]
        ]
    )
    return result, heading


def source_contact_from_human(
    human_joints: np.ndarray, *, height_threshold: float = 0.02
) -> np.ndarray:
    """Return [left, right] source contacts in Holosoma's LAFAN registry order."""
    joints = np.asarray(human_joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected human_joints shape (T, 22, 3), got {joints.shape}")
    # Holosoma registry: RightToeBase=4, LeftToeBase=8.
    return joints[:, [8, 4], 2] <= height_threshold


def resample_contact_nearest(
    contacts: np.ndarray, input_fps: float, output_frame_count: int, output_fps: float
) -> np.ndarray:
    source = np.asarray(contacts, dtype=bool)
    target_time = np.arange(output_frame_count, dtype=np.float64) / output_fps
    indices = np.rint(target_time * input_fps).astype(np.int64)
    return source[np.clip(indices, 0, source.shape[0] - 1)]


def replay_rcss_surface(
    qpos: np.ndarray, contract: PolicyContract
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prefix = "reference_"
    model = build_single_t1_soccer_model(prefix=prefix, robot_x=0.0, robot_y=0.0)
    data = mujoco.MjData(model)
    root_qpos = model.joint(prefix + "root").qposadr[0]
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    pitch = model.geom("pitch").id
    feet = [
        model.geom(prefix + "left_foot").id,
        model.geom(prefix + "right_foot").id,
    ]
    robot_bodies = {
        body_id
        for body_id in range(model.nbody)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        ).startswith(prefix)
    }

    lowest_points: list[list[float]] = []
    contacts: list[list[bool]] = []
    non_foot_pitch_frames = 0
    minimum_contact_distance = 0.0
    for frame in qpos:
        mujoco.mj_resetData(model, data)
        data.qpos[root_qpos : root_qpos + 7] = frame[:7]
        data.qpos[joint_qpos] = frame[7:]
        mujoco.mj_forward(model, data)
        lowest_points.append(
            [
                float(
                    data.geom_xpos[geom, 2]
                    - np.sum(
                        np.abs(data.geom_xmat[geom].reshape(3, 3)[2])
                        * model.geom_size[geom]
                    )
                )
                for geom in feet
            ]
        )
        pairs = {
            (
                min(data.contact[index].geom1, data.contact[index].geom2),
                max(data.contact[index].geom1, data.contact[index].geom2),
            )
            for index in range(data.ncon)
        }
        contacts.append(
            [(min(pitch, geom), max(pitch, geom)) in pairs for geom in feet]
        )
        bad_pitch = False
        for index in range(data.ncon):
            contact = data.contact[index]
            if pitch not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == pitch else contact.geom1
            if model.geom_bodyid[other] in robot_bodies and other not in feet:
                bad_pitch = True
            minimum_contact_distance = min(
                minimum_contact_distance, float(contact.dist)
            )
        non_foot_pitch_frames += int(bad_pitch)
    return (
        np.asarray(lowest_points, dtype=np.float64),
        np.asarray(contacts, dtype=bool),
        {
            "non_foot_pitch_contact_frames": non_foot_pitch_frames,
            "minimum_contact_distance_m": minimum_contact_distance,
        },
    )


def ground_reference_on_rcss(
    qpos: np.ndarray,
    source_contact: np.ndarray,
    contract: PolicyContract,
    *,
    contact_penetration: float = 0.001,
    maximum_penetration: float = 0.014,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align stance frames to the exact RCSS T1 foot boxes and replay contacts."""
    result = np.asarray(qpos, dtype=np.float64).copy()
    source = np.asarray(source_contact, dtype=bool)
    if source.shape != (result.shape[0], 2):
        raise ValueError("source_contact must have shape (T, 2)")
    lowest_before, contacts_before, _ = replay_rcss_surface(result, contract)
    stance = source.any(axis=1)
    if not stance.any():
        raise ValueError("source motion contains no stance frame")

    anchors = np.full(result.shape[0], np.nan, dtype=np.float64)
    # Use the lower RCSS foot as the support point.  This avoids forcing the
    # non-support foot through the pitch when retargeted side labels disagree.
    anchors[stance] = -contact_penetration - lowest_before[stance].min(axis=1)
    indices = np.flatnonzero(stance)
    offsets = np.interp(np.arange(result.shape[0]), indices, anchors[indices])
    result[:, 2] += offsets

    lowest_after, contacts_after, replay = replay_rcss_surface(result, contract)
    safety_lift = np.maximum(0.0, -maximum_penetration - lowest_after.min(axis=1))
    if np.any(safety_lift):
        result[:, 2] += safety_lift
        offsets += safety_lift
        lowest_after, contacts_after, replay = replay_rcss_surface(result, contract)
    replay.update(
        {
            "contact_count_before": contacts_before.sum(axis=0).astype(int).tolist(),
            "contact_count_after": contacts_after.sum(axis=0).astype(int).tolist(),
            "ground_offset_min_m": float(offsets.min()),
            "ground_offset_max_m": float(offsets.max()),
            "ground_offset_max_step_m": float(np.abs(np.diff(offsets)).max()),
            "penetration_safety_lift_frame_count": int(np.count_nonzero(safety_lift)),
            "penetration_safety_lift_max_m": float(safety_lift.max()),
            "maximum_allowed_penetration_m": maximum_penetration,
            "foot_lowest_min_m": lowest_after.min(axis=0).tolist(),
            "foot_lowest_max_m": lowest_after.max(axis=0).tolist(),
        }
    )
    return result, contacts_after, replay


def _angular_velocity_world_wxyz(quaternions: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    result = np.zeros((values.shape[0], 3), dtype=np.float64)
    for index in range(values.shape[0]):
        before = max(0, index - 1)
        after = min(values.shape[0] - 1, index + 1)
        elapsed = (after - before) * dt
        inverse = values[before].copy()
        inverse[1:] *= -1.0
        delta = _normalize_quaternion(quaternion_multiply_wxyz(values[after], inverse))
        if delta[0] < 0.0:
            delta *= -1.0
        sine = float(np.linalg.norm(delta[1:]))
        if sine > 1.0e-12 and elapsed > 0.0:
            axis = delta[1:] / sine
            result[index] = axis * (2.0 * np.arctan2(sine, delta[0]) / elapsed)
    return result


def build_motion_reference(
    holosoma_path: Path,
    contract: PolicyContract,
    *,
    output_fps: float,
    frame_start: int,
    frame_end_inclusive: int | None,
    source_height_threshold: float,
    provenance: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build schema arrays and diagnostic metadata from a Holosoma NPZ."""
    with np.load(holosoma_path, allow_pickle=False) as archive:
        if not {"qpos", "human_joints", "fps"}.issubset(archive.files):
            raise ValueError("Holosoma archive must contain qpos, human_joints and fps")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        human = np.asarray(archive["human_joints"], dtype=np.float64)
        input_fps = float(archive["fps"].item())
    end = qpos.shape[0] - 1 if frame_end_inclusive is None else frame_end_inclusive
    if frame_start < 0 or end < frame_start or end >= qpos.shape[0]:
        raise ValueError(f"invalid inclusive frame range {frame_start}:{end}")
    qpos = qpos[frame_start : end + 1]
    human = human[frame_start : end + 1]

    source_contact = source_contact_from_human(
        human, height_threshold=source_height_threshold
    )
    resampled = resample_qpos(qpos, input_fps, output_fps)
    source_contact = resample_contact_nearest(
        source_contact, input_fps, resampled.shape[0], output_fps
    )
    canonical, original_heading = canonicalize_forward(resampled)
    grounded, foot_contact, replay = ground_reference_on_rcss(
        canonical, source_contact, contract
    )

    dt = 1.0 / output_fps
    root_linear_velocity = np.gradient(grounded[:, :3], dt, axis=0)
    joint_velocity = np.gradient(grounded[:, 7:], dt, axis=0)
    root_angular_velocity = _angular_velocity_world_wxyz(grounded[:, 3:7], dt)
    duration = (grounded.shape[0] - 1) / output_fps
    displacement = grounded[-1, :2] - grounded[0, :2]
    average_speed = float(np.linalg.norm(displacement) / duration)

    metadata = dict(provenance)
    metadata.update(
        {
            "holosoma_input": str(holosoma_path.resolve()),
            "holosoma_frame_range_inclusive": [frame_start, end],
            "input_frequency_hz": input_fps,
            "output_frequency_hz": output_fps,
            "original_horizontal_heading_rad": original_heading,
            "source_contact_height_threshold_m": source_height_threshold,
            "source_contact_count": source_contact.sum(axis=0).astype(int).tolist(),
            "average_horizontal_speed_m_s": average_speed,
            "rcss_replay": replay,
        }
    )
    arrays = {
        "root_position": grounded[:, :3].astype(np.float32),
        "root_quaternion_xyzw": grounded[:, [4, 5, 6, 3]].astype(np.float32),
        "root_linear_velocity": root_linear_velocity.astype(np.float32),
        "root_angular_velocity": root_angular_velocity.astype(np.float32),
        "joint_position": grounded[:, 7:].astype(np.float32),
        "joint_velocity": joint_velocity.astype(np.float32),
        "foot_contact": foot_contact.astype(bool),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    return arrays, metadata
