"""Strict local-only loader and robot-to-robot inputs for PAiD motions.

PAiD assets are CC BY-NC 4.0 and deliberately remain outside this repository.
This module contains only the independently implemented schema boundary needed
to audit an explicitly supplied upstream clone and construct T1 candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PAID_G1_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

PAID_G1_BODY_ORDER = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

# PAiD archives omit names.  These orders are pinned to the official exporter
# at e72e470230047dedaf66df0983f1d0ab746faeb5 and must not be guessed for any
# other revision.
PAID_SCHEMA_REVISION = "e72e470230047dedaf66df0983f1d0ab746faeb5"
PAID_SOURCE_LICENSE = "CC-BY-NC-4.0"

GMR_HUMAN_TO_PAID_BODY = {
    "pelvis": "pelvis",
    "spine3": "torso_link",
    "left_hip": "left_hip_roll_link",
    "right_hip": "right_hip_roll_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_shoulder": "left_shoulder_roll_link",
    "right_shoulder": "right_shoulder_roll_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}

SEMANTIC_T1_TO_PAID_JOINT = {
    "AAHead_yaw": None,
    "Head_pitch": None,
    "Left_Shoulder_Pitch": "left_shoulder_pitch_joint",
    "Left_Shoulder_Roll": "left_shoulder_roll_joint",
    "Left_Elbow_Pitch": "left_elbow_joint",
    "Left_Elbow_Yaw": "left_wrist_roll_joint",
    "Right_Shoulder_Pitch": "right_shoulder_pitch_joint",
    "Right_Shoulder_Roll": "right_shoulder_roll_joint",
    "Right_Elbow_Pitch": "right_elbow_joint",
    "Right_Elbow_Yaw": "right_wrist_roll_joint",
    "Waist": "waist_yaw_joint",
    "Left_Hip_Pitch": "left_hip_pitch_joint",
    "Left_Hip_Roll": "left_hip_roll_joint",
    "Left_Hip_Yaw": "left_hip_yaw_joint",
    "Left_Knee_Pitch": "left_knee_joint",
    "Left_Ankle_Pitch": "left_ankle_pitch_joint",
    "Left_Ankle_Roll": "left_ankle_roll_joint",
    "Right_Hip_Pitch": "right_hip_pitch_joint",
    "Right_Hip_Roll": "right_hip_roll_joint",
    "Right_Hip_Yaw": "right_hip_yaw_joint",
    "Right_Knee_Pitch": "right_knee_joint",
    "Right_Ankle_Pitch": "right_ankle_pitch_joint",
    "Right_Ankle_Roll": "right_ankle_roll_joint",
}


@dataclass(frozen=True)
class PaidMotionClip:
    path: Path
    fps: float
    kick_leg: str
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got {array.shape}")
    return str(array.item()).strip().lower()


def load_paid_motion(path: Path) -> PaidMotionClip:
    """Load one pinned PAiD NPZ without allowing Python object deserialization."""
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "kick_leg",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"PAiD archive is missing arrays: {missing}")
        fps_array = np.asarray(archive["fps"]).reshape(-1)
        if fps_array.size != 1:
            raise ValueError("PAiD fps must contain exactly one value")
        fps = float(fps_array[0])
        clip = PaidMotionClip(
            path=path.resolve(),
            fps=fps,
            kick_leg=_scalar_text(archive["kick_leg"], "kick_leg"),
            joint_position=np.asarray(archive["joint_pos"], dtype=np.float64),
            joint_velocity=np.asarray(archive["joint_vel"], dtype=np.float64),
            body_position_world=np.asarray(archive["body_pos_w"], dtype=np.float64),
            body_quaternion_wxyz=np.asarray(
                archive["body_quat_w"], dtype=np.float64
            ),
            body_linear_velocity_world=np.asarray(
                archive["body_lin_vel_w"], dtype=np.float64
            ),
            body_angular_velocity_world=np.asarray(
                archive["body_ang_vel_w"], dtype=np.float64
            ),
        )
    validate_paid_clip(clip)
    return clip


def validate_paid_clip(clip: PaidMotionClip) -> dict[str, Any]:
    errors: list[str] = []
    frames = clip.frame_count
    expected = {
        "joint_position": (frames, len(PAID_G1_JOINT_ORDER)),
        "joint_velocity": (frames, len(PAID_G1_JOINT_ORDER)),
        "body_position_world": (frames, len(PAID_G1_BODY_ORDER), 3),
        "body_quaternion_wxyz": (frames, len(PAID_G1_BODY_ORDER), 4),
        "body_linear_velocity_world": (frames, len(PAID_G1_BODY_ORDER), 3),
        "body_angular_velocity_world": (frames, len(PAID_G1_BODY_ORDER), 3),
    }
    for name, shape in expected.items():
        value = getattr(clip, name)
        if value.shape != shape:
            errors.append(f"{name} shape {value.shape} != {shape}")
        if not np.isfinite(value).all():
            errors.append(f"{name} contains non-finite values")
    if frames < 25:
        errors.append("PAiD clip contains fewer than 25 frames")
    if clip.fps != 50.0:
        errors.append(f"PAiD clip frequency {clip.fps} Hz != 50 Hz")
    if clip.kick_leg not in {"left", "right"}:
        errors.append(f"invalid kick_leg {clip.kick_leg!r}")
    if not clip.path.stem.endswith("_" + clip.kick_leg):
        errors.append("kick_leg does not match the filename suffix")
    quaternion_norm_error = float("inf")
    if clip.body_quaternion_wxyz.shape == expected["body_quaternion_wxyz"]:
        quaternion_norm_error = float(
            np.max(
                np.abs(np.linalg.norm(clip.body_quaternion_wxyz, axis=2) - 1.0)
            )
        )
        if quaternion_norm_error > 2.0e-3:
            errors.append(
                f"body quaternion norm error {quaternion_norm_error:.6f} "
                "exceeds 0.002"
            )
    if errors:
        raise ValueError("; ".join(errors))
    numerical_joint_velocity = np.gradient(
        clip.joint_position, 1.0 / clip.fps, axis=0
    )
    return {
        "frame_count": frames,
        "frequency_hz": clip.fps,
        "kick_leg": clip.kick_leg,
        "quaternion_norm_max_error": quaternion_norm_error,
        "joint_velocity_consistency_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(numerical_joint_velocity - clip.joint_velocity)))
        ),
        "root_height_range_m": [
            float(np.min(clip.body_position_world[:, 0, 2])),
            float(np.max(clip.body_position_world[:, 0, 2])),
        ],
    }


def source_foot_contact(
    clip: PaidMotionClip,
    *,
    height_tolerance_m: float = 0.04,
    maximum_vertical_speed_m_s: float = 0.40,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer local [left, right] source contacts from PAiD ankle-link states."""
    if height_tolerance_m <= 0.0 or maximum_vertical_speed_m_s <= 0.0:
        raise ValueError("contact tolerances must be positive")
    body_index = {name: index for index, name in enumerate(PAID_G1_BODY_ORDER)}
    indices = [
        body_index["left_ankle_roll_link"],
        body_index["right_ankle_roll_link"],
    ]
    position = clip.body_position_world[:, indices]
    velocity = clip.body_linear_velocity_world[:, indices]
    ground = float(np.min(position[:, :, 2]))
    contact = (position[:, :, 2] <= ground + height_tolerance_m) & (
        np.abs(velocity[:, :, 2]) <= maximum_vertical_speed_m_s
    )
    return contact, {
        "source_ground_height_m": ground,
        "height_tolerance_m": height_tolerance_m,
        "maximum_vertical_speed_m_s": maximum_vertical_speed_m_s,
        "contact_frames": contact.sum(axis=0).astype(int).tolist(),
    }


def paid_frame_for_gmr(clip: PaidMotionClip, frame_index: int) -> dict[str, list[np.ndarray]]:
    """Expose PAiD G1 bodies through the logical names used by GMR's T1 IK."""
    if not 0 <= frame_index < clip.frame_count:
        raise IndexError(frame_index)
    body_index = {name: index for index, name in enumerate(PAID_G1_BODY_ORDER)}
    result: dict[str, list[np.ndarray]] = {}
    for human_name, paid_name in GMR_HUMAN_TO_PAID_BODY.items():
        index = body_index[paid_name]
        quaternion = clip.body_quaternion_wxyz[frame_index, index].copy()
        quaternion /= np.linalg.norm(quaternion)
        result[human_name] = [
            clip.body_position_world[frame_index, index].copy(),
            quaternion,
        ]
    return result


def semantic_projection_qpos(
    clip: PaidMotionClip, target_joint_order: Sequence[str]
) -> np.ndarray:
    """Construct the explicit no-IK A baseline for robot-to-robot comparison."""
    if set(target_joint_order) != set(SEMANTIC_T1_TO_PAID_JOINT):
        raise ValueError("target joint order is incompatible with T1 semantic mapping")
    paid_index = {name: index for index, name in enumerate(PAID_G1_JOINT_ORDER)}
    qpos = np.zeros((clip.frame_count, 7 + len(target_joint_order)), dtype=np.float64)
    qpos[:, :3] = clip.body_position_world[:, 0]
    qpos[:, 3:7] = clip.body_quaternion_wxyz[:, 0]
    for target_index, target_name in enumerate(target_joint_order, start=7):
        source_name = SEMANTIC_T1_TO_PAID_JOINT[target_name]
        if source_name is not None:
            qpos[:, target_index] = clip.joint_position[:, paid_index[source_name]]
    return qpos
