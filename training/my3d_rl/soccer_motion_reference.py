"""Build and validate non-periodic T1 soccer motion references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contract import PolicyContract
from .holosoma_motion import (
    _angular_velocity_world_wxyz,
    canonicalize_forward,
    ground_reference_on_rcss,
    resample_contact_nearest,
    resample_qpos,
)
from .motion_reference import ARRAY_DTYPES, ARRAY_SHAPES, PROVENANCE_FIELDS, sha256
from .rcss_scene import build_single_t1_soccer_model


def analyze_t1_kick_geometry(
    qpos: np.ndarray,
    foot_contact: np.ndarray,
    contract: PolicyContract,
    *,
    kick_leg: str,
    frequency_hz: float,
) -> dict[str, Any]:
    """Measure the retargeted swing and support-foot state in exact RCSS geometry."""
    values = np.asarray(qpos, dtype=np.float64)
    contact = np.asarray(foot_contact, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 7 + contract.action_size:
        raise ValueError("qpos must have shape [T, 7 + action_size]")
    if contact.shape != (values.shape[0], 2):
        raise ValueError("foot_contact must have shape [T, 2]")
    if kick_leg not in {"left", "right"} or frequency_hz <= 0.0:
        raise ValueError("invalid kick geometry configuration")

    prefix = "soccer_reference_"
    model = build_single_t1_soccer_model(prefix=prefix, robot_x=0.0, robot_y=0.0)
    data = mujoco.MjData(model)
    root_joint = model.joint(prefix + "root")
    root_qpos = int(root_joint.qposadr[0])
    root_body = int(root_joint.bodyid[0])
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    foot_geom = [
        model.geom(prefix + "left_foot").id,
        model.geom(prefix + "right_foot").id,
    ]
    foot_position = np.empty((values.shape[0], 2, 3), dtype=np.float64)
    root_tilt = np.empty(values.shape[0], dtype=np.float64)
    for index, frame in enumerate(values):
        mujoco.mj_resetData(model, data)
        data.qpos[root_qpos : root_qpos + 7] = frame[:7]
        data.qpos[joint_qpos] = frame[7:]
        mujoco.mj_forward(model, data)
        foot_position[index] = data.geom_xpos[foot_geom]
        root_matrix = data.xmat[root_body].reshape(3, 3)
        root_tilt[index] = np.arccos(np.clip(root_matrix[2, 2], -1.0, 1.0))

    dt = 1.0 / frequency_hz
    root_velocity = np.gradient(values[:, :3], dt, axis=0)
    foot_velocity = np.gradient(foot_position, dt, axis=0)
    relative_speed = np.linalg.norm(
        foot_velocity - root_velocity[:, None, :], axis=2
    )
    kick_index = 0 if kick_leg == "left" else 1
    support_index = 1 - kick_index
    peak_frame = int(np.argmax(relative_speed[:, kick_index]))
    window_start = max(0, peak_frame - 3)
    window_end = min(values.shape[0], peak_frame + 4)
    return {
        "kick_leg": kick_leg,
        "kick_leg_index": kick_index,
        "peak_kick_foot_relative_speed_m_s": float(
            relative_speed[peak_frame, kick_index]
        ),
        "peak_other_foot_relative_speed_m_s": float(
            np.max(relative_speed[:, support_index])
        ),
        "peak_kick_frame": peak_frame,
        "support_contact_near_peak": bool(
            np.any(contact[window_start:window_end, support_index])
        ),
        "kick_contact_near_peak": bool(
            np.any(contact[window_start:window_end, kick_index])
        ),
        "foot_relative_speed_p95_m_s": np.percentile(
            relative_speed, 95, axis=0
        ).tolist(),
        "maximum_root_tilt_rad": float(np.max(root_tilt)),
        "minimum_root_height_m": float(np.min(values[:, 2])),
        "horizontal_displacement_m": float(
            np.linalg.norm(values[-1, :2] - values[0, :2])
        ),
    }


def build_soccer_motion_reference(
    qpos: np.ndarray,
    source_contact: np.ndarray,
    contract: PolicyContract,
    *,
    input_fps: float,
    output_fps: float,
    kick_leg: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert one mapped robot trajectory into the local soccer schema."""
    if output_fps != 50.0:
        raise ValueError("soccer_motion_reference_v1 requires 50 Hz")
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
    geometry = analyze_t1_kick_geometry(
        grounded,
        foot_contact,
        contract,
        kick_leg=kick_leg,
        frequency_hz=output_fps,
    )
    metadata = dict(provenance)
    metadata.update(
        {
            "motion_type": "soccer_kick",
            "output_frequency_hz": output_fps,
            "kick_leg": kick_leg,
            "original_horizontal_heading_rad": original_heading,
            "rcss_replay": replay,
            "kick_geometry": geometry,
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
        "kick_leg": np.array(kick_leg),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    return arrays, metadata


def validate_soccer_motion_reference(path: Path) -> dict[str, Any]:
    """Apply the K0 physical-feasibility gate to one local-only T1 kick clip."""
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as archive:
        required = set(ARRAY_SHAPES) | {"kick_leg", "metadata_json"}
        missing = sorted(required - set(archive.files))
        arrays: dict[str, np.ndarray] = {}
        frame_count = 0
        if missing:
            errors.append(f"missing arrays: {missing}")
        else:
            arrays = {name: archive[name] for name in ARRAY_SHAPES}
            frame_count = int(arrays["joint_position"].shape[0])
            for name, width in ARRAY_SHAPES.items():
                value = arrays[name]
                if value.shape != (frame_count, width):
                    errors.append(
                        f"{name} shape {value.shape} != ({frame_count}, {width})"
                    )
                if value.dtype != ARRAY_DTYPES[name]:
                    errors.append(
                        f"{name} dtype {value.dtype} != {np.dtype(ARRAY_DTYPES[name])}"
                    )
                if name != "foot_contact" and not np.isfinite(value).all():
                    errors.append(f"{name} contains non-finite values")
        kick_leg = ""
        if "kick_leg" in archive.files:
            kick_leg_array = np.asarray(archive["kick_leg"])
            if kick_leg_array.shape == ():
                kick_leg = str(kick_leg_array.item()).strip().lower()
        if kick_leg not in {"left", "right"}:
            errors.append(f"invalid kick_leg {kick_leg!r}")
        metadata: dict[str, Any] = {}
        if "metadata_json" in archive.files:
            try:
                metadata = json.loads(str(archive["metadata_json"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid metadata_json: {exc}")

    if frame_count < 25:
        errors.append(f"frame count {frame_count} is below 25")
    if arrays:
        quaternion_norm = np.linalg.norm(arrays["root_quaternion_xyzw"], axis=1)
        if not np.allclose(quaternion_norm, 1.0, atol=1.0e-3):
            errors.append("root quaternions are not normalized within 1e-3")
        contact_count = arrays["foot_contact"].sum(axis=0).astype(int).tolist()
        if min(contact_count) < 2:
            errors.append(f"insufficient foot contacts {contact_count}")
        maximum_joint_velocity = float(np.max(np.abs(arrays["joint_velocity"])))
        if maximum_joint_velocity > 40.0:
            errors.append(
                f"maximum joint velocity {maximum_joint_velocity:.3f} exceeds 40 rad/s"
            )
    else:
        contact_count = [0, 0]
        maximum_joint_velocity = 0.0

    for field in (*PROVENANCE_FIELDS, "retarget_method"):
        if not metadata.get(field):
            errors.append(f"missing provenance field {field}")
    if metadata.get("motion_type") != "soccer_kick":
        errors.append("metadata motion_type is not soccer_kick")
    if metadata.get("output_frequency_hz") != 50.0:
        errors.append("metadata output_frequency_hz is not 50")
    if metadata.get("kick_leg") != kick_leg:
        errors.append("metadata kick_leg differs from archive")

    replay = metadata.get("rcss_replay", {})
    if replay.get("non_foot_pitch_contact_frames", 1) != 0:
        errors.append("RCSS replay contains non-foot pitch contact")
    if replay.get("minimum_contact_distance_m", -1.0) < -0.015:
        errors.append("RCSS replay penetration exceeds 15 mm")
    if replay.get("ground_offset_max_step_m", 1.0) > 0.05:
        errors.append("ground alignment changes by more than 5 cm per frame")

    clipping = metadata.get("competition_joint_limit_clipping", {})
    if clipping.get("maximum_abs_correction_rad", 1.0) > 0.35:
        errors.append("joint-limit correction exceeds 0.35 rad")
    geometry = metadata.get("kick_geometry", {})
    peak_kick_speed = geometry.get("peak_kick_foot_relative_speed_m_s", 0.0)
    peak_other_speed = geometry.get("peak_other_foot_relative_speed_m_s", 1.0e9)
    if peak_kick_speed < 0.75:
        errors.append("kick-foot relative speed is below 0.75 m/s")
    if peak_kick_speed < 1.2 * peak_other_speed:
        errors.append("kick-leg label is not dominant by a 1.2 speed ratio")
    if not geometry.get("support_contact_near_peak", False):
        errors.append("support foot has no exact contact near peak kick speed")
    if geometry.get("maximum_root_tilt_rad", 10.0) > 1.0:
        errors.append("root tilt exceeds 1.0 rad")
    if geometry.get("minimum_root_height_m", 0.0) < 0.30:
        errors.append("root height falls below 0.30 m")

    return {
        "schema_version": 1,
        "purpose": "t1_soccer_motion_reference_k0_gate",
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "frame_count": frame_count,
        "frequency_hz": 50,
        "kick_leg": kick_leg,
        "foot_contact_frames": contact_count,
        "maximum_joint_velocity_rad_s": maximum_joint_velocity,
        "provenance": metadata,
        "errors": errors,
        "passed": not errors,
    }
