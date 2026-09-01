"""Exact-CPU dynamic replay diagnostics for finite T1 soccer motions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from .contract import PolicyContract
from .rcss_scene import DEFAULT_RESOURCE_ROOT, build_single_t1_soccer_model
from .reference_dynamics import configure_pd_actuators


REFERENCE_ARRAYS = (
    "root_position",
    "root_quaternion_xyzw",
    "root_linear_velocity",
    "root_angular_velocity",
    "joint_position",
    "joint_velocity",
    "foot_contact",
)


@dataclass(frozen=True)
class DynamicReplayThresholds:
    """K1 zero-residual screening thresholds, not a release gate."""

    maximum_joint_rmse_rad: float = 0.35
    maximum_joint_p95_abs_error_rad: float = 0.70
    minimum_contact_agreement: float = 0.70
    maximum_non_foot_pitch_contact_frames: int = 0


def load_soccer_motion_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load only non-object arrays required by the dynamic replay."""
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REFERENCE_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"missing soccer motion arrays: {missing}")
        return {name: np.asarray(archive[name]) for name in REFERENCE_ARRAYS}


def quaternion_distance_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    """Return the shortest unsigned angular distance between two quaternions."""
    first = np.asarray(first_wxyz, dtype=np.float64)
    second = np.asarray(second_wxyz, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("quaternion distance expects two four-vectors")
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("quaternions must have non-zero norm")
    cosine = abs(float(np.dot(first / first_norm, second / second_norm)))
    return float(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)))


def _validate_arrays(
    reference: Mapping[str, np.ndarray], action_size: int
) -> int:
    missing = sorted(set(REFERENCE_ARRAYS) - set(reference))
    if missing:
        raise ValueError(f"missing soccer motion arrays: {missing}")
    frame_count = int(np.asarray(reference["joint_position"]).shape[0])
    shapes = {
        "root_position": (frame_count, 3),
        "root_quaternion_xyzw": (frame_count, 4),
        "root_linear_velocity": (frame_count, 3),
        "root_angular_velocity": (frame_count, 3),
        "joint_position": (frame_count, action_size),
        "joint_velocity": (frame_count, action_size),
        "foot_contact": (frame_count, 2),
    }
    if frame_count < 2:
        raise ValueError("dynamic replay requires at least two frames")
    for name, shape in shapes.items():
        value = np.asarray(reference[name])
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if name != "foot_contact" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    return frame_count


def _contact_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    pitch_geom: int,
    foot_geoms: tuple[int, int],
    robot_geoms: frozenset[int],
) -> tuple[tuple[bool, bool], bool]:
    pairs = np.asarray(data.contact.geom[: data.ncon], dtype=np.int32)
    contacts = tuple(
        bool(
            np.any(
                ((pairs[:, 0] == pitch_geom) & (pairs[:, 1] == foot))
                | ((pairs[:, 1] == pitch_geom) & (pairs[:, 0] == foot))
            )
        )
        for foot in foot_geoms
    )
    non_foot = False
    for first, second in pairs:
        if first == pitch_geom and second in robot_geoms and second not in foot_geoms:
            non_foot = True
            break
        if second == pitch_geom and first in robot_geoms and first not in foot_geoms:
            non_foot = True
            break
    return contacts, non_foot


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def replay_soccer_motion_reference(
    model: mujoco.MjModel,
    contract: PolicyContract,
    reference: Mapping[str, np.ndarray],
    *,
    prefix: str = "soccer_track_",
    start_frame: int = 0,
    target_lead_frames: int = 0,
    kp: float | np.ndarray = 25.0,
    kd: float | np.ndarray = 0.6,
    thresholds: DynamicReplayThresholds = DynamicReplayThresholds(),
) -> dict[str, Any]:
    """Replay a finite reference from an exact state with zero learned residual."""
    frame_count = _validate_arrays(reference, contract.action_size)
    if not 0 <= start_frame < frame_count - 1:
        raise ValueError("start_frame must leave at least one control transition")
    if target_lead_frames < 0:
        raise ValueError("finite replay target lead must be non-negative")
    gains_p = np.broadcast_to(np.asarray(kp, dtype=np.float64), (contract.action_size,))
    gains_d = np.broadcast_to(np.asarray(kd, dtype=np.float64), (contract.action_size,))
    if np.any(gains_p <= 0.0) or np.any(gains_d < 0.0):
        raise ValueError("PD gains must satisfy kp > 0 and kd >= 0")
    control_dt = 1.0 / contract.frequency_hz
    substeps_float = control_dt / model.opt.timestep
    substeps = int(round(substeps_float))
    if not np.isclose(substeps, substeps_float, atol=1.0e-12):
        raise ValueError("physics timestep must divide the policy control period")

    tau_actuator, pos_actuator, vel_actuator = configure_pd_actuators(
        model, contract, kp=gains_p, kd=gains_d, prefix=prefix
    )
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    joint_dof = np.array(
        [model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
    )
    root = model.joint(prefix + "root")
    root_qpos = int(root.qposadr[0])
    root_dof = int(root.dofadr[0])
    torso_body = model.body(prefix + "torso").id
    torso_site = model.site(prefix + "torso").id
    pitch_geom = model.geom("pitch").id
    foot_geoms = (
        model.geom(prefix + "left_foot").id,
        model.geom(prefix + "right_foot").id,
    )
    robot_geoms = frozenset(
        index
        for index in range(model.ngeom)
        if (model.geom(index).name or "").startswith(prefix)
    )

    root_position = np.asarray(reference["root_position"], dtype=np.float64)
    root_quaternion = np.asarray(
        reference["root_quaternion_xyzw"], dtype=np.float64
    )[:, [3, 0, 1, 2]]
    root_linear_velocity = np.asarray(
        reference["root_linear_velocity"], dtype=np.float64
    )
    root_angular_velocity = np.asarray(
        reference["root_angular_velocity"], dtype=np.float64
    )
    joint_position = np.asarray(reference["joint_position"], dtype=np.float64)
    joint_velocity = np.asarray(reference["joint_velocity"], dtype=np.float64)
    desired_contact = np.asarray(reference["foot_contact"], dtype=bool)

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    model_xy_origin = model.qpos0[root_qpos : root_qpos + 2].copy()
    reference_xy_origin = root_position[start_frame, :2].copy()
    data.qpos[root_qpos : root_qpos + 2] = model_xy_origin
    data.qpos[root_qpos + 2] = root_position[start_frame, 2]
    data.qpos[root_qpos + 3 : root_qpos + 7] = root_quaternion[start_frame]
    data.qpos[joint_qpos] = joint_position[start_frame]
    data.qvel[root_dof : root_dof + 3] = root_linear_velocity[start_frame]
    data.qvel[root_dof + 3 : root_dof + 6] = root_angular_velocity[start_frame]
    data.qvel[joint_dof] = joint_velocity[start_frame]
    data.ctrl[tau_actuator] = 0.0
    data.ctrl[vel_actuator] = 0.0
    data.ctrl[pos_actuator] = joint_position[start_frame]
    mujoco.mj_forward(model, data)

    joint_abs_error: list[float] = []
    joint_squared_error: list[float] = []
    root_position_error: list[float] = []
    root_orientation_error: list[float] = []
    actuator_force: list[float] = []
    contact_matches = 0
    contact_observations = 0
    flight_frames = 0
    non_foot_pitch_contact_frames = 0
    minimum_torso_height = float("inf")
    minimum_upright = float("inf")
    termination_frame: int | None = None
    termination_reason = "completed"
    evaluated_frames = 0

    for frame in range(start_frame + 1, frame_count):
        target_frame = min(frame + target_lead_frames, frame_count - 1)
        data.ctrl[pos_actuator] = joint_position[target_frame]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        evaluated_frames += 1

        joint_error = data.qpos[joint_qpos] - joint_position[frame]
        joint_abs_error.extend(np.abs(joint_error).tolist())
        joint_squared_error.extend(np.square(joint_error).tolist())
        desired_root = root_position[frame].copy()
        desired_root[:2] = (
            model_xy_origin + root_position[frame, :2] - reference_xy_origin
        )
        root_position_error.append(
            float(np.linalg.norm(data.qpos[root_qpos : root_qpos + 3] - desired_root))
        )
        root_orientation_error.append(
            quaternion_distance_rad(
                data.qpos[root_qpos + 3 : root_qpos + 7], root_quaternion[frame]
            )
        )
        actuator_force.extend(np.abs(data.qfrc_actuator[joint_dof]).tolist())

        contacts, non_foot = _contact_state(
            model,
            data,
            pitch_geom=pitch_geom,
            foot_geoms=foot_geoms,
            robot_geoms=robot_geoms,
        )
        contact_matches += int(np.sum(np.asarray(contacts) == desired_contact[frame]))
        contact_observations += 2
        flight_frames += int(not any(contacts))
        non_foot_pitch_contact_frames += int(non_foot)

        rotation = data.site_xmat[torso_site].reshape(3, 3)
        upright = float(rotation[2, 2])
        torso_height = float(data.xpos[torso_body, 2])
        minimum_upright = min(minimum_upright, upright)
        minimum_torso_height = min(minimum_torso_height, torso_height)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            termination_frame = frame
            termination_reason = "non_finite_state"
            break
        if torso_height < 0.35 or upright < 0.20:
            termination_frame = frame
            termination_reason = "fall"
            break

    completed = termination_frame is None
    position_stats = _percentiles(joint_abs_error)
    root_position_stats = _percentiles(root_position_error)
    orientation_stats = _percentiles(root_orientation_error)
    force_stats = _percentiles(actuator_force)
    joint_rmse = float(np.sqrt(np.mean(joint_squared_error)))
    contact_agreement = contact_matches / max(contact_observations, 1)
    screening_errors: list[str] = []
    if not completed:
        screening_errors.append(termination_reason)
    if joint_rmse > thresholds.maximum_joint_rmse_rad:
        screening_errors.append("joint_rmse")
    if position_stats["p95"] > thresholds.maximum_joint_p95_abs_error_rad:
        screening_errors.append("joint_p95_abs_error")
    if contact_agreement < thresholds.minimum_contact_agreement:
        screening_errors.append("contact_agreement")
    if (
        non_foot_pitch_contact_frames
        > thresholds.maximum_non_foot_pitch_contact_frames
    ):
        screening_errors.append("non_foot_pitch_contact")

    return {
        "schema_version": 1,
        "purpose": "k1_zero_residual_exact_cpu_dynamic_screening",
        "engine": f"MuJoCo {mujoco.__version__}",
        "start_frame": start_frame,
        "target_lead_frames": target_lead_frames,
        "frame_count": frame_count,
        "evaluated_frames": evaluated_frames,
        "completed": completed,
        "termination_frame": termination_frame,
        "termination_phase": (
            1.0 if completed else float(termination_frame / (frame_count - 1))
        ),
        "termination_reason": termination_reason,
        "kp": gains_p.tolist(),
        "kd": gains_d.tolist(),
        "control_frequency_hz": contract.frequency_hz,
        "physics_timestep_s": float(model.opt.timestep),
        "joint_tracking_rmse_rad": joint_rmse,
        "joint_abs_error_rad": position_stats,
        "root_position_error_m": root_position_stats,
        "root_orientation_error_rad": orientation_stats,
        "joint_actuator_force": force_stats,
        "foot_contact_agreement": float(contact_agreement),
        "flight_frame_rate": float(flight_frames / max(evaluated_frames, 1)),
        "non_foot_pitch_contact_frames": non_foot_pitch_contact_frames,
        "minimum_torso_height_m": minimum_torso_height,
        "minimum_upright_cosine": minimum_upright,
        "screening_thresholds": {
            "maximum_joint_rmse_rad": thresholds.maximum_joint_rmse_rad,
            "maximum_joint_p95_abs_error_rad": (
                thresholds.maximum_joint_p95_abs_error_rad
            ),
            "minimum_contact_agreement": thresholds.minimum_contact_agreement,
            "maximum_non_foot_pitch_contact_frames": (
                thresholds.maximum_non_foot_pitch_contact_frames
            ),
        },
        "screening_errors": screening_errors,
        "screening_passed": not screening_errors,
    }


def evaluate_soccer_motion_path(
    path: Path,
    contract: PolicyContract,
    *,
    start_frame: int = 0,
    target_lead_frames: int = 0,
    kp: float | np.ndarray = 25.0,
    kd: float | np.ndarray = 0.6,
    resource_root: Path = DEFAULT_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Build a fresh exact scene and replay one local-only motion artifact."""
    prefix = "soccer_track_"
    model = build_single_t1_soccer_model(
        resource_root, prefix=prefix, robot_x=-10.0, robot_y=0.0
    )
    model.opt.timestep = 0.005
    return replay_soccer_motion_reference(
        model,
        contract,
        load_soccer_motion_arrays(path),
        prefix=prefix,
        start_frame=start_frame,
        target_lead_frames=target_lead_frames,
        kp=kp,
        kd=kd,
    )
