"""Deterministic contact-aware projection of T1 motion onto a periodic cycle."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import PolicyContract
from .holosoma_motion import ground_reference_on_rcss, replay_rcss_surface
from .policy_symmetry import physical_mirror_map
from .rcss_scene import build_single_t1_soccer_model


def circular_gradient(
    values: np.ndarray, dt: float, cycle_delta: np.ndarray | None = None
) -> np.ndarray:
    """Central difference over a cycle, optionally with translational progress."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3:
        raise ValueError("circular gradient requires a [frames, width] array")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    delta = (
        np.zeros(array.shape[1], dtype=np.float64)
        if cycle_delta is None
        else np.asarray(cycle_delta, dtype=np.float64)
    )
    if delta.shape != (array.shape[1],):
        raise ValueError("cycle_delta has incompatible shape")
    before = np.roll(array, 1, axis=0)
    after = np.roll(array, -1, axis=0)
    before[0] -= delta
    after[-1] += delta
    return (after - before) / (2.0 * dt)


def project_half_cycle(
    values: np.ndarray,
    source: np.ndarray,
    factor: np.ndarray,
    source_half_weight: float = 0.5,
) -> np.ndarray:
    """Project an even sequence onto exact half-cycle reflection."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] % 2:
        raise ValueError("half-cycle projection requires an even 2D sequence")
    if source.shape != (array.shape[1],) or factor.shape != (array.shape[1],):
        raise ValueError("reflection map has incompatible shape")
    if not 0.0 <= source_half_weight <= 1.0:
        raise ValueError("source_half_weight must be in [0, 1]")
    half = array.shape[0] // 2
    mirrored_second = array[half:, source] * factor
    first = (
        source_half_weight * array[:half] + (1.0 - source_half_weight) * mirrored_second
    )
    second = first[:, source] * factor
    return np.concatenate([first, second], axis=0)


def mirror_root_quaternion_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Reflect root orientation through the sagittal XZ plane."""
    values = np.asarray(quaternions, dtype=np.float64)
    original_shape = values.shape
    flat = values.reshape(-1, 4)
    reflection = np.diag([1.0, -1.0, 1.0])
    matrices = Rotation.from_quat(flat).as_matrix()
    mirrored = reflection @ matrices @ reflection
    result = Rotation.from_matrix(mirrored).as_quat().reshape(original_shape)
    return _continuous_quaternions(result)


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)
    for index in range(1, result.shape[0]):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _midpoint_quaternions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = _continuous_quaternions(first)
    right = _continuous_quaternions(second)
    right = np.where((np.sum(left * right, axis=1) < 0.0)[:, None], -right, right)
    return _continuous_quaternions(left + right)


def _project_root_orientation(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] % 2:
        raise ValueError("root quaternion sequence must have even shape [frames, 4]")
    half = values.shape[0] // 2
    first = _midpoint_quaternions(
        values[:half], mirror_root_quaternion_xyzw(values[half:])
    )
    return _continuous_quaternions(
        np.concatenate([first, mirror_root_quaternion_xyzw(first)], axis=0)
    )


def _project_root_position(
    root_position: np.ndarray, cycle_delta: np.ndarray
) -> np.ndarray:
    values = np.asarray(root_position, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] % 2:
        raise ValueError("root position sequence must have even shape [frames, 3]")
    half = values.shape[0] // 2
    half_delta = 0.5 * np.asarray(cycle_delta, dtype=np.float64)
    mirrored_second = values[half:] - half_delta
    mirrored_second[:, 1] *= -1.0
    first = 0.5 * (values[:half] + mirrored_second)
    second = first.copy()
    second[:, 1] *= -1.0
    second += half_delta
    result = np.concatenate([first, second], axis=0)
    result[:, 0] -= result[0, 0]
    return result


def _smooth_circular(values: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if passes < 0:
        raise ValueError("smoothing passes cannot be negative")
    for _ in range(passes):
        result = (
            0.25 * np.roll(result, 1, axis=0)
            + 0.5 * result
            + 0.25 * np.roll(result, -1, axis=0)
        )
    return result


def _circular_angular_velocity(quaternions_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rotations = Rotation.from_quat(_continuous_quaternions(quaternions_xyzw))
    result = np.zeros((len(rotations), 3), dtype=np.float64)
    for index in range(len(rotations)):
        before = rotations[(index - 1) % len(rotations)]
        after = rotations[(index + 1) % len(rotations)]
        result[index] = (after * before.inv()).as_rotvec() / (2.0 * dt)
    return result


def _foot_positions(qpos: np.ndarray, contract: PolicyContract) -> np.ndarray:
    prefix = "periodic_"
    model = build_single_t1_soccer_model(prefix=prefix, robot_x=0.0, robot_y=0.0)
    data = mujoco.MjData(model)
    root_qpos = model.joint(prefix + "root").qposadr[0]
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    feet = [
        model.geom(prefix + "left_foot").id,
        model.geom(prefix + "right_foot").id,
    ]
    positions = []
    for frame in qpos:
        mujoco.mj_resetData(model, data)
        data.qpos[root_qpos : root_qpos + 7] = frame[:7]
        data.qpos[joint_qpos] = frame[7:]
        mujoco.mj_forward(model, data)
        positions.append([data.geom_xpos[geom].copy() for geom in feet])
    return np.asarray(positions, dtype=np.float64)


def _circular_runs(mask: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(mask, dtype=bool)
    if not values.any():
        return []
    if values.all():
        return [np.arange(values.size)]
    false_index = int(np.flatnonzero(~values)[0])
    ordered = (false_index + 1 + np.arange(values.size)) % values.size
    runs: list[list[int]] = []
    current: list[int] = []
    for index in ordered:
        if values[index]:
            current.append(int(index))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [np.asarray(run, dtype=int) for run in runs]


def _long_contact_mask(contact: np.ndarray, minimum_frames: int = 4) -> np.ndarray:
    values = np.asarray(contact, dtype=bool)
    result = np.zeros_like(values)
    for side in range(values.shape[1]):
        for run in _circular_runs(values[:, side]):
            if run.size >= minimum_frames:
                result[run, side] = True
    return result


def _solve_stance_correction(
    foot_position: np.ndarray,
    contact: np.ndarray,
    cycle_delta: np.ndarray,
    smoothing_passes: int,
) -> np.ndarray:
    frame_count = foot_position.shape[0]
    half = frame_count // 2
    stable = _long_contact_mask(contact)

    def solve_axis(axis: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        right_hand_side: list[float] = []
        stance_weight = 10.0
        for side in range(2):
            for index in range(frame_count):
                next_index = (index + 1) % frame_count
                if not (stable[index, side] and stable[next_index, side]):
                    continue
                next_position = foot_position[next_index, side, axis]
                if index == frame_count - 1:
                    next_position += cycle_delta[axis]
                foot_delta = next_position - foot_position[index, side, axis]
                row = np.zeros(frame_count)
                row[index] = -stance_weight
                row[next_index] = stance_weight
                rows.append(row)
                right_hand_side.append(-stance_weight * foot_delta)

        # The source clip contains brief collision contacts that can otherwise
        # make an exact foot anchor pull the root backwards between steps.
        # Curvature regularization remains strong enough to preserve a
        # continuous forward root trajectory while the higher-weight stance
        # equations remove measurable slip on genuine contact runs.
        smooth_weight = 5.0 * max(1, smoothing_passes)
        for index in range(frame_count):
            row = np.zeros(frame_count)
            row[(index - 1) % frame_count] = smooth_weight
            row[index] = -2.0 * smooth_weight
            row[(index + 1) % frame_count] = smooth_weight
            rows.append(row)
            right_hand_side.append(0.0)

        regularization_weight = 0.20
        for index in range(frame_count):
            row = np.zeros(frame_count)
            row[index] = regularization_weight
            rows.append(row)
            right_hand_side.append(0.0)

        symmetry_weight = 50.0
        symmetry_sign = 1.0 if axis == 0 else -1.0
        for index in range(half):
            row = np.zeros(frame_count)
            row[index] = -symmetry_sign * symmetry_weight
            row[index + half] = symmetry_weight
            rows.append(row)
            right_hand_side.append(0.0)

        mean_row = np.full(frame_count, 1.0 / frame_count)
        rows.append(mean_row)
        right_hand_side.append(0.0)
        matrix = np.stack(rows)
        vector = np.asarray(right_hand_side)
        solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
        return solution

    return np.stack([solve_axis(0), solve_axis(1)], axis=1)


def _plant_stance_feet(
    qpos: np.ndarray,
    contact: np.ndarray,
    contract: PolicyContract,
    cycle_delta: np.ndarray,
    smoothing_passes: int,
) -> np.ndarray:
    result = np.asarray(qpos, dtype=np.float64).copy()
    positions = _foot_positions(result, contract)
    correction = _solve_stance_correction(
        positions, contact, cycle_delta, smoothing_passes
    )
    result[:, :2] += correction
    return result


def _max_consecutive_circular(values: np.ndarray) -> int:
    mask = np.asarray(values, dtype=bool)
    if mask.all():
        return int(mask.size)
    return max((len(run) for run in _circular_runs(mask)), default=0)


def _symmetry_metrics(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    joint_position: np.ndarray,
    contact: np.ndarray,
    cycle_delta: np.ndarray,
    joint_source: np.ndarray,
    joint_factor: np.ndarray,
) -> dict[str, float | int]:
    half = joint_position.shape[0] // 2
    joint_error = joint_position[half:] - (
        joint_position[:half, joint_source] * joint_factor
    )
    expected_root = root_position[:half].copy()
    expected_root[:, 1] *= -1.0
    expected_root += 0.5 * cycle_delta
    mirrored_quaternion = mirror_root_quaternion_xyzw(root_quaternion[:half])
    quaternion_dot = np.abs(
        np.sum(mirrored_quaternion * root_quaternion[half:], axis=1)
    )
    quaternion_angle = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
    contact_error = contact[half:] != contact[:half, ::-1]
    return {
        "joint_position_max_abs_rad": float(np.max(np.abs(joint_error))),
        "root_position_max_abs_m": float(
            np.max(np.abs(root_position[half:] - expected_root))
        ),
        "root_orientation_max_angle_rad": float(np.max(quaternion_angle)),
        "contact_mismatch_values": int(np.count_nonzero(contact_error)),
    }


def _stance_slip_metrics(
    foot_position: np.ndarray,
    contact: np.ndarray,
    dt: float,
    cycle_delta: np.ndarray,
) -> dict[str, Any]:
    stable = _long_contact_mask(contact)
    selected_speeds: list[float] = []
    for side in range(2):
        for index in range(contact.shape[0]):
            next_index = (index + 1) % contact.shape[0]
            if not (stable[index, side] and stable[next_index, side]):
                continue
            next_position = foot_position[next_index, side].copy()
            if index == contact.shape[0] - 1:
                next_position += cycle_delta
            velocity = (next_position - foot_position[index, side]) / dt
            selected_speeds.append(float(np.linalg.norm(velocity[:2])))
    selected = np.asarray(selected_speeds)
    return {
        "stable_contact_frames": stable.sum(axis=0).astype(int).tolist(),
        "mean_m_s": float(np.mean(selected)) if selected.size else float("inf"),
        "p90_m_s": (
            float(np.percentile(selected, 90)) if selected.size else float("inf")
        ),
        "max_m_s": float(np.max(selected)) if selected.size else float("inf"),
    }


def build_periodic_reference(
    source: Mapping[str, np.ndarray],
    contract: PolicyContract,
    *,
    frequency_hz: float = 50.0,
    source_half_weight: float = 0.8,
    smoothing_passes: int = 4,
    stance_correction_iterations: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project one even T1 sequence and return arrays plus an acceptance report."""
    if frequency_hz != 50.0:
        raise ValueError("periodic motion reference requires 50 Hz")
    required = {
        "root_position",
        "root_quaternion_xyzw",
        "joint_position",
        "foot_contact",
    }
    missing = required - set(source)
    if missing:
        raise ValueError(f"periodic source is missing arrays: {sorted(missing)}")
    joint_position = np.asarray(source["joint_position"], dtype=np.float64)
    frame_count = joint_position.shape[0]
    if frame_count < 26 or frame_count % 2:
        raise ValueError(
            "periodic projection requires an even sequence of at least 26 frames"
        )
    dt = 1.0 / frequency_hz
    joint_source, joint_factor = physical_mirror_map(contract.joint_order)

    mean_forward_speed = float(
        np.mean(np.asarray(source["root_linear_velocity"], dtype=np.float64)[:, 0])
    )
    cycle_delta = np.array([mean_forward_speed * frame_count * dt, 0.0, 0.0])
    root_position = _project_root_position(source["root_position"], cycle_delta)
    root_quaternion = _project_root_orientation(source["root_quaternion_xyzw"])
    joint_position = project_half_cycle(
        joint_position,
        joint_source,
        joint_factor,
        source_half_weight=source_half_weight,
    )
    joint_position = _smooth_circular(joint_position, smoothing_passes)
    limit_model = build_single_t1_soccer_model(
        prefix="periodic_limit_", robot_x=0.0, robot_y=0.0
    )
    joint_lower = np.array(
        [
            limit_model.joint("periodic_limit_" + name).range[0]
            for name in contract.joint_order
        ]
    )
    joint_upper = np.array(
        [
            limit_model.joint("periodic_limit_" + name).range[1]
            for name in contract.joint_order
        ]
    )
    joint_limit_violation = float(
        max(
            0.0,
            np.max(joint_lower - joint_position),
            np.max(joint_position - joint_upper),
        )
    )

    half = frame_count // 2
    source_contact = np.asarray(source["foot_contact"], dtype=bool)
    source_contact = np.concatenate(
        [source_contact[:half], source_contact[:half, ::-1]], axis=0
    )
    qpos = np.concatenate(
        [root_position, root_quaternion[:, [3, 0, 1, 2]], joint_position], axis=1
    )
    qpos, contact, replay = ground_reference_on_rcss(qpos, source_contact, contract)
    # Grounding is frame-local and can introduce millimetre-scale asymmetric
    # root-height offsets. Reproject before selecting the exact contact runs
    # used by the stance solver so the two half cycles constrain matching feet.
    qpos[:, :3] = _project_root_position(qpos[:, :3], cycle_delta)
    _, contact, replay = replay_rcss_surface(qpos, contract)
    for _ in range(stance_correction_iterations):
        qpos = _plant_stance_feet(
            qpos, contact, contract, cycle_delta, smoothing_passes=1
        )
        _, contact, replay = replay_rcss_surface(qpos, contract)

    qpos[:, :3] = _project_root_position(qpos[:, :3], cycle_delta)
    _, contact, replay = replay_rcss_surface(qpos, contract)

    root_position = qpos[:, :3]
    root_quaternion = _continuous_quaternions(qpos[:, [4, 5, 6, 3]])
    joint_position = qpos[:, 7:]
    root_linear_velocity = circular_gradient(root_position, dt, cycle_delta)
    root_angular_velocity = _circular_angular_velocity(root_quaternion, dt)
    joint_velocity = circular_gradient(joint_position, dt)
    foot_position = _foot_positions(qpos, contract)
    slip = _stance_slip_metrics(foot_position, contact, dt=dt, cycle_delta=cycle_delta)
    symmetry = _symmetry_metrics(
        root_position,
        root_quaternion,
        joint_position,
        contact,
        cycle_delta,
        joint_source,
        joint_factor,
    )
    yaw = Rotation.from_quat(root_quaternion).as_euler("zyx")[:, 0]
    yaw_center = float(np.arctan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw))))
    yaw_deviation = np.abs(
        np.arctan2(np.sin(yaw - yaw_center), np.cos(yaw - yaw_center))
    )
    wrap_joint_step = float(np.max(np.abs(joint_position[0] - joint_position[-1])))
    internal_joint_step = float(
        np.percentile(np.max(np.abs(np.diff(joint_position, axis=0)), axis=1), 95)
    )
    longest_flight = _max_consecutive_circular(~contact.any(axis=1))
    root_velocity_cyclic_step = float(
        np.max(np.abs(root_linear_velocity - np.roll(root_linear_velocity, 1, axis=0)))
    )
    gates = {
        "half_cycle_joint_symmetry": symmetry["joint_position_max_abs_rad"] <= 1.0e-5,
        "half_cycle_root_symmetry": symmetry["root_position_max_abs_m"] <= 1.0e-5,
        "half_cycle_orientation_symmetry": symmetry["root_orientation_max_angle_rad"]
        <= 1.0e-5,
        "half_cycle_contact_symmetry": symmetry["contact_mismatch_values"] == 0,
        "wrap_joint_step_lte_0_35": wrap_joint_step <= 0.35,
        "wrap_step_ratio_lte_2_5": wrap_joint_step
        <= max(1.0e-6, 2.5 * internal_joint_step),
        "root_velocity_wrap_lte_1_0": float(
            np.max(np.abs(root_linear_velocity[0] - root_linear_velocity[-1]))
        )
        <= 1.0,
        "root_velocity_cyclic_step_lte_1_5": root_velocity_cyclic_step <= 1.5,
        "joint_velocity_wrap_lte_5_0": float(
            np.max(np.abs(joint_velocity[0] - joint_velocity[-1]))
        )
        <= 5.0,
        "joint_limits_satisfied": joint_limit_violation <= 1.0e-6,
        "stance_slip_p90_lte_1_0": slip["p90_m_s"] <= 1.0,
        "flight_between_1_and_15_frames": 1 <= longest_flight <= 15,
        "both_feet_have_stable_contact": min(slip["stable_contact_frames"]) >= 2,
        "non_foot_pitch_contact_free": replay["non_foot_pitch_contact_frames"] == 0,
        "root_yaw_deviation_lte_0_15": float(np.max(yaw_deviation)) <= 0.15,
        "lateral_excursion_lte_0_15": float(np.max(np.abs(root_position[:, 1])))
        <= 0.15,
    }
    report = {
        "schema_version": 1,
        "projection": "bilateral_half_cycle_contact_aware_v1",
        "frame_count": frame_count,
        "frequency_hz": frequency_hz,
        "cycle_duration_seconds": frame_count * dt,
        "cycle_displacement_m": cycle_delta.tolist(),
        "commanded_average_forward_speed_m_s": mean_forward_speed,
        "source_half_weight": source_half_weight,
        "smoothing_passes": smoothing_passes,
        "stance_correction_iterations": stance_correction_iterations,
        "symmetry": symmetry,
        "continuity": {
            "wrap_joint_step_max_rad": wrap_joint_step,
            "internal_joint_step_p95_rad": internal_joint_step,
            "wrap_to_internal_p95_ratio": wrap_joint_step
            / max(1.0e-12, internal_joint_step),
            "root_velocity_wrap_max_m_s": float(
                np.max(np.abs(root_linear_velocity[0] - root_linear_velocity[-1]))
            ),
            "root_velocity_cyclic_step_max_m_s": root_velocity_cyclic_step,
            "joint_velocity_wrap_max_rad_s": float(
                np.max(np.abs(joint_velocity[0] - joint_velocity[-1]))
            ),
        },
        "stance_slip": slip,
        "joint_limit_max_violation_rad": joint_limit_violation,
        "contact_frames": contact.sum(axis=0).astype(int).tolist(),
        "longest_flight_frames": longest_flight,
        "root_yaw_center_rad": yaw_center,
        "root_yaw_deviation_max_rad": float(np.max(yaw_deviation)),
        "root_lateral_abs_max_m": float(np.max(np.abs(root_position[:, 1]))),
        "rcss_replay": replay,
        "gates": gates,
        "passed": all(gates.values()),
    }
    arrays = {
        "root_position": root_position.astype(np.float32),
        "root_quaternion_xyzw": root_quaternion.astype(np.float32),
        "root_linear_velocity": root_linear_velocity.astype(np.float32),
        "root_angular_velocity": root_angular_velocity.astype(np.float32),
        "joint_position": joint_position.astype(np.float32),
        "joint_velocity": joint_velocity.astype(np.float32),
        "foot_contact": contact.astype(bool),
    }
    return arrays, report
