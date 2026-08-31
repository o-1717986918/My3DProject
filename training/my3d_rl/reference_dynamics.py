"""Dynamics diagnostics for kinematic humanoid motion references.

This module deliberately stays outside the deployed policy path.  It turns a
complete reference state into the joint targets that the competition PD
actuators would need according to MuJoCo inverse dynamics, then exposes small
pure helpers used by the diagnostic command and regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .contract import PolicyContract


@dataclass(frozen=True)
class InverseDynamicsReference:
    """Per-frame inverse-dynamics result in physical joint coordinates."""

    joint_target_position: np.ndarray
    joint_torque: np.ndarray
    joint_target_residual: np.ndarray
    root_generalized_force: np.ndarray
    qacc: np.ndarray


def circular_smooth(values: np.ndarray, passes: int) -> np.ndarray:
    """Apply a short zero-phase circular filter without moving the cycle seam."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 2:
        raise ValueError("circular smoothing expects a two-dimensional array")
    if passes < 0:
        raise ValueError("smoothing passes must be non-negative")
    for _ in range(passes):
        result = (
            0.25 * np.roll(result, 1, axis=0)
            + 0.50 * result
            + 0.25 * np.roll(result, -1, axis=0)
        )
    return result


def circular_interpolate(values: np.ndarray, phase: float) -> np.ndarray:
    """Linearly interpolate one cyclic frame array at a normalized phase."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("cyclic interpolation expects at least two frames")
    frame = (float(phase) % 1.0) * values.shape[0]
    lower = int(np.floor(frame)) % values.shape[0]
    upper = (lower + 1) % values.shape[0]
    fraction = frame - np.floor(frame)
    return (1.0 - fraction) * values[lower] + fraction * values[upper]


def failure_phase_sampling_weights(
    failure_phases: np.ndarray,
    *,
    bin_count: int,
    kernel_size: int = 3,
    kernel_decay: float = 0.8,
    uniform_ratio: float = 0.1,
) -> np.ndarray:
    """Build cyclic failure-focused reset weights in the BeyondMimic pattern.

    A non-causal decaying kernel places probability on the frames immediately
    preceding a recorded failure.  A uniform mixture keeps every phase
    reachable and makes the resulting categorical distribution finite.
    """
    phases = np.asarray(failure_phases, dtype=np.float64).reshape(-1)
    if bin_count < 2 or kernel_size < 1:
        raise ValueError("phase sampling requires at least two bins and one kernel tap")
    if not 0.0 < kernel_decay <= 1.0:
        raise ValueError("kernel decay must lie in (0, 1]")
    if not 0.0 < uniform_ratio <= 1.0:
        raise ValueError("uniform ratio must lie in (0, 1]")
    if phases.size == 0 or not np.isfinite(phases).all():
        raise ValueError("failure phases must be a non-empty finite array")
    bins = np.floor(np.mod(phases, 1.0) * bin_count).astype(np.int64)
    counts = np.bincount(bins, minlength=bin_count).astype(np.float64)
    kernel = kernel_decay ** np.arange(kernel_size, dtype=np.float64)
    kernel /= np.sum(kernel)
    focused = sum(
        weight * np.roll(counts, -offset) for offset, weight in enumerate(kernel)
    )
    focused /= np.sum(focused)
    weights = (1.0 - uniform_ratio) * focused + uniform_ratio / bin_count
    return weights / np.sum(weights)


def _joint_addresses(
    model: mujoco.MjModel, contract: PolicyContract, prefix: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order],
        dtype=np.int32,
    )
    joint_dof = np.array(
        [model.joint(prefix + name).dofadr[0] for name in contract.joint_order],
        dtype=np.int32,
    )
    root = model.joint(prefix + "root")
    return joint_qpos, joint_dof, int(root.qposadr[0]), int(root.dofadr[0])


def configure_pd_actuators(
    model: mujoco.MjModel,
    contract: PolicyContract,
    *,
    kp: float,
    kd: float,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Configure the exact position-plus-velocity actuator protocol."""
    tau = np.array(
        [model.actuator(prefix + name + "_tau").id for name in contract.effector_order]
    )
    pos = np.array(
        [model.actuator(prefix + name + "_pos").id for name in contract.effector_order]
    )
    vel = np.array(
        [model.actuator(prefix + name + "_vel").id for name in contract.effector_order]
    )
    model.actuator_gainprm[pos, 0] = kp
    model.actuator_biasprm[pos, 1] = -kp
    model.actuator_gainprm[vel, 0] = kd
    model.actuator_biasprm[vel, 2] = -kd
    return tau, pos, vel


def compute_inverse_dynamics_reference(
    model: mujoco.MjModel,
    contract: PolicyContract,
    *,
    root_position: np.ndarray,
    root_quaternion_xyzw: np.ndarray,
    root_linear_velocity: np.ndarray,
    root_angular_velocity: np.ndarray,
    joint_position_physical: np.ndarray,
    joint_velocity_physical: np.ndarray,
    frequency_hz: float,
    kp: float,
    kd: float,
    prefix: str = "train_",
    smoothing_passes: int = 2,
    maximum_residual_rad: float = 0.15,
) -> InverseDynamicsReference:
    """Estimate dynamically compensated PD targets for one periodic reference.

    The inverse problem can require non-zero generalized forces at the floating
    base.  Those forces are returned as a feasibility diagnostic; they are not
    injected into simulation or made available to a learned controller.
    """
    arrays = [
        np.asarray(root_position, dtype=np.float64),
        np.asarray(root_quaternion_xyzw, dtype=np.float64),
        np.asarray(root_linear_velocity, dtype=np.float64),
        np.asarray(root_angular_velocity, dtype=np.float64),
        np.asarray(joint_position_physical, dtype=np.float64),
        np.asarray(joint_velocity_physical, dtype=np.float64),
    ]
    frame_count = arrays[4].shape[0]
    expected = [(frame_count, 3), (frame_count, 4), (frame_count, 3), (frame_count, 3)]
    for value, shape in zip(arrays[:4], expected):
        if value.shape != shape:
            raise ValueError(f"reference array shape {value.shape} != {shape}")
    if arrays[4].shape != (frame_count, contract.action_size):
        raise ValueError("joint position reference has incompatible shape")
    if arrays[5].shape != arrays[4].shape:
        raise ValueError("joint velocity reference has incompatible shape")
    if frame_count < 3 or frequency_hz <= 0.0 or kp <= 0.0 or kd < 0.0:
        raise ValueError("invalid dynamics reference parameters")
    if maximum_residual_rad <= 0.0:
        raise ValueError("maximum residual must be positive")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("reference arrays must be finite")

    joint_qpos, joint_dof, root_qpos, root_dof = _joint_addresses(
        model, contract, prefix
    )
    configure_pd_actuators(model, contract, kp=kp, kd=kd, prefix=prefix)
    joint_position = arrays[4]
    joint_velocity = arrays[5]
    qvel = np.zeros((frame_count, model.nv), dtype=np.float64)
    qvel[:, root_dof : root_dof + 3] = arrays[2]
    qvel[:, root_dof + 3 : root_dof + 6] = arrays[3]
    qvel[:, joint_dof] = joint_velocity
    dt = 1.0 / frequency_hz
    qacc = (np.roll(qvel, -1, axis=0) - np.roll(qvel, 1, axis=0)) / (2.0 * dt)

    data = mujoco.MjData(model)
    joint_torque = np.empty_like(joint_position)
    root_generalized_force = np.empty((frame_count, 6), dtype=np.float64)
    root_xy_origin = model.qpos0[root_qpos : root_qpos + 2].copy()
    reference_xy_origin = arrays[0][0, :2].copy()

    for index in range(frame_count):
        data.qpos[:] = model.qpos0
        data.qvel[:] = qvel[index]
        data.qpos[root_qpos : root_qpos + 2] = (
            root_xy_origin + arrays[0][index, :2] - reference_xy_origin
        )
        data.qpos[root_qpos + 2] = arrays[0][index, 2]
        data.qpos[root_qpos + 3 : root_qpos + 7] = arrays[1][index, [3, 0, 1, 2]]
        data.qpos[joint_qpos] = joint_position[index]
        mujoco.mj_forward(model, data)
        data.qacc[:] = qacc[index]
        mujoco.mj_inverse(model, data)
        joint_torque[index] = data.qfrc_inverse[joint_dof]
        root_generalized_force[index] = data.qfrc_inverse[root_dof : root_dof + 6]

    # The server sends zero to the velocity actuators.  Their -kd*qdot force
    # therefore has to be cancelled by a corresponding position error.
    raw_residual = (joint_torque + kd * joint_velocity) / kp
    residual = circular_smooth(raw_residual, smoothing_passes)
    residual = np.clip(residual, -maximum_residual_rad, maximum_residual_rad)
    lower = model.jnt_range[model.dof_jntid[joint_dof], 0]
    upper = model.jnt_range[model.dof_jntid[joint_dof], 1]
    targets = np.clip(joint_position + residual, lower, upper)
    residual = targets - joint_position
    return InverseDynamicsReference(
        joint_target_position=targets,
        joint_torque=joint_torque,
        joint_target_residual=residual,
        root_generalized_force=root_generalized_force,
        qacc=qacc,
    )
