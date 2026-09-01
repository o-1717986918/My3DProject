"""Exact-CPU wrapper for Apollo's retained 78-to-23 walk policy."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .contract import PolicyContract
from .t1_control import APOLLO_DEFAULT_POSE


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_APOLLO_WALK_POLICY = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)


def apollo_walk_observation(
    *,
    angular_velocity: np.ndarray,
    projected_gravity: np.ndarray,
    velocity_command: np.ndarray,
    joint_position_offset: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Encode the finite, clipped Apollo walk actor input."""

    fields = [
        np.asarray(angular_velocity, dtype=np.float64),
        np.asarray(projected_gravity, dtype=np.float64),
        np.asarray(velocity_command, dtype=np.float64),
        np.asarray(joint_position_offset, dtype=np.float64),
        np.asarray(joint_velocity, dtype=np.float64),
        np.asarray(previous_action, dtype=np.float64),
    ]
    expected = ((3,), (3,), (3,), (23,), (23,), (23,))
    if tuple(field.shape for field in fields) != expected:
        raise ValueError("Apollo walk observation fields have incompatible shapes")
    observation = np.concatenate(fields)
    if not np.isfinite(observation).all():
        raise ValueError("Apollo walk observation contains non-finite values")
    return np.clip(observation, -10.0, 10.0).astype(np.float32)


class ApolloWalkCpu:
    """Run Apollo's deployed walk actor against an exact MuJoCo state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        contract: PolicyContract,
        *,
        prefix: str,
        policy_path: Path = DEFAULT_APOLLO_WALK_POLICY,
    ) -> None:
        if contract.action_size != 23:
            raise ValueError("Apollo walk policy requires 23 robot actions")
        if not policy_path.is_file():
            raise FileNotFoundError(f"Apollo walk policy not found: {policy_path}")
        self.policy_path = policy_path
        self._joint_qpos = np.asarray(
            [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
        )
        self._joint_dof = np.asarray(
            [model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
        )
        self._torso_site = model.site(prefix + "torso").id
        gyro = model.sensor(prefix + "torso_gyro")
        self._gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
        self._session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        actor_input = self._session.get_inputs()[0]
        actor_output = self._session.get_outputs()[0]
        if actor_input.shape != [1, 78] or actor_output.shape != [1, 23]:
            raise ValueError("Apollo walk ONNX contract must be [1,78] -> [1,23]")
        self._input_name = actor_input.name

    def target(
        self,
        data: mujoco.MjData,
        previous_action: np.ndarray,
        velocity_command: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the deployed joint target and recurrent action-history value."""

        torso_rotation = data.site_xmat[self._torso_site].reshape(3, 3)
        gravity = torso_rotation.T @ np.array([0.0, 0.0, -1.0])
        observation = apollo_walk_observation(
            angular_velocity=data.sensordata[self._gyro_slice],
            projected_gravity=gravity,
            velocity_command=velocity_command,
            joint_position_offset=(
                data.qpos[self._joint_qpos] - APOLLO_DEFAULT_POSE
            ),
            joint_velocity=data.qvel[self._joint_dof],
            previous_action=previous_action,
        )
        action = self._session.run(
            None, {self._input_name: observation[None, :]}
        )[0][0].astype(np.float64)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError("Apollo walk policy returned an invalid action")
        action = np.clip(action, -5.0, 5.0)
        return APOLLO_DEFAULT_POSE + 0.25 * action, action
