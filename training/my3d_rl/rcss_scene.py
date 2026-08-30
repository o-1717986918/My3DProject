"""Direct-MuJoCo parity scene assembled from installed RCSSServerMJ assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .contract import PolicyContract


DEFAULT_RESOURCE_ROOT = Path(
    "/home/win98/.local/pipx/venvs/rcsssmj/lib/python3.10/site-packages/rcsssmj/resources"
)


def build_single_t1_soccer_model(
    resource_root: Path = DEFAULT_RESOURCE_ROOT,
    *,
    prefix: str = "train_",
    robot_x: float = -0.32,
    robot_y: float = 0.0,
) -> mujoco.MjModel:
    """Compile the exact soccer world plus one prefixed T1 robot.

    This follows RCSSServerMJ's own MjSpec attachment path. It is the CPU
    reference used before converting an environment to MJX.
    """

    world_path = resource_root / "environments" / "soccer" / "world.xml"
    robot_path = resource_root / "robots" / "T1" / "robot.xml"
    if not world_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError(
            f"RCSSServerMJ soccer/T1 resources not found below {resource_root}"
        )

    world = mujoco.MjSpec.from_file(str(world_path))
    robot = mujoco.MjSpec.from_file(str(robot_path))
    torso = robot.body("torso")
    torso.pos[0] = robot_x
    torso.pos[1] = robot_y
    world.worldbody.add_frame().attach_body(torso, prefix, "")
    return world.compile()


@dataclass(frozen=True)
class JointState:
    position: np.ndarray
    velocity: np.ndarray


class RcssKickScene:
    """Small deterministic controller surface matching the server PD protocol."""

    def __init__(
        self,
        contract: PolicyContract,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        *,
        prefix: str = "train_",
    ) -> None:
        self.contract = contract
        self.prefix = prefix
        self.model = build_single_t1_soccer_model(resource_root, prefix=prefix)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = round((1.0 / contract.frequency_hz) / self.model.opt.timestep)
        if self.n_substeps * self.model.opt.timestep != 1.0 / contract.frequency_hz:
            raise ValueError(
                "physics timestep does not divide the 50 Hz control period"
            )

        self._joint_qpos = np.array(
            [
                self.model.joint(prefix + name).qposadr[0]
                for name in contract.joint_order
            ],
            dtype=np.int32,
        )
        self._joint_dof = np.array(
            [
                self.model.joint(prefix + name).dofadr[0]
                for name in contract.joint_order
            ],
            dtype=np.int32,
        )
        self._tau_actuator = self._actuator_ids("_tau")
        self._pos_actuator = self._actuator_ids("_pos")
        self._vel_actuator = self._actuator_ids("_vel")
        mujoco.mj_forward(self.model, self.data)

    def _actuator_ids(self, suffix: str) -> np.ndarray:
        return np.array(
            [
                self.model.actuator(self.prefix + effector + suffix).id
                for effector in self.contract.effector_order
            ],
            dtype=np.int32,
        )

    def joint_state(self) -> JointState:
        return JointState(
            position=self.data.qpos[self._joint_qpos].copy(),
            velocity=self.data.qvel[self._joint_dof].copy(),
        )

    def step_joint_targets(
        self,
        targets_rad: np.ndarray,
        *,
        kp: float | np.ndarray,
        kd: float | np.ndarray,
    ) -> JointState:
        targets = np.asarray(targets_rad, dtype=np.float64)
        gains_p = np.broadcast_to(np.asarray(kp, dtype=np.float64), targets.shape)
        gains_d = np.broadcast_to(np.asarray(kd, dtype=np.float64), targets.shape)
        if targets.shape != (self.contract.action_size,):
            raise ValueError(
                f"expected {(self.contract.action_size,)}, got {targets.shape}"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError("joint targets must be finite")
        if np.any(gains_p < 0.0) or np.any(gains_d < 0.0):
            raise ValueError("PD gains must be non-negative")

        self.data.ctrl[self._tau_actuator] = 0.0
        self.data.ctrl[self._pos_actuator] = targets
        self.data.ctrl[self._vel_actuator] = 0.0
        self.model.actuator_gainprm[self._pos_actuator, 0] = gains_p
        self.model.actuator_biasprm[self._pos_actuator, 1] = -gains_p
        self.model.actuator_gainprm[self._vel_actuator, 0] = gains_d
        self.model.actuator_biasprm[self._vel_actuator, 2] = -gains_d

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.joint_state()
