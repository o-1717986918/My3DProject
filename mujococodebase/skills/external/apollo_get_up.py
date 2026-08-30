"""Runtime adapter for ApolloCodebase's separately licensed get-up policy.

The ONNX asset remains in the ApolloCodebase git submodule and is licensed
under GPL-3.0-or-later by its authors. Deployments that enable this adapter
must include that license and comply with its terms.
"""

import logging
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from mujococodebase.skills.skill import Skill
from mujococodebase.utils.neural_network import load_network, run_network

logger = logging.getLogger(__name__)


class ApolloGetUpPolicy(Skill):
    """Execute Apollo's relative-joint recovery network when it is installed."""

    ACTION_SCALE = 0.6
    ACTION_LIMIT = 5.0
    RESET_AFTER_SECONDS = 6.0
    VERIFY_SECONDS = 0.35

    DEFAULT_POSITIONS_RAD = np.array(
        [
            0.0,
            0.0,
            0.0,
            -1.4,
            0.0,
            -0.4,
            0.0,
            1.4,
            0.0,
            0.4,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.2,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.2,
            0.0,
        ],
        dtype=float,
    )

    KP = np.array(
        [
            10,
            20,
            45,
            45,
            30,
            30,
            45,
            45,
            30,
            30,
            85,
            130,
            90,
            70,
            140,
            45,
            40,
            130,
            90,
            70,
            140,
            45,
            40,
        ],
        dtype=float,
    )
    KD = np.array(
        [
            1,
            1,
            2.5,
            2.5,
            1.2,
            1.2,
            2.5,
            2.5,
            1.2,
            1.2,
            5,
            10,
            8,
            3,
            6,
            2,
            1.8,
            10,
            8,
            3,
            6,
            2,
            1.8,
        ],
        dtype=float,
    )

    def __init__(self, agent):
        super().__init__(agent)
        self.model_path = self._resolve_model_path()
        self.model = None
        self.available = False
        self.previous_action = np.zeros(23, dtype=float)
        self.started_at = 0.0
        self.upright_since = None

        if os.environ.get("MY3D_GETUP_BACKEND", "apollo").lower() == "keyframe":
            logger.info("Apollo get-up policy disabled by MY3D_GETUP_BACKEND")
            return
        if not self.model_path.is_file():
            logger.warning(
                "Apollo get-up model not found at %s; using keyframe fallback",
                self.model_path,
            )
            return

        try:
            self.model = load_network(str(self.model_path))
        except Exception:
            logger.exception("Could not load Apollo get-up model; using fallback")
            return

        self.available = True
        logger.info("Apollo get-up policy enabled from %s", self.model_path)

    @staticmethod
    def _resolve_model_path() -> Path:
        configured = os.environ.get("MY3D_APOLLO_GETUP_MODEL")
        if configured:
            return Path(configured).expanduser().resolve()
        repository_root = Path(__file__).resolve().parents[3]
        return (
            repository_root
            / "external"
            / "ApolloCodebase"
            / "assets"
            / "networks"
            / "getup"
            / "policy.onnx"
        )

    def execute(self, reset, *args, **kwargs) -> bool:
        if not self.available:
            return False
        if reset:
            self._reset_policy()

        now = self.agent.world.server_time
        if now - self.started_at >= self.RESET_AFTER_SECONDS:
            logger.warning(
                "[%s #%d] Apollo get-up timed out; resetting policy history",
                self.agent.world.team_name,
                self.agent.world.number,
            )
            self._reset_policy()

        positions_deg, speeds_deg_s = self.agent.robot.get_ordered_motor_state()
        positions_rad = np.deg2rad(positions_deg)
        speeds_rad_s = np.deg2rad(speeds_deg_s)
        projected_gravity = (
            R.from_quat(self.agent.robot.global_orientation_quat)
            .inv()
            .apply([0.0, 0.0, -1.0])
        )
        gyro_rad_s = np.deg2rad(self.agent.robot.gyroscope)

        observation = np.concatenate(
            [
                gyro_rad_s,
                projected_gravity,
                positions_rad - self.DEFAULT_POSITIONS_RAD,
                speeds_rad_s,
                self.previous_action,
            ]
        )
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)

        try:
            action = run_network(observation, self.model)
        except Exception:
            logger.exception(
                "Apollo get-up inference failed; disabling adapter for this process"
            )
            self.available = False
            return False
        if action.shape != (23,) or not np.all(np.isfinite(action)):
            logger.error("Apollo get-up produced an invalid action; disabling adapter")
            self.available = False
            return False
        action = np.clip(action, -self.ACTION_LIMIT, self.ACTION_LIMIT)
        targets_deg = np.rad2deg(positions_rad + action * self.ACTION_SCALE)
        self.previous_action = action

        for index, motor_name in enumerate(self.agent.robot.ROBOT_MOTORS):
            self.agent.robot.set_motor_target_position(
                motor_name,
                targets_deg[index],
                kp=self.KP[index],
                kd=self.KD[index],
            )

        angular_speed = float(np.max(np.abs(self.agent.robot.gyroscope)))
        stable_upright = self.agent.world.is_upright() and angular_speed < 25.0
        if stable_upright:
            if self.upright_since is None:
                self.upright_since = now
            elif now - self.upright_since >= self.VERIFY_SECONDS:
                logger.info(
                    "[%s #%d] Apollo get-up verified in %.2fs",
                    self.agent.world.team_name,
                    self.agent.world.number,
                    now - self.started_at,
                )
                return True
        else:
            self.upright_since = None
        return False

    def _reset_policy(self) -> None:
        self.previous_action.fill(0.0)
        self.started_at = self.agent.world.server_time
        self.upright_since = None

    def is_ready(self, *args, **kwargs) -> bool:
        return self.available and self.agent.world.is_fallen()
