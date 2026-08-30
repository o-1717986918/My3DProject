from collections import deque
from enum import Enum, auto
import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from mujococodebase.skills.external.apollo_get_up import ApolloGetUpPolicy
from mujococodebase.skills.keyframe.keyframe import KeyframeSkill
from mujococodebase.skills.skill import Skill

logger = logging.getLogger(__name__)


class FallPose(Enum):
    UPRIGHT = auto()
    FRONT = auto()
    BACK = auto()
    LEFT = auto()
    RIGHT = auto()


class GetUpPhase(Enum):
    SETTLE = auto()
    MOTION = auto()
    VERIFY = auto()


def classify_fall_pose(quaternion_xyzw: np.ndarray) -> FallPose:
    """Classify the torso pose from gravity expressed in body coordinates."""
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return FallPose.FRONT

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        return FallPose.FRONT

    gravity_body = R.from_quat(quaternion / norm).inv().apply([0.0, 0.0, -1.0])
    if gravity_body[2] < -0.65:
        return FallPose.UPRIGHT
    if abs(gravity_body[0]) >= abs(gravity_body[1]):
        # The T1 camera/chest points along local +x. If gravity points along
        # +x in the torso frame, the chest faces the floor (front fall).
        return FallPose.FRONT if gravity_body[0] > 0.0 else FallPose.BACK
    return FallPose.LEFT if gravity_body[1] > 0.0 else FallPose.RIGHT


class GetUp(Skill):
    """Verified keyframe recovery with posture classification and retries."""

    STABILITY_THRESHOLD_CYCLES = 10
    MAX_SETTLED_GYRO_DEG_S = 4.0
    MAX_SETTLED_JOINT_SPEED_DEG_S = 35.0
    EARLY_UPRIGHT_SECONDS = 0.12
    STAND_HOLD_SECONDS = 0.60
    SUPPORT_TRANSITION_SECONDS = 0.80
    SUPPORT_SETTLE_SECONDS = 0.40
    STAND_TRANSITION_SECONDS = 1.20
    VERIFY_UPRIGHT_SECONDS = 0.60
    VERIFY_TIMEOUT_SECONDS = 5.2

    def __init__(self, agent):
        super().__init__(agent)
        self.apollo_policy = ApolloGetUpPolicy(agent)
        self.get_up_front = KeyframeSkill(
            agent=agent,
            file=os.path.join(os.path.dirname(__file__), "get_up_front.yaml"),
        )
        self.get_up_back = KeyframeSkill(
            agent=agent,
            file=os.path.join(os.path.dirname(__file__), "get_up_back.yaml"),
        )

    def execute(self, reset, *args, **kwargs):
        if self.apollo_policy.available:
            finished = self.apollo_policy.execute(reset, *args, **kwargs)
            if self.apollo_policy.available:
                return finished
            # Inference errors disable the optional adapter. Initialize the
            # built-in recovery immediately so the robot is never left with
            # stale joint targets or an uninitialized fallback state.
            self._reset_recovery()

        elif reset:
            self._reset_recovery()

        if self.phase is GetUpPhase.SETTLE:
            self._step_settle()
        elif self.phase is GetUpPhase.MOTION:
            self._step_motion()
        else:
            return self._step_verify()
        return False

    def _reset_recovery(self) -> None:
        self.phase = GetUpPhase.SETTLE
        self.gyro_queue = deque(maxlen=self.STABILITY_THRESHOLD_CYCLES)
        self.attempt = 0
        self.motion = None
        self.motion_reset_pending = True
        self.verify_started_at = None
        self.verify_start_positions = None
        self.upright_since = None
        self.motion_upright_since = None
        self.verify_cycles = 0

    def _step_settle(self) -> None:
        robot = self.agent.robot
        # Neutral is a zero-duration one-frame keyframe, so every call must
        # reset it before applying the pose again.
        self.agent.skills_manager.execute_sub_skill("Neutral", reset=True)

        angular_speed = float(np.max(np.abs(robot.gyroscope)))
        joint_speed = float(np.max(np.abs(tuple(robot.motor_speeds.values()))))
        self.gyro_queue.append((angular_speed, joint_speed))
        settled = len(self.gyro_queue) == self.STABILITY_THRESHOLD_CYCLES and all(
            gyro < self.MAX_SETTLED_GYRO_DEG_S
            and motor < self.MAX_SETTLED_JOINT_SPEED_DEG_S
            for gyro, motor in self.gyro_queue
        )
        if not settled:
            return

        pose = classify_fall_pose(robot.global_orientation_quat)
        self.motion = self._select_motion(pose)
        self.motion_reset_pending = True
        self.motion_upright_since = None
        self.phase = GetUpPhase.MOTION
        logger.info(
            "Starting get-up attempt %d from %s using %s",
            self.attempt + 1,
            pose.name,
            "back" if self.motion is self.get_up_back else "front",
        )

    def _select_motion(self, pose: FallPose) -> KeyframeSkill:
        # Front/back routines are deliberately alternated after failure. This
        # gives side falls and ambiguous collision poses a different escape path
        # instead of replaying the same unsuccessful motion forever.
        primary = self.get_up_back if pose is FallPose.BACK else self.get_up_front
        if self.attempt % 2 == 0:
            return primary
        return self.get_up_front if primary is self.get_up_back else self.get_up_back

    def _step_motion(self) -> None:
        if self.motion is None:
            self.phase = GetUpPhase.SETTLE
            return

        finished = self.motion.execute(reset=self.motion_reset_pending)
        self.motion_reset_pending = False
        world = self.agent.world
        now = world.server_time

        # Both legacy keyframes pass through a useful upright crouch before
        # their final open-loop frames. Hand control to the learned standing
        # policy as soon as that pose is stable, before momentum tips it over.
        angular_speed = float(np.max(np.abs(self.agent.robot.gyroscope)))
        roll, pitch, _ = self.agent.robot.global_orientation_euler
        candidate_is_stable = (
            world.is_upright(min_height=0.38, min_up_component=0.80)
            and abs(roll) < 10.0
            and abs(pitch) < 30.0
            and angular_speed < 120.0
        )
        if candidate_is_stable:
            if self.motion_upright_since is None:
                self.motion_upright_since = now
            elif now - self.motion_upright_since >= self.EARLY_UPRIGHT_SECONDS:
                self._enter_verify()
                return
        else:
            self.motion_upright_since = None

        if finished:
            if candidate_is_stable:
                self._enter_verify()
            else:
                self._schedule_retry("motion ended without a stable support pose")

    def _enter_verify(self) -> None:
        self.phase = GetUpPhase.VERIFY
        self.verify_started_at = self.agent.world.server_time
        self.verify_start_positions = dict(self.agent.robot.motor_positions)
        self.upright_since = None
        self.verify_cycles = 0
        robot = self.agent.robot
        logger.info(
            "Entering get-up stabilization z=%.2f up=%.2f "
            "legs=(%.1f %.1f %.1f %.1f %.1f %.1f | "
            "%.1f %.1f %.1f %.1f %.1f %.1f)",
            self.agent.world.global_position[2],
            self.agent.world.torso_up_component(),
            robot.motor_positions["lle1"],
            robot.motor_positions["lle2"],
            robot.motor_positions["lle3"],
            robot.motor_positions["lle4"],
            robot.motor_positions["lle5"],
            robot.motor_positions["lle6"],
            robot.motor_positions["rle1"],
            robot.motor_positions["rle2"],
            robot.motor_positions["rle3"],
            robot.motor_positions["rle4"],
            robot.motor_positions["rle5"],
            robot.motor_positions["rle6"],
        )

    def _step_verify(self) -> bool:
        world = self.agent.world
        now = world.server_time
        elapsed = now - self.verify_started_at
        self.verify_cycles += 1

        if self.verify_cycles % 10 == 0:
            logger.debug(
                "Get-up stabilization t=%.2f rpy=(%.1f %.1f %.1f) "
                "gyro=(%.1f %.1f %.1f)",
                elapsed,
                *self.agent.robot.global_orientation_euler,
                *self.agent.robot.gyroscope,
            )

        transition_end = (
            self.STAND_HOLD_SECONDS
            + self.SUPPORT_TRANSITION_SECONDS
            + self.SUPPORT_SETTLE_SECONDS
            + self.STAND_TRANSITION_SECONDS
        )
        self._command_standing_transition(elapsed)

        angular_speed = float(np.max(np.abs(self.agent.robot.gyroscope)))
        stable_upright = (
            elapsed >= transition_end and world.is_upright() and angular_speed < 12.0
        )
        if stable_upright:
            if self.upright_since is None:
                self.upright_since = now
            elif now - self.upright_since >= self.VERIFY_UPRIGHT_SECONDS:
                logger.info("Get-up verified after %d attempt(s)", self.attempt + 1)
                return True
        else:
            self.upright_since = None

        if elapsed >= self.VERIFY_TIMEOUT_SECONDS:
            self._schedule_retry("failed verification")
        return False

    def _schedule_retry(self, reason: str) -> None:
        self.attempt += 1
        self.phase = GetUpPhase.SETTLE
        self.gyro_queue.clear()
        self.motion = None
        logger.warning("Get-up attempt %s; retrying", reason)

    def _command_standing_transition(self, elapsed: float) -> None:
        """Blend from the recovered crouch into the walk policy's home pose."""
        walk = self.agent.skills_manager.get_skill_object("Walk")
        target_positions = np.rad2deg(walk.joint_nominal_position * walk.train_sim_flip)
        support_elapsed = max(0.0, elapsed - self.STAND_HOLD_SECONDS)
        support_progress = float(
            np.clip(support_elapsed / self.SUPPORT_TRANSITION_SECONDS, 0.0, 1.0)
        )
        smooth_support = (
            support_progress * support_progress * (3.0 - 2.0 * support_progress)
        )

        stand_delay = (
            self.STAND_HOLD_SECONDS
            + self.SUPPORT_TRANSITION_SECONDS
            + self.SUPPORT_SETTLE_SECONDS
        )
        stand_elapsed = max(0.0, elapsed - stand_delay)
        stand_progress = float(
            np.clip(stand_elapsed / self.STAND_TRANSITION_SECONDS, 0.0, 1.0)
        )
        smooth_stand = stand_progress * stand_progress * (3.0 - 2.0 * stand_progress)

        roll, _, _ = self.agent.robot.global_orientation_euler
        gyro_roll, _, _ = self.agent.robot.gyroscope
        roll_correction = float(np.clip(0.35 * roll + 0.025 * gyro_roll, -15.0, 15.0))

        support_overrides = {
            "lle1": -55.0,
            "lle4": 95.0,
            "rle1": -55.0,
            "rle4": 95.0,
        }

        for motor_name, target_position in zip(
            self.agent.robot.ROBOT_MOTORS, target_positions, strict=True
        ):
            start_position = self.verify_start_positions.get(motor_name, 0.0)
            support_position = support_overrides.get(motor_name, start_position)
            support_blend = start_position + smooth_support * (
                support_position - start_position
            )
            blended_position = support_blend + smooth_stand * (
                target_position - support_blend
            )
            if motor_name in {"lle6", "rle6"}:
                blended_position += roll_correction
            self.agent.robot.set_motor_target_position(
                motor_name, blended_position, kp=75.0, kd=4.5
            )

    def is_ready(self, *args):
        return self.agent.world.is_fallen()
