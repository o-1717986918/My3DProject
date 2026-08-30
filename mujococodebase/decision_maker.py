from enum import Enum, auto
import logging
from typing import Mapping

import numpy as np

from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import Field, FIFAField, HLAdultField, MyField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum


logger = logging.getLogger(__name__)


class AttackPhase(Enum):
    SEARCH = auto()
    APPROACH = auto()
    ALIGN = auto()
    KICK = auto()
    RECOVER = auto()


class DecisionMaker:
    """Competition controller using one canonical frame for both team sides."""

    # Poses are expressed in the canonical team frame: our goal is on negative x.
    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] = {
        FIFAField: {
            1: (-50.0, 0.0, 0.0),
            2: (-35.0, -12.0, 0.0),
            3: (-35.0, 0.0, 0.0),
            4: (-35.0, 12.0, 0.0),
            5: (-22.0, -16.0, 0.0),
            6: (-22.0, -6.0, 0.0),
            7: (-22.0, 6.0, 0.0),
            8: (-22.0, 16.0, 0.0),
            9: (-11.0, -8.0, 0.0),
            10: (-11.0, 8.0, 0.0),
            11: (-10.0, 0.0, 0.0),
        },
        HLAdultField: {
            1: (-6.5, 0.0, 0.0),
            2: (-2.0, -1.5, 0.0),
            3: (-2.0, 1.5, 0.0),
        },
        MyField: {
            1: (-26.0, 0.0, 0.0),
            2: (-15.0, -5.0, 0.0),
            3: (-15.0, 5.0, 0.0),
            4: (-12.0, 0.0, 0.0),
            5: (-10.0, -7.0, 0.0),
            6: (-10.0, 7.0, 0.0),
            7: (-9.5, -2.0, 0.0),
        },
    }

    # Kickoff-specific 7v7 poses. Only the active kicker enters the centre
    # circle; the passive side remains in its own half and outside the 5.5 m
    # exclusion radius. This avoids spending the opening 20+ seconds walking a
    # midfielder from the generic formation to a stationary centre ball.
    MY_FIELD_ACTIVE_KICKOFF_POSES: Mapping[int, tuple[float, float, float]] = {
        1: (-26.0, 0.0, 0.0),
        2: (-15.0, -5.0, 0.0),
        3: (-15.0, 5.0, 0.0),
        4: (-8.0, 0.0, 0.0),
        5: (-7.0, -6.0, 0.0),
        6: (-7.0, 6.0, 0.0),
        7: (-0.9, -0.4, 0.0),
    }
    MY_FIELD_PASSIVE_KICKOFF_POSES: Mapping[int, tuple[float, float, float]] = {
        1: (-26.0, 0.0, 0.0),
        2: (-15.0, -5.0, 0.0),
        3: (-15.0, 5.0, 0.0),
        4: (-8.0, 0.0, 0.0),
        5: (-7.0, -6.0, 0.0),
        6: (-7.0, 6.0, 0.0),
        7: (-6.0, 0.0, 0.0),
    }

    BALL_FRESHNESS_SECONDS = 0.75
    KICK_COOLDOWN_SECONDS = 0.8
    POST_GETUP_HOLD_SECONDS = 0.6

    def __init__(self, agent):
        from mujococodebase.agent import Agent

        self.agent: Agent = agent
        self.is_getting_up = False
        self.attack_phase = AttackPhase.SEARCH
        self.kick_cooldown_until = 0.0
        self.recovery_hold_until = 0.0
        self._beam_mode = None
        self._beam_attempts = 0
        self._beam_last_cycle = -100
        self._last_logged_mode = None
        self._last_logged_attack_phase = None

    def update_current_behavior(self) -> None:
        """Select one high-level action and always finish with a safe motor packet."""
        world = self.agent.world

        if self._last_logged_mode != world.playmode:
            self._last_logged_mode = world.playmode
            logger.info(
                "[#%d] playmode=%s group=%s",
                world.number,
                world.playmode.name,
                world.playmode_group.name,
            )

        if world.playmode is PlayModeEnum.GAME_OVER:
            self.agent.skills_manager.execute("Neutral")
            self._commit_motors()
            return

        if world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self._run_beam_state()
            self._commit_motors()
            return

        if self.is_getting_up or world.is_fallen():
            self.is_getting_up = not self.agent.skills_manager.execute("GetUp")
            if not self.is_getting_up:
                self.attack_phase = AttackPhase.SEARCH
                self.recovery_hold_until = (
                    world.server_time or 0.0
                ) + self.POST_GETUP_HOLD_SECONDS
            self._commit_motors()
            return

        if (world.server_time or 0.0) < self.recovery_hold_until:
            self.agent.skills_manager.execute(
                "Walk", target_2d=np.zeros(2), is_target_absolute=False
            )
            self._commit_motors()
            return

        if world.playmode_group is PlayModeGroupEnum.OUR_KICK:
            self.run_our_set_piece()
        elif world.playmode_group is PlayModeGroupEnum.THEIR_KICK:
            self.run_their_set_piece()
        elif world.playmode is PlayModeEnum.PLAY_ON:
            self.run_play_strategy()
        else:
            self.agent.skills_manager.execute("Neutral")

        self._commit_motors()

    def _commit_motors(self) -> None:
        if self._last_logged_attack_phase is not self.attack_phase:
            self._last_logged_attack_phase = self.attack_phase
            logger.info(
                "[#%d] attack=%s skill=%s",
                self.agent.world.number,
                self.attack_phase.name,
                self.agent.skills_manager.current_skill_name,
            )
        self._control_head()
        self.agent.robot.commit_motor_targets_pd()

    def _control_head(self) -> None:
        """Keep the ball in view, especially inside the final approach radius."""
        if self.agent.skills_manager.current_skill_name == "GetUp":
            return

        world = self.agent.world
        if world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            return

        robot = self.agent.robot
        if world.is_ball_fresh(1.5):
            delta = world.ball_pos[:2] - world.global_position[:2]
            distance = max(float(np.linalg.norm(delta)), 0.05)
            yaw = MathOps.normalize_deg(
                MathOps.vector_angle(delta) - robot.global_orientation_euler[2]
            )
            camera_height = max(float(world.global_position[2]) + 0.28, 0.55)
            pitch = np.rad2deg(
                np.arctan2(camera_height - float(world.ball_pos[2]), distance)
            )
        else:
            server_time = world.server_time or 0.0
            yaw = 65.0 * np.sin(server_time * 0.8)
            pitch = 32.0

        robot.set_motor_target_position("he1", yaw, kp=12.0, kd=0.35)
        robot.set_motor_target_position("he2", pitch, kp=12.0, kd=0.35)

    def _run_beam_state(self) -> None:
        world = self.agent.world
        beam_positions = self.BEAM_POSES[type(world.field)]
        if isinstance(world.field, MyField):
            if world.playmode_group is PlayModeGroupEnum.ACTIVE_BEAM:
                beam_positions = self.MY_FIELD_ACTIVE_KICKOFF_POSES
            elif world.playmode_group is PlayModeGroupEnum.PASSIVE_BEAM:
                beam_positions = self.MY_FIELD_PASSIVE_KICKOFF_POSES
        canonical_pose = beam_positions.get(
            world.number, beam_positions[max(beam_positions)]
        )
        canonical_position = np.array(canonical_pose[:2], dtype=float)
        canonical_orientation = canonical_pose[2]

        if self._beam_mode != world.playmode:
            self._beam_mode = world.playmode
            self._beam_attempts = 0
            self._beam_last_cycle = -100

        beam_confirmed = (
            np.linalg.norm(world.global_position[:2] - canonical_position) < 0.5
            and world.global_position[2] > 0.3
        )
        should_retry_beam = (
            not beam_confirmed
            and self._beam_attempts < 20
            and world.cycle_count - self._beam_last_cycle >= 10
        )
        if should_retry_beam:
            simulator_position, simulator_orientation = world.to_simulator_pose(
                canonical_position, canonical_orientation
            )
            self.agent.server.commit_beam(
                pos2d=simulator_position.tolist(), rotation=simulator_orientation
            )
            self._beam_attempts += 1
            self._beam_last_cycle = world.cycle_count

        self.attack_phase = AttackPhase.SEARCH
        if beam_confirmed:
            # The learned zero-velocity stance is substantially more stable than
            # a straight-knee pose during long pre-kickoff waits.  Do not invoke
            # it at the off-field staging pose before beam activation.
            self.agent.skills_manager.execute(
                "Walk", target_2d=np.zeros(2), is_target_absolute=False
            )

    def run_play_strategy(self) -> None:
        world = self.agent.world
        number = world.number
        ball_owner = self._select_ball_owner() if world.is_ball_fresh(1.0) else 7

        if number == 1:
            self.play_as_goalkeeper()
        elif number == ball_owner:
            self.run_attack()
        elif number in (2, 3):
            self.play_as_defender()
        elif number == 7:
            self.play_as_forward()
        else:
            self.play_as_midfielder()

    def _select_ball_owner(self) -> int:
        """Assign exactly one field player from stable pitch zones.

        RCSSServerMJ vision is local and the current client has no explicit
        teammate-radio arbitration. A deterministic zone map therefore avoids
        seven robots chasing the same ball while still handing possession from
        defence through midfield to the forward line.
        """
        ball_x, ball_y = self.agent.world.ball_pos[:2]
        if ball_x < -8.0:
            return 2 if ball_y <= 0.0 else 3
        if ball_x < 5.0:
            if ball_x >= -2.0 and abs(ball_y) <= 4.5:
                return 7
            if ball_y < -3.5:
                return 5
            if ball_y > 3.5:
                return 6
            return 4
        return 7

    def run_attack(self) -> None:
        """Closed-loop search, approach, alignment, kick, and recovery FSM."""
        world = self.agent.world
        server_time = world.server_time or 0.0

        if self.attack_phase is AttackPhase.KICK:
            if self.agent.skills_manager.execute("KickRight"):
                self.attack_phase = AttackPhase.RECOVER
                self.kick_cooldown_until = server_time + self.KICK_COOLDOWN_SECONDS
            return

        if self.attack_phase is AttackPhase.RECOVER:
            self.agent.skills_manager.execute(
                "Walk", target_2d=np.zeros(2), is_target_absolute=False
            )
            if server_time >= self.kick_cooldown_until:
                self.attack_phase = AttackPhase.APPROACH
            return

        if not world.is_ball_fresh(self.BALL_FRESHNESS_SECONDS):
            self.attack_phase = AttackPhase.SEARCH
            self._search_for_ball()
            return

        ball = world.ball_pos[:2]
        player = world.global_position[:2]
        goal = np.asarray(world.field.get_their_goal_position(), dtype=float)
        ball_to_goal = goal - ball
        distance_to_goal = np.linalg.norm(ball_to_goal)
        goal_direction = (
            ball_to_goal / distance_to_goal
            if distance_to_goal > 1e-6
            else np.array([1.0, 0.0])
        )
        left_direction = np.array([-goal_direction[1], goal_direction[0]])
        # global_position is the T1 root/torso pose. In simulation the root is
        # about 0.62 m behind the ball when the forward foot reaches it.
        alignment_target = ball - 0.62 * goal_direction + 0.04 * left_direction
        target_orientation = MathOps.vector_angle(goal_direction)
        distance_to_alignment = np.linalg.norm(player - alignment_target)
        orientation_error = abs(
            MathOps.normalize_deg(
                target_orientation - self.agent.robot.global_orientation_euler[2]
            )
        )
        ball_distance = np.linalg.norm(player - ball)

        if distance_to_alignment > 0.42:
            self.attack_phase = AttackPhase.APPROACH
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=alignment_target,
                is_target_absolute=True,
                orientation=(
                    target_orientation if distance_to_alignment < 1.5 else None
                ),
            )
        elif distance_to_alignment > 0.25 or orientation_error > 15.0:
            self.attack_phase = AttackPhase.ALIGN
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=alignment_target,
                is_target_absolute=True,
                orientation=target_orientation,
            )
        elif 0.48 <= ball_distance <= 0.85:
            self.attack_phase = AttackPhase.KICK
            self.agent.skills_manager.execute("KickRight")
        else:
            # Re-acquire a geometrically valid kick pose instead of kicking blindly.
            self.attack_phase = AttackPhase.ALIGN
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=alignment_target,
                is_target_absolute=True,
                orientation=target_orientation,
            )

    def _search_for_ball(self) -> None:
        self.agent.skills_manager.execute(
            "Walk",
            target_2d=np.zeros(2),
            is_target_absolute=False,
            orientation=35.0,
            is_orientation_absolute=False,
        )

    def play_as_goalkeeper(self) -> None:
        world = self.agent.world
        goal = np.asarray(world.field.get_our_goal_position(), dtype=float)
        ball_y = world.ball_pos[1] if world.is_ball_fresh(1.5) else 0.0
        target = np.array([goal[0] + 1.3, np.clip(ball_y, -3.5, 3.5)])
        self.agent.skills_manager.execute(
            "Walk", target_2d=target, is_target_absolute=True, orientation=0.0
        )

    def play_as_defender(self) -> None:
        world = self.agent.world
        number = world.number
        home = np.array([-15.0, -5.0 if number == 2 else 5.0])
        if world.is_ball_fresh(1.0) and world.ball_pos[0] < -9.0:
            ball = world.ball_pos[:2]
            own_goal = np.asarray(world.field.get_our_goal_position(), dtype=float)
            lane = ball - own_goal
            lane_norm = np.linalg.norm(lane)
            target = ball - (lane / lane_norm) * 1.2 if lane_norm > 1e-6 else home
        else:
            target = home
        self.agent.skills_manager.execute(
            "Walk", target_2d=target, is_target_absolute=True
        )

    def play_as_midfielder(self) -> None:
        world = self.agent.world
        lane_by_number = {4: 0.0, 5: -7.0, 6: 7.0}
        if world.is_ball_fresh(1.0):
            support_x = float(np.clip(world.ball_pos[0] - 4.0, -10.0, 12.0))
        else:
            support_x = -10.0
        target = np.array([support_x, lane_by_number.get(world.number, 0.0)])
        self.agent.skills_manager.execute(
            "Walk", target_2d=target, is_target_absolute=True
        )

    def play_as_forward(self) -> None:
        """Offer a forward outlet while another zone owner has the ball."""
        world = self.agent.world
        if world.is_ball_fresh(1.0):
            target = np.array(
                [
                    float(np.clip(world.ball_pos[0] + 4.0, -5.0, 16.0)),
                    float(np.clip(0.5 * world.ball_pos[1], -6.0, 6.0)),
                ]
            )
        else:
            target = np.array([-5.0, 0.0])
        self.agent.skills_manager.execute(
            "Walk", target_2d=target, is_target_absolute=True
        )

    def run_our_set_piece(self) -> None:
        world = self.agent.world
        if world.number == 7:
            self.run_attack()
            return

        beam_pose = self.BEAM_POSES[type(world.field)].get(world.number)
        if beam_pose is None:
            self.agent.skills_manager.execute("Neutral")
            return
        target = np.asarray(beam_pose[:2], dtype=float)
        if world.is_ball_fresh(1.0):
            # Maintain legal spacing while offering a passing option.
            target[0] = min(target[0], world.ball_pos[0] - 2.0)
        self.agent.skills_manager.execute(
            "Walk", target_2d=target, is_target_absolute=True
        )

    def run_their_set_piece(self) -> None:
        world = self.agent.world
        own_goal = np.asarray(world.field.get_our_goal_position(), dtype=float)
        formations = {
            1: (own_goal[0] + 1.3, 0.0),
            2: (own_goal[0] + 10.0, -4.0),
            3: (own_goal[0] + 10.0, 4.0),
            4: (own_goal[0] + 14.0, 0.0),
            5: (own_goal[0] + 15.0, -7.0),
            6: (own_goal[0] + 15.0, 7.0),
            7: (own_goal[0] + 17.0, 0.0),
        }
        target = np.asarray(formations.get(world.number, formations[7]), dtype=float)

        if world.is_ball_fresh(1.0):
            offset = target - world.ball_pos[:2]
            distance = np.linalg.norm(offset)
            if distance < 2.2:
                direction = (
                    offset / distance if distance > 1e-6 else np.array([-1.0, 0.0])
                )
                target = world.ball_pos[:2] + 2.2 * direction

        self.agent.skills_manager.execute(
            "Walk", target_2d=target, is_target_absolute=True
        )
