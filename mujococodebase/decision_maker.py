from dataclasses import Field
import logging
from typing import Mapping

import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.field import FIFAField, HLAdultField, MyField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum


logger = logging.getLogger()


class DecisionMaker:
    """
    Responsible for deciding what the agent should do at each moment.

    This class is called every simulation step to update the agent's behavior
    based on the current state of the world and game conditions.
    """

    BEAM_POSES: Mapping[type[Field], Mapping[int, tuple[float, float, float]]] ={
        FIFAField: {
            1: (2.1, 0, 0),
            2: (22.0, 12.0, 0),
            3: (22.0, 4.0, 0),
            4: (22.0, -4.0, 0),
            5: (22.0, -12.0, 0),
            6: (15.0, 0.0, 0),
            7: (4.0, 16.0, 0),
            8: (11.0, 6.0, 0),
            9: (11.0, -6.0, 0),
            10: (4.0, -16.0, 0),
            11: (7.0, 0.0, 0),
        },
        HLAdultField: {
            1: (7.0, 0.0, 0),
            2: (2.0, -1.5, 0),
            3: (2.0, 1.5, 0),
        },
        MyField: {
            1: (20.0, 0.0, 0),
            2: (10.0, -6.0, 0),
            3: (10.0, 6.0, 0),
            4: (5.0, -3.0, 0),
            5: (5.0, 3.0, 0),
            6: (15.0, 0.0, 0),
            7: (2.0, 0.0, 0),
        }
    } 

    def __init__(self, agent):
        """
        Creates a new DecisionMaker linked to the given agent.

        Args:
            agent: The main agent that owns this DecisionMaker.
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.is_getting_up: bool = False

    def update_current_behavior(self) -> None:
        """
        Chooses what the agent should do in the current step.

        This function checks the game state and decides which behavior
        or skill should be executed next.
        """

        if self.agent.world.playmode is PlayModeEnum.GAME_OVER:
            return

        if self.agent.world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            self.agent.server.commit_beam(
                pos2d=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][:2],
                rotation=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][2],
            )

        if self.is_getting_up or self.agent.skills_manager.is_ready(skill_name="GetUp"):
            self.is_getting_up = not self.agent.skills_manager.execute(skill_name="GetUp")

        elif self.agent.world.playmode_group == PlayModeGroupEnum.OTHER:
            if self.agent.world.playmode == PlayModeEnum.PLAY_ON:
                self.run_play_strategy()
            elif self.agent.world.playmode in (
                PlayModeEnum.BEFORE_KICK_OFF,
                PlayModeEnum.THEIR_GOAL,
                PlayModeEnum.OUR_GOAL,
            ):
                self.agent.skills_manager.execute("Neutral")

        elif self.agent.world.playmode_group == PlayModeGroupEnum.OUR_KICK:
            self.run_our_set_piece()

        elif self.agent.world.playmode_group == PlayModeGroupEnum.THEIR_KICK:
            self.run_their_set_piece()

        self.agent.robot.commit_motor_targets_pd()

    def carry_ball(self):
        """
        Basic example of a behavior: moves the robot toward the goal while handling the ball.
        """
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            return 
        ball_to_goal_dir = ball_to_goal / bg_norm

        dist_from_ball_to_start_carrying = 0.30
        carry_ball_pos = ball_pos - ball_to_goal_dir * dist_from_ball_to_start_carrying

        my_to_ball = ball_pos - my_pos
        my_to_ball_norm = np.linalg.norm(my_to_ball)
        if my_to_ball_norm == 0:
            my_to_ball_dir = np.zeros(2)
        else:
            my_to_ball_dir = my_to_ball / my_to_ball_norm

        cosang = np.dot(my_to_ball_dir, ball_to_goal_dir)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_diff = np.arccos(cosang)

        ANGLE_TOL = np.deg2rad(7.5)
        aligned = (my_to_ball_norm > 1e-6) and (angle_diff <= ANGLE_TOL)

        behind_ball = np.dot(my_pos - ball_pos, ball_to_goal_dir) < 0
        desired_orientation = MathOps.vector_angle(ball_to_goal)

        if not aligned or not behind_ball:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=carry_ball_pos,
                is_target_absolute=True,
                orientation=None if np.linalg.norm(my_pos - carry_ball_pos) > 2 else desired_orientation
            )
        else:
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=their_goal_pos,
                is_target_absolute=True,
                orientation=desired_orientation
            )

    def run_play_strategy(self):
        """根据球员号码分配角色行为"""
        number = self.agent.world.number
        if number == 1:
            self.play_as_goalkeeper()
        elif number in (2, 3):
            self.play_as_defender()
        elif number in (4, 5, 6):
            self.play_as_midfielder()
        else:
            self.play_as_forward()

    def play_as_goalkeeper(self):
        """守门员：守住球门，跟随球移动"""
        my_pos = self.agent.world.global_position[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        our_goal = self.agent.world.field.get_our_goal_position()[:2]

        guard_x = our_goal[0] + 1.5
        guard_y = np.clip(ball_pos[1], -4, 4)
        guard_pos = np.array([guard_x, guard_y])

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=guard_pos,
            is_target_absolute=True,
            orientation=0.0,
        )

    def play_as_defender(self):
        """后卫：守住后场，球靠近时上前逼抢"""
        my_pos = self.agent.world.global_position[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        our_goal = self.agent.world.field.get_our_goal_position()[:2]

        # 根据号码分散站位 (2号偏左, 3号偏右)
        y_offset = (self.agent.world.number - 2.5) * 8
        defense_x = our_goal[0] + 15
        home_pos = np.array([defense_x, y_offset])

        dist_to_ball = np.linalg.norm(my_pos - ball_pos)
        if dist_to_ball < 6.0:
            target = ball_pos
        else:
            target = home_pos

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=target,
            is_target_absolute=True,
        )

    def play_as_midfielder(self):
        """中场：在中圈活动，连接攻防"""
        ball_pos = self.agent.world.ball_pos[:2]

        # 根据号码分散站位
        y_offset = (self.agent.world.number - 5.5) * 6
        home_pos = np.array([5, y_offset])

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=home_pos,
            is_target_absolute=True,
        )

    def play_as_forward(self):
        """前锋：抢球并带向对方球门"""
        self.carry_ball()

    def _get_kicker_number(self) -> int:
        """决定由几号球员去开球。
        优先选离球最近的球员，如果无法判断则默认让1号开。
        """
        my_pos = self.agent.world.global_position[:2]
        ball_pos = self.agent.world.ball_pos[:2]
        dist = np.linalg.norm(my_pos - ball_pos)
        # 距球1m内则认为是开球者
        if dist < 1.0:
            return self.agent.world.number
        return 0  # 非开球者

    def run_our_set_piece(self):
        """我方开球/任意球/角球等定位球。"""
        number = self.agent.world.number
        playmode = self.agent.world.playmode
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]

        # 谁是开球者：距球最近的去开，一号优先级最高
        dist_to_ball = np.linalg.norm(my_pos - ball_pos)
        is_kicker = dist_to_ball < 1.0

        if is_kicker:
            # 开球者：走到球旁，面向对方球门
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=ball_pos,
                is_target_absolute=True,
                orientation=0.0,
            )
        else:
            # 其他人：分散站位策应
            y_offset = (number - 4) * 4
            support_pos = np.array([ball_pos[0] - 5, y_offset])
            self.agent.skills_manager.execute(
                "Walk",
                target_2d=support_pos,
                is_target_absolute=True,
            )

    def run_their_set_piece(self):
        """对方开球/任意球等定位球——全员退防。"""
        number = self.agent.world.number
        our_goal = self.agent.world.field.get_our_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]

        # 在球门和球之间排人墙
        fm = {
            1:  (our_goal[0] + 1.5,  0.0),
            2:  (our_goal[0] + 8,   -3.0),
            3:  (our_goal[0] + 8,    3.0),
            4:  (our_goal[0] + 12,  -5.0),
            5:  (our_goal[0] + 12,   5.0),
            6:  (our_goal[0] + 5,   -1.5),
            7:  (our_goal[0] + 5,    1.5),
        }
        default = (our_goal[0] + 10, (number - 4) * 3)
        defend_pos = np.array(fm.get(number, default))

        self.agent.skills_manager.execute(
            "Walk",
            target_2d=defend_pos,
            is_target_absolute=True,
        )
