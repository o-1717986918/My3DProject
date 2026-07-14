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
            7: (-2.0, 0.0, 0),
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

        # 诊断日志：playmode变化时输出
        if not hasattr(self, '_last_logged_mode'):
            self._last_logged_mode = None
        if self._last_logged_mode != self.agent.world.playmode:
            self._last_logged_mode = self.agent.world.playmode
            logger.info(
                f"[#{self.agent.world.number}] playmode={self.agent.world.playmode.name} "
                f"group={self.agent.world.playmode_group.name}"
            )

        # Beam只发一次：检测playmode变化，首次进入beam状态时发送
        # 注意：beam帧不发送电机指令，避免冲突
        if not hasattr(self, '_has_beamed_for_mode'):
            self._has_beamed_for_mode = None
        if self.agent.world.playmode_group in (
            PlayModeGroupEnum.ACTIVE_BEAM,
            PlayModeGroupEnum.PASSIVE_BEAM,
        ):
            if self._has_beamed_for_mode != self.agent.world.playmode:
                self._has_beamed_for_mode = self.agent.world.playmode
                pos2d = list(self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][:2])
                # 右队需要翻转X坐标，确保beam在己方半场
                if not self.agent.world.is_left_team:
                    pos2d[0] = -pos2d[0]
                self.agent.server.commit_beam(
                    pos2d=pos2d,
                    rotation=self.BEAM_POSES[type(self.agent.world.field)][self.agent.world.number][2],
                )
            else:
                self.agent.skills_manager.execute("Neutral")

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

    def run_our_set_piece(self):
        """我方开球/任意球/角球等定位球。"""
        number = self.agent.world.number
        playmode = self.agent.world.playmode
        ball_pos = self.agent.world.ball_pos[:2]
        my_pos = self.agent.world.global_position[:2]
        their_goal_pos = self.agent.world.field.get_their_goal_position()[:2]

        # 球到对方球门的方向
        ball_to_goal = their_goal_pos - ball_pos
        bg_norm = np.linalg.norm(ball_to_goal)
        if bg_norm == 0:
            ball_to_goal_dir = np.array([1.0, 0.0])
        else:
            ball_to_goal_dir = ball_to_goal / bg_norm
        orientation = MathOps.vector_angle(ball_to_goal)

        if playmode == PlayModeEnum.OUR_KICK_OFF:
            # --- 开球专用逻辑 ---
            # 开球者：beam位置离球（中点）最近的球员
            # 所有agent共享BEAM_POSES，各自独立计算，结果一致
            beam_positions = self.BEAM_POSES[type(self.agent.world.field)]
            closest_to_ball = min(beam_positions.keys(),
                key=lambda p: np.linalg.norm(np.array(beam_positions[p][:2]) - ball_pos))
            is_kicker = (number == closest_to_ball)

            if is_kicker:
                # 两阶段：先走到球后方（面向球门），再执行踢球动作
                behind_ball = ball_pos - ball_to_goal_dir * 0.20
                dist_to_behind = np.linalg.norm(my_pos - behind_ball)

                if dist_to_behind < 0.25:
                    # 阶段2：已就位，执行踢球
                    self.agent.skills_manager.execute("KickRight")
                else:
                    # 阶段1：走到球后方
                    self.agent.skills_manager.execute(
                        "Walk",
                        target_2d=behind_ball,
                        is_target_absolute=True,
                        orientation=orientation,
                    )
            else:
                # 非开球者：分散站位准备接球
                y_offset = (number - 4) * 5
                support_pos = np.array([ball_pos[0] - 3, y_offset])
                self.agent.skills_manager.execute(
                    "Walk",
                    target_2d=support_pos,
                    is_target_absolute=True,
                )
        else:
            # --- 其他定位球（界外球、角球、任意球等）---
            dist_to_ball = np.linalg.norm(my_pos - ball_pos)
            is_kicker = dist_to_ball < 1.0

            if is_kicker:
                self.agent.skills_manager.execute(
                    "Walk",
                    target_2d=ball_pos,
                    is_target_absolute=True,
                    orientation=orientation,
                )
            else:
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
        playmode = self.agent.world.playmode
        our_goal = self.agent.world.field.get_our_goal_position()[:2]
        ball_pos = self.agent.world.ball_pos[:2]

        if playmode == PlayModeEnum.THEIR_KICK_OFF:
            # --- 对方开球：分散站位，前锋前压 ---
            kickoff_positions = {
                1:  (our_goal[0] + 1.5,  0.0),    # 门将
                2:  (our_goal[0] + 10,  -4.0),    # 后卫
                3:  (our_goal[0] + 10,   4.0),    # 后卫
                4:  (our_goal[0] + 16,  -6.0),    # 中场
                5:  (our_goal[0] + 16,   6.0),    # 中场
                6:  (our_goal[0] + 20,  -3.0),    # 中场
                7:  (our_goal[0] + 22,   3.0),    # 前锋，靠近中线压迫
            }
            defend_pos = np.array(kickoff_positions.get(number, (our_goal[0] + 15, (number - 4) * 4)))
        else:
            # --- 其他定位球：在球门和球之间排人墙 ---
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
