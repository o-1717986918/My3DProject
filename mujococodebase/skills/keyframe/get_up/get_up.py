from collections import deque
import logging
import os
from mujococodebase.skills.keyframe.keyframe import KeyframeSkill
from mujococodebase.skills.skill import Skill

logger = logging.getLogger()

class GetUp(Skill):
    STABILITY_THRESHOLD_CYCLES: int = 3
    NEUTRAL_EXECUTION_TIME: float = 1.5
    def __init__(self, agent):
        super().__init__(agent)
        self.get_up_front = KeyframeSkill(
            agent=agent,
            file=os.path.join(os.path.dirname(__file__), "get_up_front.yaml"),
        )
        self.get_up_back = KeyframeSkill(
            agent=agent,
            file=os.path.join(os.path.dirname(__file__), "get_up_back.yaml"),
        )

    def execute(self, reset, *args, **kwargs):

        robot = self.agent.robot

        if reset:
            self.neutral_start_time = None
            self.has_get_up = False
            self.gyro_queue = deque(maxlen=self.STABILITY_THRESHOLD_CYCLES)
            self.state = 0
            self.chosen_get_up = None
            self.should_reset_get_up = True

        if not self.has_get_up:
            if not self.chosen_get_up and self.agent.skills_manager.execute_sub_skill("Neutral", True):
                self.gyro_queue.append(max(abs(robot.gyroscope)))
                if len(self.gyro_queue) == self.STABILITY_THRESHOLD_CYCLES and all(g < 2.5 for g in self.gyro_queue):
                    if abs(robot.accelerometer[1]) < 2 and abs(robot.accelerometer[2]) < 3:
                        if robot.accelerometer[0] < -8:
                            self.chosen_get_up = self.get_up_front
                        elif robot.accelerometer[0] > 8:
                            self.chosen_get_up = self.get_up_back
            if self.chosen_get_up:
                self.has_get_up = self.chosen_get_up.execute(reset=self.should_reset_get_up)
                self.should_reset_get_up = False
        else:
            if not self.neutral_start_time:
                self.neutral_start_time = self.agent.world.server_time

            neutral_elapsed_time = (
                self.agent.world.server_time - self.neutral_start_time
            )

            if neutral_elapsed_time < self.NEUTRAL_EXECUTION_TIME:
                self.agent.skills_manager.execute_sub_skill(
                    "Walk", reset=neutral_elapsed_time <= 1e-6, target_2d=(0, 0), is_target_absolute=False
                )
            else:
                return True

        return False

    def is_ready(self, *args):
        return self.agent.world.is_fallen()
