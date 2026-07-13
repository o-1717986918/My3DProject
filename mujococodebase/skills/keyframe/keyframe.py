import logging
from mujococodebase.skills.skill import Skill
import yaml


class KeyframeSkill(Skill):
    def __init__(self, agent, file: str):
        super().__init__(agent)

        self.keyframe_step: int = 0
        self.keyframe_start_time: float = None

        with open(f"{file}", "r") as f:
            self.skill_description = yaml.safe_load(f)

        self.has_symmetry: bool = self.skill_description["symmetry"]
        self.keyframes: list = self.skill_description["keyframes"]
        self.kp: float = self.skill_description["kp"]
        self.kd: float = self.skill_description["kd"]

    def execute(self, reset, *args, **kwargs):
        if reset:
            self.keyframe_step = 0
            self.keyframe_start_time = self.agent.world.server_time

        current_keyframe = self.keyframes[self.keyframe_step]

        raw_motor_positions = current_keyframe["motor_positions"]

        current_keyframe_kp = current_keyframe.get("kp", self.kp)
        current_keyframe_kd = current_keyframe.get("kd", self.kd)

        p_gains = current_keyframe.get("p_gains", None)
        d_gains = current_keyframe.get("d_gains", None)

        for readable_motor_name, position in raw_motor_positions.items():

            p_gain = (
                p_gains.get(readable_motor_name, current_keyframe_kp)
                if p_gains
                else current_keyframe_kp
            )
            d_gain = (
                d_gains.get(readable_motor_name, current_keyframe_kd)
                if d_gains
                else current_keyframe_kd
            )

            if self.has_symmetry:
                motor_names, is_inverse_direction = self.agent.robot.MOTOR_SYMMETRY[
                    readable_motor_name
                ]
                for motor_name in motor_names:
                    server_motor_name = self.agent.robot.MOTOR_FROM_READABLE_TO_SERVER[
                        motor_name
                    ]

                    self.agent.robot.set_motor_target_position(
                        motor_name=server_motor_name,
                        target_position=position if is_inverse_direction else -position,
                        kp=p_gain,
                        kd=d_gain,
                    )
                    if is_inverse_direction:
                        is_inverse_direction = False  # Only inverts one joint
            else:
                server_motor_name = self.agent.robot.MOTOR_FROM_READABLE_TO_SERVER[
                    readable_motor_name
                ]
                self.agent.robot.set_motor_target_position(
                    motor_name=server_motor_name,
                    target_position=position,
                    kp=self.kp,
                    kd=self.kd,
                )

        keyframe_time: float = current_keyframe["delta"]
        elapsed_keyframe_time = self.agent.world.server_time - self.keyframe_start_time

        if elapsed_keyframe_time >= keyframe_time:
            self.keyframe_start_time = self.agent.world.server_time
            self.keyframe_step += 1

            if self.keyframe_step >= len(self.keyframes):
                return True

        return False

    def is_ready(self, *args):
        return True
