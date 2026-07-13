import math
import os
import numpy as np
from mujococodebase.skills.skill import Skill
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.utils.neural_network import run_network, load_network
from scipy.spatial.transform import Rotation as R

class Walk(Skill):
    def __init__(self, agent):
        super().__init__(agent)
        self.joint_nominal_position = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.4,
                0.0,
                -0.4,
                0.0,
                -1.4,
                0.0,
                0.4,
                0.0,
                -0.4,
                0.0,
                0.0,
                0.8,
                -0.4,
                0.0,
                0.4,
                0.0,
                0.0,
                -0.8,
                0.4,
                0.0,
            ]
        )
        self.train_sim_flip = np.array(
            [
                1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ]
        )
        self.scaling_factor = 0.5

        self.previous_action = np.zeros(len(self.agent.robot.ROBOT_MOTORS))

        self.model = load_network(model_path=os.path.join(os.path.dirname(__file__), "walk.onnx"))


    def execute(self, reset: bool, target_2d: list, is_target_absolute: bool, orientation: float=None, is_orientation_absolute: bool=True) -> bool:
        
        robot = self.agent.robot
        world = self.agent.world
        
        velocity = None

        if is_target_absolute:
            raw_target = target_2d - world.global_position[:2]
            velocity = MathOps.rotate_2d_vec(raw_target, -robot.global_orientation_euler[2], is_rad=False)
        else:
            velocity = target_2d

        rel_orientation = None
        if orientation is None:
            rel_orientation = MathOps.vector_angle(velocity) * 0.3
        elif is_orientation_absolute:
            rel_orientation = MathOps.normalize_deg(orientation - robot.global_orientation_euler[2])
        else:
            rel_orientation = orientation * 0.3

        rel_orientation = np.clip(rel_orientation, -0.25, 0.25)

        velocity = np.concat([velocity, np.array([rel_orientation])], axis=0)

        velocity[0] = np.clip(velocity[0], -0.5, 0.5)
        velocity[1] = np.clip(velocity[1], -0.25, 0.25)

        radian_joint_positions = np.deg2rad(list(robot.motor_positions.values()))
        radian_joint_speeds = np.deg2rad(list(robot.motor_speeds.values()))

        qpos_qvel_previous_action = np.vstack(
            (
                [
                    (
                        (radian_joint_positions * self.train_sim_flip)
                        - self.joint_nominal_position
                    )
                    / 4.6,
                    radian_joint_speeds / 110.0 * self.train_sim_flip,
                    self.previous_action / 10.0,
                ]
            )
        ).T.flatten()

        ang_vel = np.clip(np.deg2rad(robot.gyroscope) / 50.0, -1.0, 1.0)
        orientation_quat_inv = R.from_quat(robot._global_cheat_orientation).inv()
        projected_gravity = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        #[0.5,0.25,0.25]
        observation = np.concatenate(
            [
                qpos_qvel_previous_action,
                ang_vel,
                velocity,
                projected_gravity,
            ]
        )
        observation = np.clip(observation, -10.0, 10.0)

        nn_action = run_network(obs=observation, model=self.model)

        target_joint_positions = (
            self.joint_nominal_position + self.scaling_factor * nn_action
        )
        target_joint_positions *= self.train_sim_flip

        self.previous_action = nn_action

        for idx, target in enumerate(target_joint_positions):
            robot.set_motor_target_position(
                robot.ROBOT_MOTORS[idx], target*180/math.pi, kp=25, kd=0.6
            )

    def is_ready(self, *args, **kwargs) -> bool:
        return True