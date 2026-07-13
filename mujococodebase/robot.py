from abc import ABC, abstractmethod
from typing import override
import numpy as np
from mujococodebase.server import Server


class Robot(ABC):
    """
    Base class for all robot models.

    This class defines the main structure and common data used by any robot,
    such as motor positions, sensors, and control messages.
    """
    def __init__(self, agent):
        """
        Creates a new robot linked to the given agent.

        Args:
            agent: The main agent that owns this robot.
        """
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.server: Server = self.agent.server

        self.motor_targets: dict = {
            motor: {"target_position": 0.0, "kp": 0.0, "kd": 0.0}
            for motor in self.ROBOT_MOTORS
        }

        self.motor_positions: dict = {motor: 0.0 for motor in self.ROBOT_MOTORS}  # degrees

        self.motor_speeds: dict = {motor: 0.0 for motor in self.ROBOT_MOTORS}  # degrees/s

        self._global_cheat_orientation = np.array([0, 0, 0, 1])  # quaternion [x, y, z, w]

        self.global_orientation_quat = np.array([0, 0, 0, 1]) # quaternion [x, y, z, w]

        self.global_orientation_euler = np.zeros(3) # euler [roll, pitch, yaw]

        self.gyroscope = np.zeros(3)  # angular velocity [roll, pitch, yaw] (degrees/s)

        self.accelerometer = np.zeros(3)  # linear acceleration [x, y, z] (m/s²)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the robot model name.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def ROBOT_MOTORS(self) -> tuple[str, ...]:
        """
        Returns the list of motor names used by this robot.
        """
        raise NotImplementedError()

    def set_motor_target_position(
        self, motor_name: str, target_position: float, kp: float = 10, kd: float = 0.1
    ) -> None:
        """
        Sets the desired position and PD gains for a given motor.

        For now, directly sets positions, as the simulator is doing the control
        Args:
            motor_name: Name of the motor.
            target_position: Desired position in radians.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        self.motor_targets[motor_name] = {
            "target_position": target_position,
            "kp": kp,
            "kd": kd,
        }

    def commit_motor_targets_pd(self) -> None:
        """
        Sends all motor target commands to the simulator.
        """
        for motor_name, target_description in self.motor_targets.items():
            motor_msg = f"({motor_name} {target_description["target_position"]:.2f} 0.0 {target_description["kp"]:.2f} {target_description["kd"]:.2f} 0.0)"
            self.server.commit(motor_msg)


class T1(Robot):
    """
    Booster T1
    """
    @override
    def __init__(self, agent):
        super().__init__(agent)

        self.joint_nominal_position = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    @property
    @override
    def name(self) -> str:
        return "T1"

    @property
    @override
    def ROBOT_MOTORS(self) -> tuple[str, ...]:
        return (
            "he1",
            "he2",
            "lae1",
            "lae2",
            "lae3",
            "lae4",
            "rae1",
            "rae2",
            "rae3",
            "rae4",
            "te1",
            "lle1",
            "lle2",
            "lle3",
            "lle4",
            "lle5",
            "lle6",
            "rle1",
            "rle2",
            "rle3",
            "rle4",
            "rle5",
            "rle6",
        )

    @property
    def MOTOR_FROM_READABLE_TO_SERVER(self) -> dict:
        """
        Maps readable joint names to their simulator motor codes.
        """
        return {
            "Head_yaw": "he1",
            "Head_pitch": "he2",
            "Left_Shoulder_Pitch": "lae1",
            "Left_Shoulder_Roll": "lae2",
            "Left_Elbow_Pitch": "lae3",
            "Left_Elbow_Yaw": "lae4",
            "Right_Shoulder_Pitch": "rae1",
            "Right_Shoulder_Roll": "rae2",
            "Right_Elbow_Pitch": "rae3",
            "Right_Elbow_Yaw": "rae4",
            "Waist": "te1",
            "Left_Hip_Pitch": "lle1",
            "Left_Hip_Roll": "lle2",
            "Left_Hip_Yaw": "lle3",
            "Left_Knee_Pitch": "lle4",
            "Left_Ankle_Pitch": "lle5",
            "Left_Ankle_Roll": "lle6",
            "Right_Hip_Pitch": "rle1",
            "Right_Hip_Roll": "rle2",
            "Right_Hip_Yaw": "rle3",
            "Right_Knee_Pitch": "rle4",
            "Right_Ankle_Pitch": "rle5",
            "Right_Ankle_Roll": "rle6",
        }

    @property
    def MOTOR_SYMMETRY(self) -> list[str, bool]:
        """
        Defines pairs of symmetric motors and whether their direction is inverted.

        Returns:
            A dictionary where each key is a logical joint group name,
            and the value is a tuple (motor_names, inverted).
        """
        return {
            "Head_yaw": (("Head_yaw",), False),
            "Head_pitch": (("Head_pitch",), False),
            "Shoulder_Pitch": (
                (
                    "Left_Shoulder_Pitch",
                    "Right_Shoulder_Pitch",
                ),
                False,
            ),
            "Shoulder_Roll": (
                (
                    "Left_Shoulder_Roll",
                    "Right_Shoulder_Roll",
                ),
                True,
            ),
            "Elbow_Pitch": (
                (
                    "Left_Elbow_Pitch",
                    "Right_Elbow_Pitch",
                ),
                False,
            ),
            "Elbow_Yaw": (
                (
                    "Left_Elbow_Yaw",
                    "Right_Elbow_Yaw",
                ),
                True,
            ),
            "Waist": (("Waist",), False),
            "Hip_Pitch": (
                (
                    "Left_Hip_Pitch",
                    "Right_Hip_Pitch",
                ),
                False,
            ),
            "Hip_Roll": (
                (
                    "Left_Hip_Roll",
                    "Right_Hip_Roll",
                ),
                True,
            ),
            "Hip_Yaw": (
                (
                    "Left_Hip_Yaw",
                    "Right_Hip_Yaw",
                ),
                True,
            ),
            "Knee_Pitch": (
                (
                    "Left_Knee_Pitch",
                    "Right_Knee_Pitch",
                ),
                False,
            ),
            "Ankle_Pitch": (
                (
                    "Left_Ankle_Pitch",
                    "Right_Ankle_Pitch",
                ),
                False,
            ),
            "Ankle_Roll": (
                (
                    "Left_Ankle_Roll",
                    "Right_Ankle_Roll",
                ),
                True,
            ),
        }
