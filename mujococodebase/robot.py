from abc import ABC, abstractmethod
import logging
from typing import override
import numpy as np
from mujococodebase.server import Server

logger = logging.getLogger(__name__)


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

        self.motor_positions: dict = {
            motor: 0.0 for motor in self.ROBOT_MOTORS
        }  # degrees

        self.motor_speeds: dict = {
            motor: 0.0 for motor in self.ROBOT_MOTORS
        }  # degrees/s

        self._global_cheat_orientation = np.array(
            [0, 0, 0, 1]
        )  # quaternion [x, y, z, w]

        self.global_orientation_quat = np.array([0, 0, 0, 1])  # quaternion [x, y, z, w]

        self.global_orientation_euler = np.zeros(3)  # euler [roll, pitch, yaw]

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
            target_position: Desired position in degrees (the wire protocol unit).
            kp: Proportional gain.
            kd: Derivative gain.
        """
        if motor_name not in self.motor_targets:
            raise KeyError(f"Unknown motor '{motor_name}'")

        values = np.asarray([target_position, kp, kd], dtype=float)
        if not np.all(np.isfinite(values)):
            logger.error(
                "Rejected non-finite motor command for %s: position=%r kp=%r kd=%r",
                motor_name,
                target_position,
                kp,
                kd,
            )
            return

        lower, upper = self.MOTOR_LIMITS_DEG[motor_name]
        safe_position = float(np.clip(target_position, lower, upper))
        safe_kp = float(np.clip(kp, 0.0, 100.0))
        safe_kd = float(np.clip(kd, 0.0, 10.0))
        self.motor_targets[motor_name] = {
            "target_position": safe_position,
            "kp": safe_kp,
            "kd": safe_kd,
        }

    def get_ordered_motor_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return joint positions and speeds in the policy's fixed motor order."""
        positions = np.array(
            [self.motor_positions.get(name, 0.0) for name in self.ROBOT_MOTORS],
            dtype=float,
        )
        speeds = np.array(
            [self.motor_speeds.get(name, 0.0) for name in self.ROBOT_MOTORS],
            dtype=float,
        )
        return (
            np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(speeds, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def commit_motor_targets_pd(self) -> None:
        """
        Sends all motor target commands to the simulator.
        """
        for motor_name, target_description in self.motor_targets.items():
            target = target_description["target_position"]
            kp = target_description["kp"]
            kd = target_description["kd"]
            motor_msg = f"({motor_name} {target:.2f} 0.0 {kp:.2f} {kd:.2f} 0.0)"
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
    def MOTOR_LIMITS_DEG(self) -> dict[str, tuple[float, float]]:
        """T1 MuJoCo joint limits, converted from radians to degrees."""
        radian_limits = (
            (-1.57, 1.57),
            (-0.35, 1.22),
            (-3.31, 1.22),
            (-1.74, 1.57),
            (-2.27, 2.27),
            (-2.44, 0.0),
            (-3.31, 1.22),
            (-1.57, 1.74),
            (-2.27, 2.27),
            (0.0, 2.44),
            (-1.57, 1.57),
            (-1.8, 1.57),
            (-0.2, 1.57),
            (-1.0, 1.0),
            (0.0, 2.34),
            (-0.87, 0.35),
            (-0.44, 0.44),
            (-1.8, 1.57),
            (-1.57, 0.2),
            (-1.0, 1.0),
            (0.0, 2.34),
            (-0.87, 0.35),
            (-0.44, 0.44),
        )
        return {
            motor: tuple(float(v) for v in np.rad2deg(limit))
            for motor, limit in zip(self.ROBOT_MOTORS, radian_limits, strict=True)
        }

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
    def MOTOR_FROM_SENSOR_TO_SERVER(self) -> dict[str, str]:
        """Map T1 joint sensor identifiers to motor command identifiers."""
        sensor_groups = (
            ("q_hj", ("he1", "he2")),
            ("q_laj", ("lae1", "lae2", "lae3", "lae4")),
            ("q_raj", ("rae1", "rae2", "rae3", "rae4")),
            ("q_tj", ("te1",)),
            ("q_llj", ("lle1", "lle2", "lle3", "lle4", "lle5", "lle6")),
            ("q_rlj", ("rle1", "rle2", "rle3", "rle4", "rle5", "rle6")),
        )
        return {
            f"{prefix}{index}": motor
            for prefix, motors in sensor_groups
            for index, motor in enumerate(motors, start=1)
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
