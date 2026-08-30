from types import SimpleNamespace

import numpy as np
import pytest

from mujococodebase.robot import T1


class RecordingServer:
    def __init__(self):
        self.messages = []

    def commit(self, message):
        self.messages.append(message)


def make_robot() -> tuple[T1, RecordingServer]:
    server = RecordingServer()
    robot = T1(agent=SimpleNamespace(server=server))
    return robot, server


def test_joint_targets_are_clamped_to_t1_limits():
    robot, _ = make_robot()
    robot.set_motor_target_position("lle4", -70.0, kp=250.0, kd=20.0)
    assert robot.motor_targets["lle4"] == {
        "target_position": 0.0,
        "kp": 100.0,
        "kd": 10.0,
    }


def test_non_finite_target_is_rejected_without_poisoning_buffer():
    robot, server = make_robot()
    robot.set_motor_target_position("rle4", 35.0)
    robot.set_motor_target_position("rle4", np.nan)
    assert robot.motor_targets["rle4"]["target_position"] == 35.0
    robot.commit_motor_targets_pd()
    assert server.messages
    assert all("nan" not in message.lower() for message in server.messages)
    assert all("inf" not in message.lower() for message in server.messages)


def test_motor_state_is_always_in_policy_order():
    robot, _ = make_robot()
    robot.motor_positions = {"rle6": 23.0, "he1": 1.0}
    robot.motor_speeds = {"rle6": 42.0, "he1": 2.0}
    positions, speeds = robot.get_ordered_motor_state()
    assert positions[0] == 1.0
    assert positions[-1] == 23.0
    assert speeds[0] == 2.0
    assert speeds[-1] == 42.0


def test_unknown_motor_is_not_silently_added():
    robot, _ = make_robot()
    with pytest.raises(KeyError):
        robot.set_motor_target_position("not-a-motor", 0.0)


def test_perception_joint_names_map_to_command_motor_names():
    robot, _ = make_robot()
    mapping = robot.MOTOR_FROM_SENSOR_TO_SERVER
    assert mapping["q_hj1"] == "he1"
    assert mapping["q_tj1"] == "te1"
    assert mapping["q_llj4"] == "lle4"
    assert mapping["q_rlj6"] == "rle6"
