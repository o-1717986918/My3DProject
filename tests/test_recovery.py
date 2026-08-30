from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from mujococodebase.skills.external.apollo_get_up import ApolloGetUpPolicy
from mujococodebase.skills.keyframe.get_up.get_up import FallPose, classify_fall_pose
from mujococodebase.world.world import World


@pytest.mark.parametrize(
    ("axis", "angle", "expected"),
    [
        ("x", 0.0, FallPose.UPRIGHT),
        ("y", 90.0, FallPose.FRONT),
        ("y", -90.0, FallPose.BACK),
        ("x", -90.0, FallPose.LEFT),
        ("x", 90.0, FallPose.RIGHT),
    ],
)
def test_fall_pose_classification(axis, angle, expected):
    quaternion = R.from_euler(axis, angle, degrees=True).as_quat()

    assert classify_fall_pose(quaternion) is expected


def test_world_upright_requires_height_and_torso_orientation():
    robot = SimpleNamespace(global_orientation_quat=np.array([0.0, 0.0, 0.0, 1.0]))
    agent = SimpleNamespace(robot=robot)
    world = World(agent=agent, team_name="Team", number=1, field_name="my_field")

    world.global_position[2] = 0.6
    assert world.is_upright()

    robot.global_orientation_quat = R.from_euler("x", 90, degrees=True).as_quat()
    assert not world.is_upright()

    robot.global_orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
    world.global_position[2] = 0.2
    assert not world.is_upright()


class RecordingPolicyRobot:
    ROBOT_MOTORS = tuple(f"motor_{index}" for index in range(23))

    def __init__(self):
        self.global_orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gyroscope = np.zeros(3)
        self.motor_positions = {name: 0.0 for name in self.ROBOT_MOTORS}
        self.motor_speeds = {name: 0.0 for name in self.ROBOT_MOTORS}
        self.commands = []

    def get_ordered_motor_state(self):
        return np.zeros(23), np.zeros(23)

    def set_motor_target_position(self, name, position, kp, kd):
        self.commands.append((name, position, kp, kd))


def make_policy_agent():
    robot = RecordingPolicyRobot()
    world = SimpleNamespace(server_time=1.0, is_upright=lambda: False)
    return SimpleNamespace(robot=robot, world=world)


def test_apollo_policy_observation_and_joint_targets(monkeypatch, tmp_path):
    model_path = tmp_path / "policy.onnx"
    model_path.touch()
    observed = {}

    monkeypatch.setenv("MY3D_APOLLO_GETUP_MODEL", str(model_path))
    monkeypatch.delenv("MY3D_GETUP_BACKEND", raising=False)
    monkeypatch.setattr(
        "mujococodebase.skills.external.apollo_get_up.load_network",
        lambda _: object(),
    )

    def fake_inference(observation, _model):
        observed["value"] = observation.copy()
        return np.zeros(23)

    monkeypatch.setattr(
        "mujococodebase.skills.external.apollo_get_up.run_network", fake_inference
    )
    agent = make_policy_agent()
    policy = ApolloGetUpPolicy(agent)

    assert not policy.execute(reset=True)
    observation = observed["value"]
    assert observation.shape == (75,)
    np.testing.assert_allclose(observation[:3], 0.0)
    np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(
        observation[6:29], -ApolloGetUpPolicy.DEFAULT_POSITIONS_RAD
    )
    np.testing.assert_allclose(observation[-23:], 0.0)
    assert len(agent.robot.commands) == 23
    np.testing.assert_allclose(
        [command[1] for command in agent.robot.commands], np.zeros(23)
    )


def test_apollo_policy_disables_itself_on_bad_output(monkeypatch, tmp_path):
    model_path = tmp_path / "policy.onnx"
    model_path.touch()
    monkeypatch.setenv("MY3D_APOLLO_GETUP_MODEL", str(model_path))
    monkeypatch.setattr(
        "mujococodebase.skills.external.apollo_get_up.load_network",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "mujococodebase.skills.external.apollo_get_up.run_network",
        lambda *_: np.zeros(22),
    )
    policy = ApolloGetUpPolicy(make_policy_agent())

    assert not policy.execute(reset=True)
    assert not policy.available
