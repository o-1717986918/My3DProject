from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from my3d_rl.apollo_walk_cpu import ApolloWalkCpu, apollo_walk_observation


def test_apollo_walk_observation_has_deployed_shape_and_clipping():
    observation = apollo_walk_observation(
        angular_velocity=np.array([20.0, 0.0, -20.0]),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        velocity_command=np.zeros(3),
        joint_position_offset=np.zeros(23),
        joint_velocity=np.zeros(23),
        previous_action=np.zeros(23),
    )

    assert observation.shape == (78,)
    assert observation.dtype == np.float32
    assert observation[0] == 10.0
    assert observation[2] == -10.0


def test_apollo_walk_observation_rejects_non_finite_or_bad_shapes():
    valid = {
        "angular_velocity": np.zeros(3),
        "projected_gravity": np.zeros(3),
        "velocity_command": np.zeros(3),
        "joint_position_offset": np.zeros(23),
        "joint_velocity": np.zeros(23),
        "previous_action": np.zeros(23),
    }
    with pytest.raises(ValueError, match="shapes"):
        apollo_walk_observation(**{**valid, "previous_action": np.zeros(22)})
    with pytest.raises(ValueError, match="non-finite"):
        apollo_walk_observation(
            **{**valid, "angular_velocity": np.full(3, np.nan)}
        )


def test_apollo_walk_cpu_hides_head_state_like_cpp_runner():
    class CaptureSession:
        def __init__(self):
            self.observation = None

        def run(self, unused_outputs, inputs):
            self.observation = inputs["obs"].copy()
            return [np.zeros((1, 23), dtype=np.float32)]

    actor = ApolloWalkCpu.__new__(ApolloWalkCpu)
    actor._joint_qpos = np.arange(23)
    actor._joint_dof = np.arange(23)
    actor._torso_site = 0
    actor._gyro_slice = slice(0, 3)
    actor._input_name = "obs"
    actor._session = CaptureSession()
    data = SimpleNamespace(
        site_xmat=np.eye(3).reshape(1, 9),
        sensordata=np.zeros(3),
        qpos=np.linspace(0.1, 2.3, 23),
        qvel=np.linspace(0.2, 4.6, 23),
    )

    actor.target(data, np.ones(23), np.zeros(3))

    observation = actor._session.observation[0]
    np.testing.assert_array_equal(observation[9:11], np.zeros(2))
    np.testing.assert_array_equal(observation[32:34], np.zeros(2))
    np.testing.assert_array_equal(observation[55:57], np.zeros(2))
