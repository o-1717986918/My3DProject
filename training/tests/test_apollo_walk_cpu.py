from __future__ import annotations

import numpy as np
import pytest

from my3d_rl.apollo_walk_cpu import apollo_walk_observation


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
