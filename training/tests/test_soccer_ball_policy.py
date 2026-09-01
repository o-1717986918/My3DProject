from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from my3d_rl.contract import load_policy_contract
from my3d_rl.soccer_ball_policy import (
    SOCCER_BALL_ACTOR_SIZE,
    append_ball_target_features,
    empty_ball_target_features,
    soccer_ball_target_features,
)


CONTRACT = (
    Path(__file__).parents[1]
    / "contracts"
    / "soccer_ball_motion_policy_v1.yaml"
)


def test_ball_policy_contract_is_append_only_126_to_23():
    contract = load_policy_contract(CONTRACT)

    assert contract.policy_name == "soccer_ball_motion_policy_v1"
    assert contract.observation_size == SOCCER_BALL_ACTOR_SIZE == 126
    assert contract.action_size == 23
    assert sum(size for _, size in contract.observation_fields[:9]) == 110
    assert sum(size for _, size in contract.observation_fields[9:]) == 16


def test_ball_target_features_use_torso_yaw_frame_and_declared_scales():
    features = soccer_ball_target_features(
        torso_position_world=np.array([1.0, 2.0, 0.5]),
        torso_yaw_rad=np.pi / 2.0,
        torso_linear_velocity_world=np.array([0.0, 1.0, 0.0]),
        ball_position_world=np.array([1.0, 3.0, 0.1]),
        ball_velocity_world=np.array([2.0, 1.0, 0.0]),
        target_position_world_xy=np.array([2.0, 3.0]),
        requested_launch_speed_m_s=7.5,
        requested_arrival_speed_m_s=2.5,
        action_mode="pass",
        observation_age_s=0.2,
        observation_valid=True,
    )

    np.testing.assert_allclose(features[:3], [1.0 / 6.0, 0.0, -0.2], atol=1e-7)
    np.testing.assert_allclose(features[3:6], [0.0, -2.0 / 15.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(features[6:8], [0.0, -1.0], atol=1e-7)
    np.testing.assert_allclose(features[8:11], [0.05, 0.5, 0.25], atol=1e-7)
    np.testing.assert_array_equal(features[11:14], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(features[14:], [0.2, 1.0], atol=1e-7)


def test_invalid_ball_observation_is_the_neutral_transfer_point():
    features = soccer_ball_target_features(
        torso_position_world=np.full(3, np.nan),
        torso_yaw_rad=np.nan,
        torso_linear_velocity_world=np.full(3, np.nan),
        ball_position_world=np.full(3, np.nan),
        ball_velocity_world=np.full(3, np.nan),
        target_position_world_xy=np.full(2, np.nan),
        requested_launch_speed_m_s=np.nan,
        requested_arrival_speed_m_s=np.nan,
        action_mode="pass",
        observation_age_s=np.nan,
        observation_valid=False,
    )

    np.testing.assert_array_equal(features, empty_ball_target_features())


def test_append_preserves_inherited_actor_prefix():
    actor = np.linspace(-1.0, 1.0, 110, dtype=np.float32)
    features = np.linspace(0.0, 1.0, 16, dtype=np.float32)

    result = append_ball_target_features(actor, features)

    np.testing.assert_array_equal(result[:110], actor)
    np.testing.assert_array_equal(result[110:], features)
    with pytest.raises(ValueError, match="shape"):
        append_ball_target_features(actor[:-1], features)
