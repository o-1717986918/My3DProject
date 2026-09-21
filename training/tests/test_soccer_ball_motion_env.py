from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from my3d_rl.soccer_ball_motion_env import (
    BallConditionedSoccerMotionTracking,
    default_config,
    post_contact_ball_reward,
)
from my3d_rl.soccer_motion_corpus import SoccerMotionCorpus
from my3d_rl.t1_control import APOLLO_DEFAULT_POSE


def _synthetic_corpus() -> SoccerMotionCorpus:
    motions = 13
    frames = 140
    root_position = np.zeros((motions, frames, 3), dtype=np.float32)
    root_position[:, :, 0] = np.linspace(0.0, 0.35, frames)
    root_position[:, :, 2] = 0.70
    root_quaternion = np.zeros((motions, frames, 4), dtype=np.float32)
    root_quaternion[:, :, 0] = 1.0
    joint_position = np.broadcast_to(
        np.asarray(APOLLO_DEFAULT_POSE, dtype=np.float32),
        (motions, frames, 23),
    ).copy()
    reset_weights = np.ones((motions, frames), dtype=np.float32)
    reset_weights[:, -1] = 0.0
    reset_weights /= reset_weights.sum(axis=1, keepdims=True)
    kick_leg = np.zeros((motions, 2), dtype=np.float32)
    kick_leg[:, 1] = 1.0
    return SoccerMotionCorpus(
        relative_paths=tuple(f"motion-{index}.npz" for index in range(motions)),
        sha256=tuple("0" * 64 for _ in range(motions)),
        lengths=np.full(motions, frames, dtype=np.int32),
        root_position=root_position,
        root_quaternion_wxyz=root_quaternion,
        root_linear_velocity=np.zeros((motions, frames, 3), dtype=np.float32),
        root_angular_velocity=np.zeros((motions, frames, 3), dtype=np.float32),
        joint_position=joint_position,
        joint_velocity=np.zeros((motions, frames, 23), dtype=np.float32),
        foot_contact=np.ones((motions, frames, 2), dtype=bool),
        kick_leg_one_hot=kick_leg,
        reset_weights=reset_weights,
    )


def test_k2_environment_has_finite_126_and_134_boundaries():
    env = BallConditionedSoccerMotionTracking(
        _synthetic_corpus(),
        config_overrides={
            "impl": "jax",
            "episode_length": 2,
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
        },
        prefix="test_k2_ball_",
    )

    state = env.reset(jax.random.PRNGKey(20260989))

    assert int(state.info["motion"]) == 12
    assert 113 <= int(state.info["reference_frame"]) <= 118
    assert state.obs["state"].shape == (126,)
    assert state.obs["privileged_state"].shape == (134,)
    assert float(state.metrics["event/ball_target_success"]) == 0.0
    assert not bool(state.info["post_contact_fell"])
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    np.testing.assert_allclose(
        np.asarray(state.obs["state"][-2:]), [0.0, 1.0], atol=1.0e-7
    )

    stepped = env.step(state, jp.zeros(env.action_size))
    assert int(stepped.info["step"]) == 1
    assert np.isfinite(float(stepped.reward))
    assert np.isfinite(np.asarray(stepped.obs["state"])).all()


def test_k2_reward_separates_lateral_ball_motion_from_forward_progress():
    config = default_config()

    assert config.target_progress_reward_scale > 0.0
    assert config.launch_speed_reward_scale > 0.0
    assert config.lateral_speed_cost > 0.0
    assert config.post_contact_upright_reward_scale > 0.0
    assert config.post_contact_fall_cost < config.success_event_reward

    common = {
        "target_progress_rate": jp.array(2.0),
        "directional_speed": jp.array(1.0),
        "requested_launch_speed": jp.array(1.0),
        "upright": jp.array(1.0),
        "post_contact_fall_event": jp.array(False),
        "dt": 0.02,
        "target_progress_reward_scale": config.target_progress_reward_scale,
        "launch_speed_reward_scale": config.launch_speed_reward_scale,
        "lateral_speed_cost": config.lateral_speed_cost,
        "post_contact_upright_reward_scale": (
            config.post_contact_upright_reward_scale
        ),
        "post_contact_fall_cost": config.post_contact_fall_cost,
    }
    straight = post_contact_ball_reward(lateral_speed=jp.array(0.0), **common)
    diagonal = post_contact_ball_reward(lateral_speed=jp.array(1.0), **common)

    assert float(straight - diagonal) == pytest.approx(
        config.lateral_speed_cost * common["dt"]
    )


def test_k2_exposes_one_shot_outcome_metrics():
    env = BallConditionedSoccerMotionTracking(
        _synthetic_corpus(),
        config_overrides={
            "impl": "jax",
            "episode_length": 2,
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
        },
        prefix="test_k2_outcome_",
    )

    state = env.reset(jax.random.PRNGKey(20260993))

    for key in (
        "cost/ball_lateral_speed",
        "event/post_contact_fall",
        "event/ball_outcome_terminal",
        "outcome/final_ball_progress_m",
        "outcome/final_lateral_error_m",
        "outcome/final_target_distance_m",
        "outcome/minimum_target_distance_m",
        "outcome/maximum_ball_progress_m",
        "outcome/final_directional_speed_m_s",
    ):
        assert key in state.metrics
        assert float(state.metrics[key]) == 0.0
