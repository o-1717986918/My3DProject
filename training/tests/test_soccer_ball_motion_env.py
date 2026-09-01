from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.soccer_ball_motion_env import (
    BallConditionedSoccerMotionTracking,
    mjx_ball_foot_contacts,
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


def test_mjx_ball_contact_flags_ignore_inactive_candidates():
    geom = jp.array([[1, 8], [9, 1], [1, 9], [0, 1]])
    distance = jp.array([-0.01, 0.02, -0.03, -0.1])

    left, right = mjx_ball_foot_contacts(
        geom,
        distance,
        ball_geom=1,
        left_foot_geom=8,
        right_foot_geom=9,
    )

    assert bool(left)
    assert bool(right)


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
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    np.testing.assert_allclose(
        np.asarray(state.obs["state"][-2:]), [0.0, 1.0], atol=1.0e-7
    )

    stepped = env.step(state, jp.zeros(env.action_size))
    assert int(stepped.info["step"]) == 1
    assert np.isfinite(float(stepped.reward))
    assert np.isfinite(np.asarray(stepped.obs["state"])).all()
