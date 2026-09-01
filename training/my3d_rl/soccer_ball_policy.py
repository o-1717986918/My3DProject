"""Ball/target command boundary appended to the retained soccer-motion actor."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jp
import numpy as np


SOCCER_MOTION_ACTOR_SIZE = 110
SOCCER_BALL_FEATURE_SIZE = 16
SOCCER_BALL_ACTOR_SIZE = SOCCER_MOTION_ACTOR_SIZE + SOCCER_BALL_FEATURE_SIZE
SOCCER_BALL_PRIVILEGED_SIZE = SOCCER_BALL_ACTOR_SIZE + 8

BallActionMode = Literal["pass", "shot", "clear"]
_ACTION_MODE_INDEX = {"pass": 0, "shot": 1, "clear": 2}


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} shape {result.shape} != ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def empty_ball_target_features() -> np.ndarray:
    """Return the neutral extension used for invalid perception and transfer."""
    return np.zeros(SOCCER_BALL_FEATURE_SIZE, dtype=np.float32)


def soccer_ball_target_features(
    *,
    torso_position_world: np.ndarray,
    torso_yaw_rad: float,
    torso_linear_velocity_world: np.ndarray,
    ball_position_world: np.ndarray,
    ball_velocity_world: np.ndarray,
    target_position_world_xy: np.ndarray,
    requested_launch_speed_m_s: float,
    requested_arrival_speed_m_s: float,
    action_mode: BallActionMode,
    observation_age_s: float,
    observation_valid: bool,
) -> np.ndarray:
    """Encode the appended 16-value deployment command in the torso-yaw frame.

    Invalid perception deliberately maps to the all-zero transfer point.  The
    runtime contract additionally requires the caller to reject policy output
    in this state and enter the Apollo walking/search fallback.
    """
    if not observation_valid:
        return empty_ball_target_features()
    torso_position = _finite_vector(
        "torso_position_world", torso_position_world, 3
    )
    torso_velocity = _finite_vector(
        "torso_linear_velocity_world", torso_linear_velocity_world, 3
    )
    ball_position = _finite_vector("ball_position_world", ball_position_world, 3)
    ball_velocity = _finite_vector("ball_velocity_world", ball_velocity_world, 3)
    target_position = _finite_vector(
        "target_position_world_xy", target_position_world_xy, 2
    )
    scalar_values = np.asarray(
        [
            torso_yaw_rad,
            requested_launch_speed_m_s,
            requested_arrival_speed_m_s,
            observation_age_s,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scalar_values).all():
        raise ValueError("ball command scalar contains non-finite values")
    if requested_launch_speed_m_s < 0.0 or requested_arrival_speed_m_s < 0.0:
        raise ValueError("requested ball speeds must be non-negative")
    if observation_age_s < 0.0:
        raise ValueError("ball observation age must be non-negative")
    if action_mode not in _ACTION_MODE_INDEX:
        raise ValueError(f"unsupported ball action mode: {action_mode!r}")

    cosine = float(np.cos(torso_yaw_rad))
    sine = float(np.sin(torso_yaw_rad))
    world_to_yaw = np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ball_position_local = world_to_yaw @ (ball_position - torso_position)
    ball_velocity_local = world_to_yaw @ (ball_velocity - torso_velocity)
    target_delta_world = target_position - ball_position[:2]
    target_distance = float(np.linalg.norm(target_delta_world))
    if target_distance <= 1.0e-6:
        raise ValueError("ball target must differ from the current ball position")
    target_direction_local = (
        world_to_yaw[:2, :2] @ target_delta_world
    ) / target_distance
    mode = np.zeros(3, dtype=np.float64)
    mode[_ACTION_MODE_INDEX[action_mode]] = 1.0
    features = np.concatenate(
        [
            ball_position_local / np.array([6.0, 4.0, 2.0]),
            ball_velocity_local / np.array([15.0, 15.0, 10.0]),
            target_direction_local,
            np.array(
                [
                    target_distance / 20.0,
                    requested_launch_speed_m_s / 15.0,
                    requested_arrival_speed_m_s / 10.0,
                ]
            ),
            mode,
            np.array([min(observation_age_s, 1.0), 1.0]),
        ]
    )
    if features.shape != (SOCCER_BALL_FEATURE_SIZE,):
        raise AssertionError("internal ball feature shape mismatch")
    return np.clip(features, -10.0, 10.0).astype(np.float32)


def append_ball_target_features(
    soccer_motion_actor: np.ndarray, ball_target_features: np.ndarray
) -> np.ndarray:
    """Append the versioned ball command without changing the inherited prefix."""
    actor = _finite_vector(
        "soccer_motion_actor", soccer_motion_actor, SOCCER_MOTION_ACTOR_SIZE
    )
    features = _finite_vector(
        "ball_target_features", ball_target_features, SOCCER_BALL_FEATURE_SIZE
    )
    return np.concatenate([actor, features]).astype(np.float32)


def soccer_ball_target_features_jax(
    *,
    torso_position_world: jax.Array,
    torso_yaw_rad: jax.Array,
    torso_linear_velocity_world: jax.Array,
    ball_position_world: jax.Array,
    ball_velocity_world: jax.Array,
    target_position_world_xy: jax.Array,
    requested_launch_speed_m_s: jax.Array,
    requested_arrival_speed_m_s: jax.Array,
    action_mode_one_hot: jax.Array,
    observation_age_s: jax.Array,
    observation_valid: jax.Array,
) -> jax.Array:
    """JAX form of the exact same 16-value deployment feature contract."""
    cosine, sine = jp.cos(torso_yaw_rad), jp.sin(torso_yaw_rad)
    world_to_yaw = jp.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    ball_position_local = world_to_yaw @ (
        ball_position_world - torso_position_world
    )
    ball_velocity_local = world_to_yaw @ (
        ball_velocity_world - torso_linear_velocity_world
    )
    target_delta_world = target_position_world_xy - ball_position_world[:2]
    target_distance = jp.linalg.norm(target_delta_world)
    target_direction_local = (
        world_to_yaw[:2, :2] @ target_delta_world
    ) / jp.maximum(target_distance, 1.0e-6)
    features = jp.concatenate(
        [
            ball_position_local / jp.array([6.0, 4.0, 2.0]),
            ball_velocity_local / jp.array([15.0, 15.0, 10.0]),
            target_direction_local,
            jp.array(
                [
                    target_distance / 20.0,
                    requested_launch_speed_m_s / 15.0,
                    requested_arrival_speed_m_s / 10.0,
                ]
            ),
            action_mode_one_hot,
            jp.array([jp.minimum(observation_age_s, 1.0), 1.0]),
        ]
    )
    features = jp.clip(
        jp.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0),
        -10.0,
        10.0,
    )
    return jp.where(observation_valid, features, jp.zeros_like(features))
