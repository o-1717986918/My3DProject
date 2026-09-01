"""Shared deployable actor boundary for finite soccer-motion policies."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from .ppo_profile import get_ppo_profile


SoccerMotionPolicy = Callable[[np.ndarray], np.ndarray]
SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE = 118


def _validated_actor_observation(
    actor: np.ndarray, observation_size: int
) -> np.ndarray:
    value = np.asarray(actor, dtype=np.float32)
    if value.shape != (observation_size,):
        raise ValueError(
            f"soccer-motion actor observation {value.shape} != "
            f"({observation_size},)"
        )
    if not np.isfinite(value).all():
        raise ValueError("soccer-motion actor observation contains non-finite values")
    return value


def soccer_motion_actor_observation(
    data: mujoco.MjData,
    *,
    joint_qpos: np.ndarray,
    joint_dof: np.ndarray,
    gyro_slice: slice,
    torso_site: int,
    reference_joint_position: np.ndarray,
    reference_joint_velocity: np.ndarray,
    reference_root_linear_velocity: np.ndarray,
    reference_root_angular_velocity: np.ndarray,
    reference_contact: np.ndarray,
    previous_action: np.ndarray,
    progress: float,
    kick_leg_one_hot: np.ndarray,
) -> np.ndarray:
    """Encode the versioned 110-value finite-motion actor observation."""
    triplets = np.stack(
        [
            (data.qpos[joint_qpos] - reference_joint_position) / 4.6,
            (data.qvel[joint_dof] - reference_joint_velocity) / 110.0,
            previous_action / 10.0,
        ],
        axis=1,
    ).reshape(-1)
    angular_velocity = data.sensordata[gyro_slice] / 50.0
    gravity = data.site_xmat[torso_site].reshape(3, 3).T @ np.array(
        [0.0, 0.0, -1.0]
    )
    angle = 2.0 * np.pi * progress
    actor = np.concatenate(
        [
            triplets,
            angular_velocity,
            gravity,
            reference_joint_position / 4.6,
            reference_root_linear_velocity / 5.0,
            reference_root_angular_velocity / 10.0,
            reference_contact.astype(np.float64),
            np.array([np.cos(angle), np.sin(angle)]),
            kick_leg_one_hot,
        ]
    )
    if actor.shape != (110,):
        raise ValueError(f"soccer-motion actor observation {actor.shape} != (110,)")
    return np.clip(np.nan_to_num(actor), -10.0, 10.0).astype(np.float32)


def load_soccer_motion_policy(
    *,
    zero_policy: bool,
    checkpoint: Path | None,
    profile_name: str,
    policy_contract_name: str,
    observation_size: int,
    action_size: int,
) -> SoccerMotionPolicy:
    """Load one deterministic PPO actor or the explicit zero residual."""
    if zero_policy == (checkpoint is not None):
        raise ValueError("select exactly one of zero policy or checkpoint")
    profile = get_ppo_profile(profile_name)
    if profile.policy_contract != policy_contract_name:
        raise ValueError("PPO profile and policy contract differ")
    if zero_policy:
        def zero(actor: np.ndarray) -> np.ndarray:
            _validated_actor_observation(actor, observation_size)
            return np.zeros(action_size, dtype=np.float32)

        return zero
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"soccer-motion checkpoint does not exist: {checkpoint}")
    preprocess = (
        running_statistics.normalize
        if profile.normalize_observations
        else brax_types.identity_observation_preprocessor
    )
    networks = profile.network_factory()(
        {
            "state": observation_size,
            "privileged_state": SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE,
        },
        action_size,
        preprocess_observations_fn=preprocess,
    )
    params = ppo_checkpoint.load(checkpoint)
    inference = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    @jax.jit
    def infer(actor: jax.Array) -> jax.Array:
        return inference({"state": actor}, jax.random.PRNGKey(0))[0]

    def policy(actor: np.ndarray) -> np.ndarray:
        validated = _validated_actor_observation(actor, observation_size)
        action = np.asarray(infer(jp.asarray(validated)), dtype=np.float64)
        if action.shape != (action_size,) or not np.isfinite(action).all():
            raise ValueError("soccer-motion policy returned an invalid action")
        return action

    return policy
