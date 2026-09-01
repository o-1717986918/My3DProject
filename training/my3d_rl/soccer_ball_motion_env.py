"""K2 target-conditioned ball outcomes above the retained K1-D motion actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

from .apollo_walk_jax import load_apollo_walk_jax
from .contract import PolicyContract, load_policy_contract
from .rcss_scene import DEFAULT_RESOURCE_ROOT
from .soccer_ball_policy import (
    SOCCER_BALL_ACTOR_SIZE,
    SOCCER_BALL_FEATURE_SIZE,
    SOCCER_BALL_PRIVILEGED_SIZE,
    soccer_ball_target_features_jax,
)
from .soccer_motion_corpus import SoccerMotionCorpus
from .soccer_motion_env import (
    FiniteSoccerMotionTracking,
    default_config as motion_default_config,
)
from .t1_control import APOLLO_DEFAULT_POSE


DEFAULT_CONTRACT = (
    Path(__file__).parents[1]
    / "contracts"
    / "soccer_ball_motion_policy_v1.yaml"
)
DEFAULT_WALK_POLICY = (
    Path(__file__).parents[2]
    / "runtime"
    / "apollo"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)


def default_config() -> config_dict.ConfigDict:
    """Return the locked fixed-2 m K2-B first curriculum."""
    config = motion_default_config()
    config.episode_length = 120
    config.fixed_motion_index = 12
    config.fixed_start_frame_min = 113
    config.fixed_start_frame_max = 118
    config.reset_joint_noise = 0.001
    config.reset_root_velocity_noise = 0.003
    config.reset_yaw_range = 0.005
    config.ball_radius_noise_m = 0.0
    config.ball_arc_noise_rad = 0.0
    config.target_angle_range = [0.0, 0.0]
    config.target_distance_range = [2.0, 2.0]
    config.requested_arrival_speed_m_s = 0.8
    config.rolling_deceleration_m_s2 = 0.08
    config.post_contact_recovery_steps = 50
    config.target_radius_m = 0.5
    config.lateral_tolerance_m = 0.5
    config.arrival_speed_tolerance_m_s = 0.5
    config.contact_event_reward = 5.0
    config.success_event_reward = 30.0
    config.miss_event_cost = 20.0
    config.wrong_foot_cost = 20.0
    config.post_contact_fall_cost = 100.0
    return config


def mjx_ball_foot_contacts(
    contact_geom: jax.Array,
    contact_distance: jax.Array,
    *,
    ball_geom: int,
    left_foot_geom: int,
    right_foot_geom: int,
) -> tuple[jax.Array, jax.Array]:
    """Return active exact MJX ball-left and ball-right contact flags."""
    pairs = jp.asarray(contact_geom)
    distances = jp.asarray(contact_distance)
    active = distances <= 0.0

    def pair_contact(other: int) -> jax.Array:
        pair = (
            ((pairs[:, 0] == ball_geom) & (pairs[:, 1] == other))
            | ((pairs[:, 1] == ball_geom) & (pairs[:, 0] == other))
        )
        return jp.any(active & pair)

    return pair_contact(left_foot_geom), pair_contact(right_foot_geom)


class BallConditionedSoccerMotionTracking(FiniteSoccerMotionTracking):
    """Learn pre-contact outcome corrections; Apollo owns post-contact balance."""

    def __init__(
        self,
        corpus: SoccerMotionCorpus,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
        walk_policy_path: Path = DEFAULT_WALK_POLICY,
    ) -> None:
        contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        if contract.policy_name != "soccer_ball_motion_policy_v1":
            raise ValueError(
                "BallConditionedSoccerMotionTracking requires the K2 contract"
            )
        config = default_config() if config is None else config
        super().__init__(
            corpus,
            config=config,
            config_overrides=config_overrides,
            contract=contract,
            resource_root=resource_root,
            prefix=prefix,
        )
        if self.contract.observation_size != SOCCER_BALL_ACTOR_SIZE:
            raise ValueError("K2 actor size differs from its feature encoder")
        if not (
            0
            <= self._config.fixed_motion_index
            < self.corpus.motion_count
            and self._config.fixed_start_frame_max
            < self.corpus.lengths[self._config.fixed_motion_index] - 1
        ):
            raise ValueError("K2 fixed motion/start window is outside the corpus")
        scalar_positive = (
            self._config.requested_arrival_speed_m_s,
            self._config.rolling_deceleration_m_s2,
            self._config.target_radius_m,
            self._config.lateral_tolerance_m,
            self._config.arrival_speed_tolerance_m_s,
        )
        if any(value <= 0.0 for value in scalar_positive):
            raise ValueError("K2 ball outcome thresholds must be positive")
        if self._config.post_contact_recovery_steps < 1:
            raise ValueError("post-contact recovery must contain at least one step")
        if (
            len(self._config.target_angle_range) != 2
            or len(self._config.target_distance_range) != 2
            or self._config.target_angle_range[0]
            > self._config.target_angle_range[1]
            or self._config.target_distance_range[0]
            > self._config.target_distance_range[1]
            or self._config.target_distance_range[0] <= 0.0
        ):
            raise ValueError("K2 target curriculum ranges are invalid")

        self._ball_qpos = self._mj_model.joint("ball-root").qposadr[0]
        self._ball_dof = self._mj_model.joint("ball-root").dofadr[0]
        self._ball_body = self._mj_model.body("ball").id
        self._ball_geom = self._mj_model.geom("ball").id
        self._left_foot_geom = self._mj_model.geom(prefix + "left_foot").id
        self._right_foot_geom = self._mj_model.geom(prefix + "right_foot").id
        self._default_pose = jp.asarray(APOLLO_DEFAULT_POSE)
        self._walk_policy = load_apollo_walk_jax(walk_policy_path)

    def _step_targets(
        self,
        state: mjx_env.State,
        applied_action: jax.Array,
        motion: jax.Array,
        frame: jax.Array,
    ) -> jax.Array:
        motion_targets = self.decode_action_targets(
            applied_action, motion, frame
        )
        torso_xmat = state.data.site_xmat[self._torso_site]
        gravity = torso_xmat.T @ jp.array([0.0, 0.0, -1.0])
        walk_observation = jp.concatenate(
            [
                state.data.sensordata[self._gyro_slice],
                gravity,
                jp.zeros(3),
                state.data.qpos[self._joint_qpos] - self._default_pose,
                state.data.qvel[self._joint_dof],
                state.info["walk_last_action"],
            ]
        )
        walk_action = self._walk_policy(walk_observation)
        recovery_targets = jp.clip(
            self._default_pose + 0.25 * walk_action,
            self._lowers,
            self._uppers,
        )
        recovery_active = state.info["contacted"]
        state.info["walk_last_action"] = jp.where(
            recovery_active, walk_action, state.info["walk_last_action"]
        )
        return jp.where(recovery_active, recovery_targets, motion_targets)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        rng, radius_rng, arc_rng, target_rng, distance_rng = jax.random.split(
            state.info["rng"], 5
        )
        motion = state.info["motion"]
        frame = state.info["reference_frame"]
        length = self._lengths[motion]
        remaining = (
            self._root_position[motion, length - 1, :2]
            - self._root_position[motion, frame, :2]
        )
        base_radius = jp.linalg.norm(remaining)
        base_angle = jp.arctan2(remaining[1], remaining[0])
        radius = base_radius + jax.random.uniform(
            radius_rng,
            minval=-self._config.ball_radius_noise_m,
            maxval=self._config.ball_radius_noise_m,
        )
        arc = jax.random.uniform(
            arc_rng,
            minval=-self._config.ball_arc_noise_rad,
            maxval=self._config.ball_arc_noise_rad,
        )
        ball_heading = base_angle + state.info["yaw"] + arc
        ball_xy = self._model_root_xy + radius * jp.array(
            [jp.cos(ball_heading), jp.sin(ball_heading)]
        )
        ball_position = jp.array([ball_xy[0], ball_xy[1], 0.11])
        qpos = state.data.qpos.at[self._ball_qpos : self._ball_qpos + 3].set(
            ball_position
        )
        qvel = state.data.qvel.at[self._ball_dof : self._ball_dof + 6].set(0.0)
        data = state.data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        target_angle = jax.random.uniform(
            target_rng,
            minval=self._config.target_angle_range[0],
            maxval=self._config.target_angle_range[1],
        )
        target_heading = state.info["yaw"] + target_angle
        target_direction = jp.array(
            [jp.cos(target_heading), jp.sin(target_heading)]
        )
        target_distance = jax.random.uniform(
            distance_rng,
            minval=self._config.target_distance_range[0],
            maxval=self._config.target_distance_range[1],
        )
        goal_world = ball_xy + target_distance * target_direction
        requested_launch_speed = jp.sqrt(
            self._config.requested_arrival_speed_m_s**2
            + 2.0
            * self._config.rolling_deceleration_m_s2
            * target_distance
        )
        state.info.update(
            {
                "rng": rng,
                "walk_last_action": jp.zeros(self.action_size),
                "initial_ball_xy": ball_xy,
                "goal_world": goal_world,
                "target_direction_world": target_direction,
                "target_distance_command": target_distance,
                "requested_launch_speed": requested_launch_speed,
                "requested_arrival_speed": jp.asarray(
                    self._config.requested_arrival_speed_m_s
                ),
                "action_mode": jp.array([1.0, 0.0, 0.0]),
                "ball_observation_age": jp.array(0.0),
                "ball_observation_valid": jp.array(True),
                "contacted": jp.array(False),
                "wrong_foot_contacted": jp.array(False),
                "post_contact_steps": jp.array(0, dtype=jp.int32),
                "last_target_distance": target_distance,
                "minimum_target_distance": target_distance,
                "maximum_ball_progress": jp.array(0.0),
            }
        )
        state.metrics.update(
            {
                "event/correct_foot_contact": jp.array(0.0),
                "event/wrong_foot_contact": jp.array(0.0),
                "event/recovery_complete": jp.array(0.0),
                "event/target_success": jp.array(0.0),
                "reward/ball_target_progress": jp.array(0.0),
                "reward/ball_speed": jp.array(0.0),
                "cost/target_error": jp.array(0.0),
                "diagnostic/ball_progress": jp.array(0.0),
                "diagnostic/target_distance": target_distance,
                "diagnostic/lateral_error": jp.array(0.0),
                "diagnostic/arrival_speed_error": jp.array(0.0),
            }
        )
        return state.replace(data=data, obs=self._get_obs(data, state.info))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        was_contacted = state.info["contacted"]
        was_wrong_contacted = state.info["wrong_foot_contacted"]
        last_target_distance = state.info["last_target_distance"]
        result = super().step(state, action)
        contact = result.data._impl.contact
        left_contact, right_contact = mjx_ball_foot_contacts(
            contact.geom,
            contact.dist,
            ball_geom=self._ball_geom,
            left_foot_geom=self._left_foot_geom,
            right_foot_geom=self._right_foot_geom,
        )
        kick_leg = self._kick_leg[result.info["motion"]]
        correct_contact = jp.where(kick_leg[0] > 0.5, left_contact, right_contact)
        wrong_contact = jp.where(kick_leg[0] > 0.5, right_contact, left_contact)
        contact_event = correct_contact & ~was_contacted
        wrong_contact_event = wrong_contact & ~was_wrong_contacted
        contacted = was_contacted | correct_contact
        wrong_contacted = was_wrong_contacted | wrong_contact

        ball_position = result.data.xpos[self._ball_body]
        ball_velocity = result.data.qvel[self._ball_dof : self._ball_dof + 3]
        displacement = ball_position[:2] - result.info["initial_ball_xy"]
        ball_progress = jp.dot(
            displacement, result.info["target_direction_world"]
        )
        lateral_error = jp.abs(
            result.info["target_direction_world"][0] * displacement[1]
            - result.info["target_direction_world"][1] * displacement[0]
        )
        target_distance = jp.linalg.norm(
            result.info["goal_world"] - ball_position[:2]
        )
        directional_speed = jp.dot(
            ball_velocity[:2], result.info["target_direction_world"]
        )
        arrival_speed_error = jp.abs(
            directional_speed - result.info["requested_arrival_speed"]
        )
        target_progress_rate = jp.clip(
            (last_target_distance - target_distance) / self.dt, -6.0, 6.0
        )
        speed_tracking = jp.exp(
            -jp.square(
                directional_speed - result.info["requested_launch_speed"]
            )
        )
        post_contact_steps = jp.where(
            contacted, result.info["post_contact_steps"] + 1, 0
        )
        recovery_complete = contacted & (
            post_contact_steps >= self._config.post_contact_recovery_steps
        )
        fall = result.metrics["cost/fall"] > 0.5
        invalid = (
            jp.isnan(result.data.qpos).any()
            | jp.isnan(result.data.qvel).any()
            | jp.isnan(action).any()
        )
        target_success = (
            recovery_complete
            & ~fall
            & (target_distance <= self._config.target_radius_m)
            & (lateral_error <= self._config.lateral_tolerance_m)
            & (
                arrival_speed_error
                <= self._config.arrival_speed_tolerance_m_s
            )
        )
        timeout = result.info["step"] >= self._config.episode_length

        inherited_completion_bonus = (
            self._config.reward.completion
            * result.metrics["reward/completion"]
        )
        pre_contact_reward = (
            result.reward
            - inherited_completion_bonus
            + self._config.contact_event_reward
            * contact_event.astype(jp.float32)
            - self._config.wrong_foot_cost
            * wrong_contact_event.astype(jp.float32)
        )
        post_contact_reward = (
            4.0 * target_progress_rate * self.dt
            + 0.5 * speed_tracking * self.dt
            + 0.2 * jp.clip(result.metrics["reward/upright"], 0.0, 1.0) * self.dt
            - self._config.post_contact_fall_cost * fall.astype(jp.float32)
        )
        terminal_reward = jp.where(
            recovery_complete,
            jp.where(
                target_success,
                self._config.success_event_reward,
                -self._config.miss_event_cost,
            ),
            0.0,
        )
        reward = jp.where(
            was_contacted, post_contact_reward, pre_contact_reward
        ) + terminal_reward

        result.info["contacted"] = contacted
        result.info["wrong_foot_contacted"] = wrong_contacted
        result.info["post_contact_steps"] = post_contact_steps
        result.info["last_target_distance"] = target_distance
        result.info["minimum_target_distance"] = jp.minimum(
            result.info["minimum_target_distance"], target_distance
        )
        result.info["maximum_ball_progress"] = jp.maximum(
            result.info["maximum_ball_progress"], ball_progress
        )
        result.metrics.update(
            {
                "event/correct_foot_contact": contact_event.astype(jp.float32),
                "event/wrong_foot_contact": wrong_contact_event.astype(jp.float32),
                "event/recovery_complete": recovery_complete.astype(jp.float32),
                "event/target_success": target_success.astype(jp.float32),
                "reward/ball_target_progress": target_progress_rate,
                "reward/ball_speed": speed_tracking,
                "cost/target_error": target_distance,
                "diagnostic/ball_progress": ball_progress,
                "diagnostic/target_distance": target_distance,
                "diagnostic/lateral_error": lateral_error,
                "diagnostic/arrival_speed_error": arrival_speed_error,
            }
        )
        done = fall | invalid | recovery_complete | timeout
        obs = self._get_obs(result.data, result.info)
        return result.replace(
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        inherited = FiniteSoccerMotionTracking._get_obs(self, data, info)
        inherited_actor = inherited["state"]
        if "goal_world" not in info:
            features = jp.zeros(SOCCER_BALL_FEATURE_SIZE)
        else:
            torso_xmat = data.site_xmat[self._torso_site]
            torso_yaw = jp.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
            features = soccer_ball_target_features_jax(
                torso_position_world=data.xpos[self._torso_body],
                torso_yaw_rad=torso_yaw,
                torso_linear_velocity_world=data.qvel[
                    self._root_dof : self._root_dof + 3
                ],
                ball_position_world=data.xpos[self._ball_body],
                ball_velocity_world=data.qvel[
                    self._ball_dof : self._ball_dof + 3
                ],
                target_position_world_xy=info["goal_world"],
                requested_launch_speed_m_s=info["requested_launch_speed"],
                requested_arrival_speed_m_s=info["requested_arrival_speed"],
                action_mode_one_hot=info["action_mode"],
                observation_age_s=info["ball_observation_age"],
                observation_valid=info["ball_observation_valid"],
            )
        actor = jp.concatenate([inherited_actor, features])
        root_velocity = data.qvel[self._root_dof : self._root_dof + 6]
        torso_xmat = data.site_xmat[self._torso_site]
        privileged = jp.concatenate(
            [
                actor,
                root_velocity,
                jp.array(
                    [torso_xmat[2, 2], data.xpos[self._torso_body, 2]]
                ),
            ]
        )
        if actor.shape != (SOCCER_BALL_ACTOR_SIZE,):
            raise ValueError("K2 actor observation has an incompatible shape")
        if privileged.shape != (SOCCER_BALL_PRIVILEGED_SIZE,):
            raise ValueError("K2 critic observation has an incompatible shape")
        return {"state": actor, "privileged_state": privileged}
