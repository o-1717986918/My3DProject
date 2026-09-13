"""Apollo-initialized standing goalkeeper block task.

The actor keeps the deployed 78 -> 23 Walk network frozen and adds a zero-output
residual adapter over the original observation plus six normalized incoming-ball
values.  The initial policy is therefore exactly equal to Apollo Walk while PPO
can learn shot-dependent timing and body shape.  Controlled dives and recovery
remain separate stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

from .apollo_run_env import ApolloRuntimeRun, default_config as apollo_default_config
from .contract import PolicyContract, load_policy_contract
from .rcss_scene import DEFAULT_RESOURCE_ROOT


DEFAULT_CONTRACT = (
    Path(__file__).parents[1] / "contracts" / "apollo_goalkeeper_policy_v1.yaml"
)


def default_config() -> config_dict.ConfigDict:
    config = apollo_default_config()
    # The server fixture's rolling ball needs roughly five simulated seconds
    # to travel from x=-18 m to the goal line.  Six seconds resolves the same
    # low-speed tail without shortening the shot into an easier task.
    config.episode_length = 300
    config.use_fixed_command = True
    config.fixed_command = [0.0, 0.0, 0.0]
    config.gait_frequency = [1.5, 1.5]
    config.stand_probability = 0.0
    config.reset_joint_noise = 0.01
    config.reset_joint_velocity_noise = 0.0
    config.reset_policy_action_noise = 0.0
    config.reset_root_velocity_noise = 0.02
    config.reset_yaw_range = 0.03
    config.push_enable = False
    config.action_delay_max_steps = 0

    # RCSSServerMJ evidence currently puts the missed-shot region between
    # 1.2--1.8 m lateral displacement at 5--7 m/s.  Keep the model-local shot
    # distance close to the server fixture's 9 m approach.
    config.shot_speed_range = [5.0, 7.0]
    config.shot_lateral_range = [1.2, 1.8]
    config.shot_start_distance_range = [8.0, 9.0]
    config.fixed_shot_side = 0
    config.goal_plane_offset_m = -0.50
    config.goal_half_width_m = 1.83
    config.lateral_command_mps = 0.50
    config.contact_velocity_delta_mps = 0.25

    # Preserve the warm-started gait while giving direct football outcomes
    # enough weight to dominate small locomotion shaping terms.
    config.reward.tracking_linear = 1.0
    config.reward.tracking_yaw = 1.0
    config.reward.upright = 1.0
    config.reward.height = 1.0
    config.reward.alive = 0.2
    config.reward.vertical_velocity = -0.25
    config.reward.angular_xy = -0.20
    config.reward.action_rate = -0.02
    config.reward.action_acceleration = -0.01
    config.reward.joint_velocity = -0.00005
    config.reward.pose = -0.01
    config.reward.fall = -120.0
    config.reward.goalkeeper_progress = 5.0
    config.reward.goalkeeper_coverage = 2.0
    config.reward.goalkeeper_contact = 8.0
    config.reward.goalkeeper_save = 30.0
    config.reward.goalkeeper_concede = -30.0
    return config


class ApolloGoalkeeperBlock(ApolloRuntimeRun):
    """Train an upright left/right block from the frozen Apollo Walk actor."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root=DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
    ) -> None:
        contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        super().__init__(
            config=default_config() if config is None else config,
            config_overrides=config_overrides,
            contract=contract,
            resource_root=resource_root,
            prefix=prefix,
        )
        for name in (
            "shot_speed_range",
            "shot_lateral_range",
            "shot_start_distance_range",
        ):
            values = np.asarray(self._config[name], dtype=np.float64)
            if (
                values.shape != (2,)
                or not np.isfinite(values).all()
                or values[0] <= 0.0
                or values[0] > values[1]
            ):
                raise ValueError(f"{name} must be a positive increasing pair")
        if int(self._config.fixed_shot_side) not in (-1, 0, 1):
            raise ValueError("fixed_shot_side must be -1, 0, or 1")
        if not self._config.goal_plane_offset_m < 0.0:
            raise ValueError("goal_plane_offset_m must place the goal behind the keeper")
        if self._config.goal_half_width_m <= 0.0:
            raise ValueError("goal_half_width_m must be positive")
        if not 0.0 < self._config.lateral_command_mps <= 0.5:
            raise ValueError("lateral_command_mps must be in (0, 0.5]")
        if self._config.contact_velocity_delta_mps <= 0.0:
            raise ValueError("contact_velocity_delta_mps must be positive")

        self._ball_qpos = self._mj_model.joint("ball-root").qposadr[0]
        self._ball_dof = self._mj_model.joint("ball-root").dofadr[0]
        self._ball_body = self._mj_model.body("ball").id

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> dict[str, jax.Array]:
        observation = super()._get_obs(data, info)
        torso_position = data.site_xpos[self._torso_site]
        torso_rotation = data.site_xmat[self._torso_site]
        ball_position = data.xpos[self._ball_body]
        ball_velocity = data.qvel[self._ball_dof : self._ball_dof + 3]
        root_velocity = data.qvel[self._root_dof : self._root_dof + 3]
        local_ball_position = torso_rotation.T @ (ball_position - torso_position)
        local_ball_velocity = torso_rotation.T @ (ball_velocity - root_velocity)
        goal_plane_x = torso_position[0] + self._config.goal_plane_offset_m
        time_to_plane = jp.maximum(
            (ball_position[0] - goal_plane_x)
            / jp.maximum(-ball_velocity[0], 1.0e-3),
            0.0,
        )
        actor_task = jp.array(
            [
                local_ball_position[0] / 10.0,
                local_ball_position[1] / 2.0,
                local_ball_position[2],
                local_ball_velocity[0] / 8.0,
                local_ball_velocity[1] / 8.0,
                jp.clip(time_to_plane / 4.0, 0.0, 1.0),
            ]
        )
        privileged_task = jp.array(
            [
                local_ball_position[0],
                local_ball_position[1],
                local_ball_velocity[0],
                local_ball_velocity[1],
                time_to_plane,
                info["command"][1],
            ]
        )
        observation["state"] = jp.clip(
            jp.nan_to_num(
                jp.concatenate([observation["state"], actor_task]),
                nan=0.0,
                posinf=10.0,
                neginf=-10.0,
            ),
            -10.0,
            10.0,
        )
        observation["privileged_state"] = jp.concatenate(
            [observation["privileged_state"], privileged_task]
        )
        return observation

    def reset(self, rng: jax.Array) -> mjx_env.State:
        base_rng, speed_rng, lateral_rng, distance_rng, side_rng = jax.random.split(
            rng, 5
        )
        state = super().reset(base_rng)
        speed = jax.random.uniform(
            speed_rng,
            minval=self._config.shot_speed_range[0],
            maxval=self._config.shot_speed_range[1],
        )
        lateral_magnitude = jax.random.uniform(
            lateral_rng,
            minval=self._config.shot_lateral_range[0],
            maxval=self._config.shot_lateral_range[1],
        )
        random_side = jp.where(jax.random.bernoulli(side_rng), 1.0, -1.0)
        side = jp.where(
            self._config.fixed_shot_side == 0,
            random_side,
            jp.asarray(self._config.fixed_shot_side, dtype=jp.float32),
        )
        start_distance = jax.random.uniform(
            distance_rng,
            minval=self._config.shot_start_distance_range[0],
            maxval=self._config.shot_start_distance_range[1],
        )
        torso_position = state.data.site_xpos[self._torso_site]
        goal_plane_x = torso_position[0] + self._config.goal_plane_offset_m
        goal_center_y = torso_position[1]
        shot_lateral = side * lateral_magnitude
        ball_position = jp.array(
            [
                torso_position[0] + start_distance,
                goal_center_y + shot_lateral,
                0.11,
            ]
        )
        ball_velocity = jp.array([-speed, 0.0, 0.0])
        qpos = state.data.qpos.at[
            self._ball_qpos : self._ball_qpos + 3
        ].set(ball_position)
        qvel = state.data.qvel.at[
            self._ball_dof : self._ball_dof + 6
        ].set(jp.concatenate([ball_velocity, jp.zeros(3)]))
        data = mjx.forward(self._mjx_model, state.data.replace(qpos=qpos, qvel=qvel))

        command = jp.array([0.0, side * self._config.lateral_command_mps, 0.0])
        state.info["command"] = command
        state.info["goal_plane_x"] = goal_plane_x
        state.info["goal_center_y"] = goal_center_y
        state.info["shot_lateral_m"] = shot_lateral
        state.info["shot_speed_mps"] = speed
        state.info["last_keeper_error_m"] = jp.abs(
            ball_position[1] - torso_position[1]
        )
        state.info["last_ball_velocity"] = ball_velocity
        state.info["contacted"] = jp.array(False)
        state.info["saved"] = jp.array(False)
        state.info["conceded"] = jp.array(False)

        state.metrics.update(
            {
                "reward/goalkeeper_progress": jp.array(0.0),
                "reward/goalkeeper_coverage": jp.array(0.0),
                "reward/goalkeeper_contact": jp.array(0.0),
                "reward/goalkeeper_save": jp.array(0.0),
                "cost/goalkeeper_concede": jp.array(0.0),
                "event/goalkeeper_contact": jp.array(0.0),
                "event/goalkeeper_save": jp.array(0.0),
                "event/goalkeeper_concede": jp.array(0.0),
                "diagnostic/goalkeeper_error_m": state.info[
                    "last_keeper_error_m"
                ],
                "diagnostic/goalkeeper_time_to_plane_s": start_distance / speed,
                "diagnostic/goalkeeper_shot_lateral_m": shot_lateral,
                "diagnostic/goalkeeper_shot_speed_mps": speed,
            }
        )
        return state.replace(data=data, obs=self._get_obs(data, state.info))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        previous_error = state.info["last_keeper_error_m"]
        previous_ball_velocity = state.info["last_ball_velocity"]
        state = super().step(state, action)

        torso_position = state.data.site_xpos[self._torso_site]
        ball_position = state.data.xpos[self._ball_body]
        ball_velocity = state.data.qvel[self._ball_dof : self._ball_dof + 3]
        keeper_error = jp.abs(ball_position[1] - torso_position[1])
        progress = previous_error - keeper_error
        distance_to_plane = ball_position[0] - state.info["goal_plane_x"]
        time_to_plane = jp.maximum(
            distance_to_plane / jp.maximum(-ball_velocity[0], 1.0e-3), 0.0
        )
        urgency = jp.clip(1.0 - time_to_plane / 1.5, 0.0, 1.0)
        coverage = jp.exp(-4.0 * jp.square(keeper_error)) * urgency

        velocity_delta = jp.linalg.norm(
            ball_velocity[:2] - previous_ball_velocity[:2]
        )
        near_keeper = jp.abs(ball_position[0] - torso_position[0]) <= 0.9
        contact_event = (
            (velocity_delta >= self._config.contact_velocity_delta_mps)
            & near_keeper
            & ~state.info["contacted"]
        )
        contacted = state.info["contacted"] | contact_event
        crossed_plane = ball_position[0] <= state.info["goal_plane_x"]
        outside_goal = (
            jp.abs(ball_position[1] - state.info["goal_center_y"])
            > self._config.goal_half_width_m
        )
        rebound = contacted & (ball_velocity[0] >= -0.15)
        conceded_event = (
            crossed_plane
            & ~outside_goal
            & ~state.info["conceded"]
            & ~state.info["saved"]
        )
        fall = state.metrics["cost/fall"] > 0.5
        timed_out = state.info["step"] >= self._config.episode_length
        save_event = (
            (rebound | (crossed_plane & outside_goal) | (timed_out & contacted))
            & ~fall
            & ~state.info["saved"]
            & ~state.info["conceded"]
            & ~conceded_event
        )

        goalkeeper_reward = (
            self._config.reward.goalkeeper_progress * progress
            + self._config.reward.goalkeeper_coverage * coverage * self.dt
            + self._config.reward.goalkeeper_contact
            * contact_event.astype(jp.float32)
            + self._config.reward.goalkeeper_save * save_event.astype(jp.float32)
            + self._config.reward.goalkeeper_concede
            * conceded_event.astype(jp.float32)
        )
        state.info["last_keeper_error_m"] = keeper_error
        state.info["last_ball_velocity"] = ball_velocity
        state.info["contacted"] = contacted
        state.info["saved"] = state.info["saved"] | save_event
        state.info["conceded"] = state.info["conceded"] | conceded_event
        state.metrics.update(
            {
                "reward/goalkeeper_progress": progress,
                "reward/goalkeeper_coverage": coverage,
                "reward/goalkeeper_contact": contact_event.astype(jp.float32),
                "reward/goalkeeper_save": save_event.astype(jp.float32),
                "cost/goalkeeper_concede": conceded_event.astype(jp.float32),
                "event/goalkeeper_contact": contact_event.astype(jp.float32),
                "event/goalkeeper_save": save_event.astype(jp.float32),
                "event/goalkeeper_concede": conceded_event.astype(jp.float32),
                "diagnostic/goalkeeper_error_m": keeper_error,
                "diagnostic/goalkeeper_time_to_plane_s": time_to_plane,
                "diagnostic/goalkeeper_shot_lateral_m": state.info[
                    "shot_lateral_m"
                ],
                "diagnostic/goalkeeper_shot_speed_mps": state.info[
                    "shot_speed_mps"
                ],
            }
        )
        done = (
            state.done.astype(bool) | save_event | conceded_event
        ).astype(jp.float32)
        return state.replace(
            obs=self._get_obs(state.data, state.info),
            reward=state.reward + goalkeeper_reward,
            done=done,
        )
