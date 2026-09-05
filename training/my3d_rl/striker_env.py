"""Long-horizon closed-loop approach-and-kick task for the Booster T1.

Apollo's stable walk policy remains the locomotion baseline.  A learned
residual is smoothly enabled only near a target-relative contact pose.  This
keeps early exploration from destroying the approach while still letting a
privileged teacher learn when and how to kick from changing physical state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

from .contract import PolicyContract, load_policy_contract
from .kick_env import DirectionalKick, default_config as kick_default_config
from .kick_transition import estimate_locomotion_phase_jax, hip_pitch_indices
from .rcss_scene import DEFAULT_RESOURCE_ROOT
from .t1_control import APOLLO_DEFAULT_POSE, KICK_ACTION_SCALE


DEFAULT_CONTRACT = (
    Path(__file__).parents[1] / "contracts" / "striker_policy_v1.yaml"
)


def default_config() -> config_dict.ConfigDict:
    """Return the first closed-loop teacher curriculum configuration."""
    config = kick_default_config()
    config.episode_length = 1000
    config.robot_distance_range = [0.55, 1.40]
    config.robot_lateral_range = [-0.15, 0.15]
    config.robot_yaw_noise_range = [-0.20, 0.20]
    config.reset_joint_noise = 0.01
    config.reset_root_velocity_noise = 0.03
    config.approach_standoff = 0.31
    config.approach_ball_lateral = -0.04
    config.approach_position_gain = 2.0
    config.approach_lateral_gain = 2.5
    config.approach_yaw_gain = 2.0
    config.approach_max_forward_speed = 0.70
    config.approach_max_backward_speed = 0.15
    config.approach_max_lateral_speed = 0.25
    config.approach_max_yaw_speed = 0.55
    config.kick_activation_radius = 0.28
    config.kick_full_radius = 0.07
    config.kick_heading_radius = 0.45
    config.kick_full_heading = 0.10
    # A perfect activation of 1.0 plus a 25-cycle confirmation reproduced the
    # runtime's release starvation: ordinary gait sway repeatedly crossed the
    # boundary.  Keep a bounded pose gate, but let the learned transition own
    # the final centimetres and gait phase after five consecutive 50 Hz steps.
    config.kick_trigger_threshold = 0.80
    config.kick_settled_distance = 0.14
    config.kick_settled_heading = 0.15
    config.kick_settled_confirmation_steps = 5
    config.kick_rearm_steps = 25
    config.kick_prior_enabled = True
    # Zero preserves the deployed kick-residual contract.  The staged T1
    # striker curriculum raises this floor so the actor can first learn ball
    # chasing and then a continuous chase-to-contact transition.
    config.learned_approach_residual_floor = 0.0
    config.kick_walk_command = [0.50, -0.04, 0.0]
    config.kick_walk_duration_steps = 33
    config.fixed_kick_prior_index = -1
    config.residual_scale = 0.10
    config.success_radius = 0.50
    config.arrival_speed_tolerance = 0.50
    config.contact_event_reward = 5.0
    config.gate_success_reward = 30.0
    config.fall_penalty = 100.0
    config.miss_penalty = 20.0
    config.post_contact_timeout_steps = 250
    config.action_magnitude_cost = 0.002
    return config


def _safe_unit(vector: jax.Array) -> jax.Array:
    return vector / jp.maximum(jp.linalg.norm(vector), 1.0e-6)


def closed_loop_approach_control(
    ball_local_xy: jax.Array,
    target_local: jax.Array,
    *,
    standoff: float,
    ball_lateral: float,
    position_gain: float,
    lateral_gain: float,
    yaw_gain: float,
    max_forward_speed: float,
    max_backward_speed: float,
    max_lateral_speed: float,
    max_yaw_speed: float,
    activation_radius: float,
    full_radius: float,
    heading_radius: float,
    full_heading: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map relative geometry to a walk command and smooth kick activation."""
    target_local = _safe_unit(target_local)
    target_left = jp.array([-target_local[1], target_local[0]])
    contact_error = (
        ball_local_xy - target_local * standoff - target_left * ball_lateral
    )
    distance = jp.linalg.norm(contact_error)
    heading_error = jp.arctan2(target_local[1], target_local[0])

    distance_activation = jp.clip(
        (activation_radius - distance)
        / jp.maximum(activation_radius - full_radius, 1.0e-6),
        0.0,
        1.0,
    )
    heading_activation = jp.clip(
        (heading_radius - jp.abs(heading_error))
        / jp.maximum(heading_radius - full_heading, 1.0e-6),
        0.0,
        1.0,
    )
    activation = distance_activation * heading_activation
    command = jp.array(
        [
            jp.clip(
                position_gain * contact_error[0],
                -max_backward_speed,
                max_forward_speed,
            ),
            jp.clip(
                lateral_gain * contact_error[1],
                -max_lateral_speed,
                max_lateral_speed,
            ),
            jp.clip(
                yaw_gain * heading_error,
                -max_yaw_speed,
                max_yaw_speed,
            ),
        ]
    )
    # Do not attenuate the approach before the discrete prior trigger.  Small
    # Apollo commands fall below its effective gait deadband and previously
    # left the robot stranded just outside the contact envelope.  At the exact
    # target the proportional command is already zero; after triggering, step
    # replaces it with the versioned kick-walk command.
    return command, activation, contact_error, heading_error


class LongHorizonStriker(DirectionalKick):
    """Twenty-second privileged-teacher task with a deployable actor boundary."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
        kick_prior_joint_residuals: np.ndarray | None = None,
        kick_prior_target_distances: np.ndarray | None = None,
    ) -> None:
        contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        if contract.policy_name != "striker_policy_v1":
            raise ValueError("LongHorizonStriker requires striker_policy_v1")
        config = default_config() if config is None else config
        super().__init__(
            config=config,
            config_overrides=config_overrides,
            contract=contract,
            resource_root=resource_root,
            prefix=prefix,
        )
        if not (
            0.0 < self._config.kick_full_radius
            < self._config.kick_activation_radius
            and 0.0 < self._config.kick_full_heading
            < self._config.kick_heading_radius
        ):
            raise ValueError("kick activation thresholds must be strictly nested")
        if not 0.0 < self._config.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        if self._config.arrival_speed_tolerance <= 0.0:
            raise ValueError("arrival_speed_tolerance must be positive")
        if self.contract.observation_size != 102 or self.action_size != 23:
            raise ValueError("striker-v1 must preserve the 102 -> 23 boundary")
        if not 0.0 < self._config.kick_trigger_threshold <= 1.0:
            raise ValueError("kick_trigger_threshold must be in (0, 1]")
        if not 0.0 <= self._config.learned_approach_residual_floor <= 1.0:
            raise ValueError("learned_approach_residual_floor must be in [0, 1]")
        if (
            self._config.kick_settled_distance < self._config.kick_full_radius
            or self._config.kick_settled_distance
            > self._config.kick_activation_radius
            or self._config.kick_settled_heading < self._config.kick_full_heading
            or self._config.kick_settled_heading
            > self._config.kick_heading_radius
            or self._config.kick_settled_confirmation_steps < 1
        ):
            raise ValueError("settled kick gate must stay inside activation bounds")
        if self._config.kick_rearm_steps < 0:
            raise ValueError("kick_rearm_steps must be non-negative")
        if (
            len(self._config.kick_walk_command) != 3
            or self._config.kick_walk_duration_steps < 0
        ):
            raise ValueError("kick walk baseline configuration is invalid")
        if kick_prior_joint_residuals is None:
            kick_prior_joint_residuals = np.zeros(
                (1, 1, self.action_size), dtype=np.float32
            )
            kick_prior_target_distances = np.array([2.0], dtype=np.float32)
            self._uses_kick_prior = False
        else:
            kick_prior_joint_residuals = np.asarray(
                kick_prior_joint_residuals, dtype=np.float32
            )
            if kick_prior_joint_residuals.ndim == 2:
                kick_prior_joint_residuals = kick_prior_joint_residuals[None, ...]
            if (
                kick_prior_joint_residuals.ndim != 3
                or kick_prior_joint_residuals.shape[1] < 2
                or kick_prior_joint_residuals.shape[2] != self.action_size
                or not np.isfinite(kick_prior_joint_residuals).all()
            ):
                raise ValueError(
                    "kick prior must be a finite [K, T, 23] trajectory bank"
                )
            if kick_prior_target_distances is None:
                if kick_prior_joint_residuals.shape[0] != 1:
                    raise ValueError("a multi-entry kick prior requires distances")
                kick_prior_target_distances = np.array([2.0], dtype=np.float32)
            kick_prior_target_distances = np.asarray(
                kick_prior_target_distances, dtype=np.float32
            )
            if (
                kick_prior_target_distances.shape
                != (kick_prior_joint_residuals.shape[0],)
                or not np.isfinite(kick_prior_target_distances).all()
                or np.any(kick_prior_target_distances <= 0.0)
                or np.any(np.diff(kick_prior_target_distances) < 0.0)
            ):
                raise ValueError(
                    "kick-prior target distances must be finite, positive and sorted"
                )
            self._uses_kick_prior = True
        self._kick_prior_joint_residuals = jp.asarray(kick_prior_joint_residuals)
        self._kick_prior_target_distances = jp.asarray(
            kick_prior_target_distances
        )
        if not -1 <= self._config.fixed_kick_prior_index < len(
            kick_prior_target_distances
        ):
            raise ValueError("fixed_kick_prior_index is outside the prior bank")
        self._root_qpos = self._mj_model.joint(prefix + "root").qposadr[0]
        self._left_hip_pitch, self._right_hip_pitch = hip_pitch_indices(
            self.contract.joint_order
        )

    def _task_features(
        self, data: mjx.Data, goal_world: jax.Array
    ) -> dict[str, jax.Array]:
        torso_xmat = data.site_xmat[self._torso_site]
        yaw = jp.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        world_to_yaw = jp.array([[c, s], [-s, c]])
        torso_pos = data.xpos[self._torso_body]
        ball_pos = data.xpos[self._ball_body]
        ball_world_vel = data.qvel[self._ball_dof : self._ball_dof + 3]
        torso_world_vel = data.qvel[self._root_dof : self._root_dof + 3]
        goal_delta = goal_world - ball_pos[:2]
        goal_distance = jp.linalg.norm(goal_delta)
        target_world = _safe_unit(goal_delta)
        target_local = world_to_yaw @ target_world
        ball_local_xy = world_to_yaw @ (ball_pos[:2] - torso_pos[:2])
        ball_local_vel_xy = world_to_yaw @ (
            ball_world_vel[:2] - torso_world_vel[:2]
        )
        command, activation, contact_error, heading_error = (
            closed_loop_approach_control(
                ball_local_xy,
                target_local,
                standoff=self._config.approach_standoff,
                ball_lateral=self._config.approach_ball_lateral,
                position_gain=self._config.approach_position_gain,
                lateral_gain=self._config.approach_lateral_gain,
                yaw_gain=self._config.approach_yaw_gain,
                max_forward_speed=self._config.approach_max_forward_speed,
                max_backward_speed=self._config.approach_max_backward_speed,
                max_lateral_speed=self._config.approach_max_lateral_speed,
                max_yaw_speed=self._config.approach_max_yaw_speed,
                activation_radius=self._config.kick_activation_radius,
                full_radius=self._config.kick_full_radius,
                heading_radius=self._config.kick_heading_radius,
                full_heading=self._config.kick_full_heading,
            )
        )
        return {
            "world_to_yaw": world_to_yaw,
            "torso_pos": torso_pos,
            "torso_world_vel": torso_world_vel,
            "ball_pos": ball_pos,
            "ball_world_vel": ball_world_vel,
            "ball_local_xy": ball_local_xy,
            "ball_local_vel_xy": ball_local_vel_xy,
            "target_world": target_world,
            "target_local": target_local,
            "goal_distance": goal_distance,
            "command": command,
            "activation": activation,
            "contact_error": contact_error,
            "contact_distance": jp.linalg.norm(contact_error),
            "heading_error": heading_error,
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            distance_rng,
            lateral_rng,
            target_rng,
            yaw_rng,
            joint_rng,
            velocity_rng,
            arrival_rng,
            mode_rng,
            target_distance_rng,
        ) = jax.random.split(rng, 10)
        qpos = jp.asarray(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)
        target_angle = jax.random.uniform(
            target_rng,
            minval=self._config.target_angle_range[0],
            maxval=self._config.target_angle_range[1],
        )
        target_world = jp.array([jp.cos(target_angle), jp.sin(target_angle)])
        target_left = jp.array([-target_world[1], target_world[0]])
        robot_distance = jax.random.uniform(
            distance_rng,
            minval=self._config.robot_distance_range[0],
            maxval=self._config.robot_distance_range[1],
        )
        robot_lateral = jax.random.uniform(
            lateral_rng,
            minval=self._config.robot_lateral_range[0],
            maxval=self._config.robot_lateral_range[1],
        )
        ball_pos = jp.array([0.0, 0.0, 0.11])
        robot_xy = (
            ball_pos[:2]
            - target_world * robot_distance
            + target_left * robot_lateral
        )
        robot_yaw = target_angle + jax.random.uniform(
            yaw_rng,
            minval=self._config.robot_yaw_noise_range[0],
            maxval=self._config.robot_yaw_noise_range[1],
        )
        qpos = qpos.at[self._root_qpos : self._root_qpos + 2].set(robot_xy)
        qpos = qpos.at[self._root_qpos + 3 : self._root_qpos + 7].set(
            jp.array(
                [
                    jp.cos(0.5 * robot_yaw),
                    0.0,
                    0.0,
                    jp.sin(0.5 * robot_yaw),
                ]
            )
        )
        joint_noise = jax.random.uniform(
            joint_rng,
            (self.action_size,),
            minval=-self._config.reset_joint_noise,
            maxval=self._config.reset_joint_noise,
        )
        qpos = qpos.at[self._joint_qpos].add(joint_noise)
        root_velocity = jax.random.uniform(
            velocity_rng,
            (6,),
            minval=-self._config.reset_root_velocity_noise,
            maxval=self._config.reset_root_velocity_noise,
        )
        qvel = qvel.at[self._root_dof : self._root_dof + 6].set(root_velocity)
        qpos = qpos.at[self._ball_qpos : self._ball_qpos + 3].set(ball_pos)

        target_distance = jax.random.uniform(
            target_distance_rng,
            minval=self._config.target_distance_range[0],
            maxval=self._config.target_distance_range[1],
        )
        goal_world = ball_pos[:2] + target_world * target_distance
        action_mode_index = (
            jp.asarray(self._config.fixed_action_mode, dtype=jp.int32)
            if self._config.fixed_action_mode >= 0
            else jax.random.randint(mode_rng, (), 0, 3)
        )
        arrival_min = jp.array([0.4, 1.5, 1.0])[action_mode_index]
        arrival_max = jp.array([1.2, 2.5, 2.0])[action_mode_index]
        desired_arrival_speed = jax.random.uniform(
            arrival_rng, minval=arrival_min, maxval=arrival_max
        )
        if self._config.fixed_desired_arrival_speed >= 0.0:
            desired_arrival_speed = jp.asarray(
                self._config.fixed_desired_arrival_speed, dtype=jp.float32
            )

        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._pos_actuator].set(jp.asarray(APOLLO_DEFAULT_POSE))
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)
        features = self._task_features(data, goal_world)
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "time_out": jp.array(False),
            "last_action": jp.zeros(self.action_size),
            "walk_last_action": jp.zeros(self.action_size),
            "kick_step": jp.array(-1, dtype=jp.int32),
            "kick_prior_index": jp.array(0, dtype=jp.int32),
            "kick_cooldown": jp.array(0, dtype=jp.int32),
            "kick_settled_steps": jp.array(0, dtype=jp.int32),
            "goal_world": goal_world,
            "initial_ball_xy": ball_pos[:2],
            "initial_robot_distance": robot_distance,
            "initial_robot_lateral": robot_lateral,
            "initial_robot_yaw_error": robot_yaw - target_angle,
            "initial_target_angle": target_angle,
            "initial_target_distance": target_distance,
            "initial_goal_distance": features["goal_distance"],
            "last_goal_distance": features["goal_distance"],
            "last_contact_distance": features["contact_distance"],
            "desired_arrival_speed": desired_arrival_speed,
            "action_mode": jax.nn.one_hot(action_mode_index, 3),
            "contacted": jp.array(False),
            "succeeded": jp.array(False),
            "maximum_directional_speed": jp.array(0.0),
            "post_contact_steps": jp.array(0, dtype=jp.int32),
        }
        metrics = {
            "reward/approach_progress": jp.array(0.0),
            "reward/approach_velocity": jp.array(0.0),
            "reward/heading": jp.array(0.0),
            "reward/ready": jp.array(0.0),
            "reward/ball_speed_tracking": jp.array(0.0),
            "reward/ball_progress": jp.array(0.0),
            "reward/upright": jp.array(0.0),
            "cost/action_rate": jp.array(0.0),
            "cost/action_magnitude": jp.array(0.0),
            "cost/far_action": jp.array(0.0),
            "cost/fall": jp.array(0.0),
            "cost/arrival_speed": jp.array(0.0),
            "cost/miss": jp.array(0.0),
            "event/contact": jp.array(0.0),
            "event/kick_trigger": jp.array(0.0),
            "event/success": jp.array(0.0),
            "diagnostic/kick_activation": features["activation"],
            "diagnostic/goal_distance": features["goal_distance"],
            "diagnostic/contact_distance": features["contact_distance"],
            "diagnostic/heading_error": features["heading_error"],
            "diagnostic/arrival_speed_error": jp.array(0.0),
        }
        obs = self._get_obs(data, info)
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        features = self._task_features(state.data, state.info["goal_world"])
        inside_settled_envelope = (
            (features["contact_distance"] <= self._config.kick_settled_distance)
            & (
                jp.abs(features["heading_error"])
                <= self._config.kick_settled_heading
            )
        )
        kick_settled_steps = jp.where(
            inside_settled_envelope,
            jp.minimum(
                state.info["kick_settled_steps"] + 1,
                self._config.kick_settled_confirmation_steps,
            ),
            0,
        )
        settled_trigger = (
            kick_settled_steps >= self._config.kick_settled_confirmation_steps
        )
        kick_trigger = (
            self._uses_kick_prior
            & bool(self._config.kick_prior_enabled)
            & (state.info["kick_step"] < 0)
            & (state.info["kick_cooldown"] <= 0)
            & (
                (features["activation"] >= self._config.kick_trigger_threshold)
                | settled_trigger
            )
        )
        selected_prior_index = jp.argmin(
            jp.abs(
                self._kick_prior_target_distances - features["goal_distance"]
            )
        )
        selected_prior_index = jp.where(
            self._config.fixed_kick_prior_index >= 0,
            self._config.fixed_kick_prior_index,
            selected_prior_index,
        )
        kick_prior_index = jp.where(
            kick_trigger, selected_prior_index, state.info["kick_prior_index"]
        )
        kick_step = jp.where(kick_trigger, 0, state.info["kick_step"])
        kick_active = kick_step >= 0
        prior_index = jp.clip(
            kick_step, 0, self._kick_prior_joint_residuals.shape[1] - 1
        )
        prior_joint_residual = jp.where(
            kick_active,
            self._kick_prior_joint_residuals[kick_prior_index, prior_index],
            jp.zeros(self.action_size),
        )
        kick_walk_active = kick_active & (
            kick_step < self._config.kick_walk_duration_steps
        )
        approach_command = jp.where(
            kick_active,
            jp.where(
                kick_walk_active,
                jp.asarray(self._config.kick_walk_command),
                jp.zeros(3),
            ),
            features["command"],
        )
        torso_xmat = state.data.site_xmat[self._torso_site]
        gravity = torso_xmat.T @ jp.array([0.0, 0.0, -1.0])
        walk_observation = jp.concatenate(
            [
                state.data.sensordata[self._gyro_slice],
                gravity,
                approach_command,
                state.data.qpos[self._joint_qpos] - self._default_pose,
                state.data.qvel[self._joint_dof],
                state.info["walk_last_action"],
            ]
        )
        walk_action = self._walk_policy(walk_observation)
        correction_gate = jp.maximum(
            jp.maximum(
                features["activation"], kick_active.astype(jp.float32)
            ),
            jp.asarray(
                self._config.learned_approach_residual_floor,
                dtype=jp.float32,
            ),
        )
        learned_correction = (
            correction_gate
            * self._config.residual_scale
            * action
            * jp.asarray(KICK_ACTION_SCALE)
        )
        targets = jp.clip(
            self._default_pose
            + 0.25 * walk_action
            + prior_joint_residual
            + learned_correction,
            self._lowers,
            self._uppers,
        )
        ctrl = state.data.ctrl.at[self._pos_actuator].set(targets)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        next_features = self._task_features(data, state.info["goal_world"])

        torso_xmat = data.site_xmat[self._torso_site]
        torso_height = data.xpos[self._torso_body, 2]
        ball_displacement = (
            next_features["ball_pos"][:2] - state.info["initial_ball_xy"]
        )
        ball_speed = jp.linalg.norm(next_features["ball_world_vel"][:2])
        directional_speed = jp.clip(
            jp.dot(
                next_features["ball_world_vel"][:2],
                next_features["target_world"],
            ),
            0.0,
            6.0,
        )
        maximum_directional_speed = jp.maximum(
            state.info["maximum_directional_speed"], directional_speed
        )
        contact = (ball_speed >= 0.15) | (jp.linalg.norm(ball_displacement) >= 0.08)
        contact_event = contact & ~state.info["contacted"]
        arrival_speed_error = jp.abs(
            directional_speed - state.info["desired_arrival_speed"]
        )
        requested_ball_speed = jp.sqrt(
            jp.square(state.info["desired_arrival_speed"])
            + 2.0
            * self._config.rolling_deceleration_mps2
            * next_features["goal_distance"]
        )
        ball_speed_tracking_error = jp.abs(
            directional_speed - requested_ball_speed
        )
        ball_speed_tracking = (
            jp.exp(-2.0 * jp.square(ball_speed_tracking_error))
            * contact.astype(jp.float32)
        )
        arrival_zone = (
            jp.clip(1.5 - next_features["goal_distance"], 0.0, 1.0)
            * contact.astype(jp.float32)
        )
        excess_arrival_speed_error = jp.maximum(
            arrival_speed_error - self._config.arrival_speed_tolerance, 0.0
        )
        success = (
            contact
            & (next_features["goal_distance"] <= self._config.success_radius)
            & (arrival_speed_error <= self._config.arrival_speed_tolerance)
        )
        success_event = success & ~state.info["succeeded"]
        fall = (torso_height < 0.35) | (torso_xmat[2, 2] < 0.0)
        invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        approach_progress_rate = jp.clip(
            (
                state.info["last_contact_distance"]
                - next_features["contact_distance"]
            )
            / self.dt,
            -1.0,
            2.0,
        )
        goal_progress_rate = jp.clip(
            (state.info["last_goal_distance"] - next_features["goal_distance"])
            / self.dt,
            -2.0,
            6.0,
        )
        torso_local_velocity = (
            next_features["world_to_yaw"]
            @ next_features["torso_world_vel"][:2]
        )
        velocity_error = torso_local_velocity - next_features["command"][:2]
        approach_velocity = jp.exp(-jp.sum(jp.square(velocity_error)) / 0.25)
        approach_active = (~contact).astype(jp.float32)
        heading_reward = jp.exp(-4.0 * jp.square(next_features["heading_error"]))
        upright = jp.clip(torso_xmat[2, 2], 0.0, 1.0)
        action_rate = jp.sum(jp.square(action - state.info["last_action"]))
        action_magnitude = jp.sum(jp.square(action))
        far_action = (1.0 - correction_gate) * action_magnitude
        post_contact_steps = jp.where(
            state.info["contacted"] | contact,
            state.info["post_contact_steps"] + 1,
            0,
        )
        contact_time_out = (
            post_contact_steps >= self._config.post_contact_timeout_steps
        )
        miss_event = contact_time_out & ~success

        reward_terms = {
            "approach_progress": 3.0 * approach_progress_rate * approach_active,
            "approach_velocity": 1.0 * approach_velocity * approach_active,
            "heading": 0.5 * heading_reward,
            "ready": 0.5 * next_features["activation"],
            "ball_speed_tracking": 2.0 * ball_speed_tracking,
            "ball_progress": 4.0 * goal_progress_rate * contact.astype(jp.float32),
            "upright": 0.2 * upright,
            "action_rate": -0.01 * action_rate * correction_gate,
            "action_magnitude": -self._config.action_magnitude_cost
            * action_magnitude
            * correction_gate,
            "far_action": -0.01 * far_action,
            "contact": self._config.contact_event_reward
            / self.dt
            * contact_event.astype(jp.float32),
            "success": self._config.gate_success_reward
            / self.dt
            * success_event.astype(jp.float32),
            "arrival_speed": -4.0
            * excess_arrival_speed_error
            * arrival_zone,
            "miss": -self._config.miss_penalty
            / self.dt
            * miss_event.astype(jp.float32),
            "fall": -self._config.fall_penalty
            / self.dt
            * fall.astype(jp.float32),
        }
        reward = sum(reward_terms.values()) * self.dt

        state.info["step"] += 1
        time_out = (
            (state.info["step"] >= self._config.episode_length)
            | contact_time_out
        )
        state.info["time_out"] = time_out
        next_kick_step = jp.where(kick_active, kick_step + 1, kick_step)
        kick_finished = next_kick_step >= self._kick_prior_joint_residuals.shape[1]
        state.info["kick_step"] = jp.where(kick_finished, -1, next_kick_step)
        state.info["kick_prior_index"] = kick_prior_index
        state.info["kick_cooldown"] = jp.where(
            kick_finished,
            self._config.kick_rearm_steps,
            jp.maximum(state.info["kick_cooldown"] - 1, 0),
        )
        state.info["kick_settled_steps"] = jp.where(
            kick_trigger | kick_active, 0, kick_settled_steps
        )
        state.info["last_action"] = action
        state.info["walk_last_action"] = walk_action
        state.info["last_goal_distance"] = next_features["goal_distance"]
        state.info["last_contact_distance"] = next_features["contact_distance"]
        state.info["contacted"] = state.info["contacted"] | contact
        state.info["succeeded"] = state.info["succeeded"] | success
        state.info["maximum_directional_speed"] = maximum_directional_speed
        state.info["post_contact_steps"] = post_contact_steps
        done = fall | invalid | success | time_out
        obs = self._get_obs(data, state.info)
        state.metrics.update(
            {
                "reward/approach_progress": approach_progress_rate,
                "reward/approach_velocity": approach_velocity,
                "reward/heading": heading_reward,
                "reward/ready": next_features["activation"],
                "reward/ball_speed_tracking": ball_speed_tracking,
                "reward/ball_progress": goal_progress_rate,
                "reward/upright": upright,
                "cost/action_rate": action_rate,
                "cost/action_magnitude": action_magnitude,
                "cost/far_action": far_action,
                "cost/fall": fall.astype(jp.float32),
                "cost/arrival_speed": excess_arrival_speed_error * arrival_zone,
                "cost/miss": miss_event.astype(jp.float32),
                "event/contact": contact_event.astype(jp.float32),
                "event/kick_trigger": kick_trigger.astype(jp.float32),
                "event/success": success_event.astype(jp.float32),
                "diagnostic/kick_activation": next_features["activation"],
                "diagnostic/goal_distance": next_features["goal_distance"],
                "diagnostic/contact_distance": next_features["contact_distance"],
                "diagnostic/heading_error": next_features["heading_error"],
                "diagnostic/arrival_speed_error": arrival_speed_error,
            }
        )
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        features = self._task_features(data, info["goal_world"])
        torso_xmat = data.site_xmat[self._torso_site]
        gravity = torso_xmat.T @ jp.array([0.0, 0.0, -1.0])
        joint_position_offset = data.qpos[self._joint_qpos] - self._default_pose
        joint_velocity = data.qvel[self._joint_dof]
        locomotion_phase, support_hint, _ = estimate_locomotion_phase_jax(
            joint_position_offset,
            joint_velocity,
            self._left_hip_pitch,
            self._right_hip_pitch,
        )
        requested_ball_speed = jp.sqrt(
            jp.square(info["desired_arrival_speed"])
            + 2.0
            * self._config.rolling_deceleration_mps2
            * features["goal_distance"]
        )
        kick_active = info["kick_step"] >= 0
        kick_progress = jp.where(
            kick_active,
            jp.clip(
                info["kick_step"]
                / jp.maximum(self._kick_prior_joint_residuals.shape[1] - 1, 1),
                0.0,
                1.0,
            ),
            0.0,
        )
        kick_phase = jp.where(
            kick_active,
            jp.array(
                [jp.sin(jp.pi * kick_progress), jp.cos(jp.pi * kick_progress)]
            ),
            jp.array([0.0, 1.0]),
        )
        kick_walk_active = kick_active & (
            info["kick_step"] < self._config.kick_walk_duration_steps
        )
        approach_command = jp.where(
            kick_active,
            jp.where(
                kick_walk_active,
                jp.asarray(self._config.kick_walk_command),
                jp.zeros(3),
            ),
            features["command"],
        )
        actor = jp.concatenate(
            [
                data.sensordata[self._gyro_slice],
                gravity,
                joint_position_offset,
                joint_velocity,
                info["last_action"],
                jp.array(
                    [
                        features["ball_local_xy"][0],
                        features["ball_local_xy"][1],
                        features["ball_pos"][2] - features["torso_pos"][2],
                    ]
                ),
                jp.array(
                    [
                        features["ball_local_vel_xy"][0],
                        features["ball_local_vel_xy"][1],
                        features["ball_world_vel"][2],
                    ]
                ),
                features["target_local"],
                jp.array([features["goal_distance"]]),
                jp.array([requested_ball_speed]),
                jp.array([info["desired_arrival_speed"]]),
                info["action_mode"],
                jp.array([0.0, 1.0]),
                approach_command,
                jp.array([features["activation"]]),
                kick_phase,
                locomotion_phase,
                support_hint,
            ]
        )
        teacher = jp.concatenate(
            [
                actor,
                info["walk_last_action"],
                features["torso_world_vel"],
                features["ball_pos"],
                features["ball_world_vel"],
                jp.array([features["torso_pos"][2]]),
                features["contact_error"],
                jp.array([features["heading_error"]]),
            ]
        )
        return {
            "state": actor,
            "teacher_state": teacher,
            "privileged_state": teacher,
        }
