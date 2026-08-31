"""Minimal MJX directional-kick task built from RCSSServerMJ physics assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

from .contract import PolicyContract, load_policy_contract
from .rcss_scene import DEFAULT_RESOURCE_ROOT, build_single_t1_soccer_model


DEFAULT_CONTRACT = Path(__file__).parents[1] / "contracts" / "kick_policy_v2.yaml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        impl="jax",
        # Warp reported a peak requirement of 928 active contact candidates
        # during randomized PPO rollouts.  Keep explicit headroom so contacts
        # are never silently dropped by the training backend.
        naconmax=1024,
        njmax=256,
        kp=20.0,
        kd=0.5,
        action_scale=[
            0.10,
            0.10,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
            0.15,
            0.35,
            0.25,
            0.25,
            0.45,
            0.25,
            0.20,
            0.35,
            0.25,
            0.25,
            0.45,
            0.25,
            0.20,
        ],
        ball_x_range=[-0.01, 0.08],
        ball_y_range=[-0.08, 0.08],
        target_angle_range=[-0.261799, 0.261799],
        target_distance_range=[2.0, 5.0],
        rolling_deceleration_mps2=0.08,
    )


class DirectionalKick(mjx_env.MjxEnv):
    """Target-conditioned fixed-ball task with a deployable 96-value state."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
    ) -> None:
        config = default_config() if config is None else config
        super().__init__(config, config_overrides)
        self.contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        self.prefix = prefix
        self._resource_root = resource_root

        self._mj_model = build_single_t1_soccer_model(
            resource_root, prefix=prefix, robot_x=-0.32, robot_y=0.0
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._configure_pd_actuators()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._joint_qpos = np.array(
            [
                self._mj_model.joint(prefix + name).qposadr[0]
                for name in self.contract.joint_order
            ]
        )
        self._joint_dof = np.array(
            [
                self._mj_model.joint(prefix + name).dofadr[0]
                for name in self.contract.joint_order
            ]
        )
        self._pos_actuator = np.array(
            [
                self._mj_model.actuator(prefix + name + "_pos").id
                for name in self.contract.effector_order
            ]
        )
        self._ball_qpos = self._mj_model.joint("ball-root").qposadr[0]
        self._ball_dof = self._mj_model.joint("ball-root").dofadr[0]
        self._root_dof = self._mj_model.joint(prefix + "root").dofadr[0]
        self._ball_body = self._mj_model.body("ball").id
        self._torso_body = self._mj_model.body(prefix + "torso").id
        self._torso_site = self._mj_model.site(prefix + "torso").id
        gyro = self._mj_model.sensor(prefix + "torso_gyro")
        self._gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])

        self._default_pose = jp.asarray(self._mj_model.qpos0[self._joint_qpos])
        self._lowers = jp.asarray(
            [
                self._mj_model.joint(prefix + name).range[0]
                for name in self.contract.joint_order
            ]
        )
        self._uppers = jp.asarray(
            [
                self._mj_model.joint(prefix + name).range[1]
                for name in self.contract.joint_order
            ]
        )
        self._action_scale = jp.asarray(self._config.action_scale)

    def _configure_pd_actuators(self) -> None:
        for effector in self.contract.effector_order:
            pos_id = self._mj_model.actuator(self.prefix + effector + "_pos").id
            vel_id = self._mj_model.actuator(self.prefix + effector + "_vel").id
            self._mj_model.actuator_gainprm[pos_id, 0] = self._config.kp
            self._mj_model.actuator_biasprm[pos_id, 1] = -self._config.kp
            self._mj_model.actuator_gainprm[vel_id, 0] = self._config.kd
            self._mj_model.actuator_biasprm[vel_id, 2] = -self._config.kd

    @property
    def xml_path(self) -> str:
        return str(self._resource_root / "environments" / "soccer" / "world.xml")

    @property
    def action_size(self) -> int:
        return self.contract.action_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            ball_x_rng,
            ball_y_rng,
            target_rng,
            distance_rng,
            arrival_rng,
            mode_rng,
        ) = jax.random.split(rng, 7)
        qpos = jp.asarray(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)
        ball_x = jax.random.uniform(
            ball_x_rng,
            minval=self._config.ball_x_range[0],
            maxval=self._config.ball_x_range[1],
        )
        ball_y = jax.random.uniform(
            ball_y_rng,
            minval=self._config.ball_y_range[0],
            maxval=self._config.ball_y_range[1],
        )
        ball_pos = jp.array([ball_x, ball_y, 0.11])
        qpos = qpos.at[self._ball_qpos : self._ball_qpos + 3].set(ball_pos)

        target_angle = jax.random.uniform(
            target_rng,
            minval=self._config.target_angle_range[0],
            maxval=self._config.target_angle_range[1],
        )
        target_world = jp.array([jp.cos(target_angle), jp.sin(target_angle)])
        target_distance = jax.random.uniform(
            distance_rng,
            minval=self._config.target_distance_range[0],
            maxval=self._config.target_distance_range[1],
        )
        action_mode_index = jax.random.randint(mode_rng, (), 0, 3)
        arrival_min = jp.array([0.4, 1.5, 1.0])[action_mode_index]
        arrival_max = jp.array([1.2, 2.5, 2.0])[action_mode_index]
        desired_arrival_speed = jax.random.uniform(
            arrival_rng, minval=arrival_min, maxval=arrival_max
        )
        requested_ball_speed = jp.sqrt(
            desired_arrival_speed * desired_arrival_speed
            + 2.0 * self._config.rolling_deceleration_mps2 * target_distance
        )
        action_mode = jax.nn.one_hot(action_mode_index, 3)

        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._pos_actuator].set(self._default_pose)
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
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "last_action": jp.zeros(self.action_size),
            "target_world": target_world,
            "target_distance": target_distance,
            "requested_ball_speed": requested_ball_speed,
            "desired_arrival_speed": desired_arrival_speed,
            "action_mode": action_mode,
            "initial_ball_xy": ball_pos[:2],
            "last_progress": jp.array(0.0),
        }
        metrics = {
            "reward/directional_velocity": jp.array(0.0),
            "reward/ball_progress": jp.array(0.0),
            "reward/target_range": jp.array(0.0),
            "reward/arrival_speed": jp.array(0.0),
            "reward/corridor": jp.array(0.0),
            "reward/upright": jp.array(0.0),
            "cost/action_rate": jp.array(0.0),
            "cost/fall": jp.array(0.0),
            "cost/overshoot": jp.array(0.0),
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
        targets = jp.clip(
            self._default_pose + action * self._action_scale,
            self._lowers,
            self._uppers,
        )
        ctrl = state.data.ctrl.at[self._pos_actuator].set(targets)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        torso_xmat = data.site_xmat[self._torso_site]
        torso_height = data.xpos[self._torso_body, 2]
        ball_xy = data.xpos[self._ball_body, :2]
        ball_vel_xy = data.qvel[self._ball_dof : self._ball_dof + 2]
        target = state.info["target_world"]
        displacement = ball_xy - state.info["initial_ball_xy"]
        progress = jp.dot(displacement, target)
        lateral_error = jp.dot(displacement, jp.array([-target[1], target[0]]))
        directional_velocity = jp.clip(jp.dot(ball_vel_xy, target), 0.0, 6.0)
        progress_delta = jp.clip(progress - state.info["last_progress"], -0.1, 0.1)
        remaining_distance = state.info["target_distance"] - progress
        target_range = jp.exp(-2.0 * jp.abs(remaining_distance))
        arrival_speed = target_range * jp.exp(
            -2.0 * jp.abs(directional_velocity - state.info["desired_arrival_speed"])
        )
        corridor = jp.exp(-8.0 * jp.square(lateral_error))
        before_target = (remaining_distance >= -0.25).astype(jp.float32)
        overshoot = jp.maximum(-remaining_distance, 0.0)
        upright = jp.clip(torso_xmat[2, 2], 0.0, 1.0)
        action_rate = jp.sum(jp.square(action - state.info["last_action"]))
        fall = (torso_height < 0.35) | (torso_xmat[2, 2] < 0.0)
        invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        reward_terms = {
            "directional_velocity": 1.5 * directional_velocity * before_target,
            "ball_progress": 20.0 * progress_delta,
            "target_range": 1.0 * target_range,
            "arrival_speed": 0.75 * arrival_speed,
            "corridor": 0.15 * corridor,
            "upright": 0.2 * upright,
            "action_rate": -0.01 * action_rate,
            "fall": -5.0 * fall.astype(jp.float32),
            "overshoot": -2.0 * overshoot,
        }
        reward = sum(reward_terms.values()) * self.dt

        state.info["step"] += 1
        state.info["last_action"] = action
        state.info["last_progress"] = progress
        done = fall | invalid | (state.info["step"] >= self._config.episode_length)
        obs = self._get_obs(data, state.info)
        state.metrics.update(
            {
                "reward/directional_velocity": directional_velocity,
                "reward/ball_progress": progress_delta,
                "reward/target_range": target_range,
                "reward/arrival_speed": arrival_speed,
                "reward/corridor": corridor,
                "reward/upright": upright,
                "cost/action_rate": action_rate,
                "cost/fall": fall.astype(jp.float32),
                "cost/overshoot": overshoot,
            }
        )
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        torso_xmat = data.site_xmat[self._torso_site]
        yaw = jp.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        world_to_yaw = jp.array([[c, s], [-s, c]])
        torso_pos = data.xpos[self._torso_body]
        ball_pos = data.xpos[self._ball_body]
        ball_world_vel = data.qvel[self._ball_dof : self._ball_dof + 3]
        torso_world_vel = data.qvel[self._root_dof : self._root_dof + 3]

        ball_local_xy = world_to_yaw @ (ball_pos[:2] - torso_pos[:2])
        ball_local_vel_xy = world_to_yaw @ (ball_world_vel[:2] - torso_world_vel[:2])
        target_local = world_to_yaw @ info["target_world"]
        gravity = torso_xmat.T @ jp.array([0.0, 0.0, -1.0])
        phase = 2.0 * jp.pi * 1.5 * info["step"] * self.dt

        actor = jp.concatenate(
            [
                data.sensordata[self._gyro_slice],
                gravity,
                data.qpos[self._joint_qpos] - self._default_pose,
                data.qvel[self._joint_dof],
                info["last_action"],
                jp.array(
                    [ball_local_xy[0], ball_local_xy[1], ball_pos[2] - torso_pos[2]]
                ),
                jp.array(
                    [ball_local_vel_xy[0], ball_local_vel_xy[1], ball_world_vel[2]]
                ),
                target_local,
                jp.array([info["target_distance"]]),
                jp.array([info["requested_ball_speed"]]),
                jp.array([info["desired_arrival_speed"]]),
                info["action_mode"],
                jp.array([0.0, 1.0]),
                jp.array([jp.sin(phase), jp.cos(phase)]),
                jp.array([0.0, 1.0, 0.0]),
            ]
        )
        privileged = jp.concatenate(
            [
                actor,
                torso_world_vel,
                ball_pos,
                ball_world_vel,
                jp.array([torso_pos[2]]),
            ]
        )
        return {"state": actor, "privileged_state": privileged}
