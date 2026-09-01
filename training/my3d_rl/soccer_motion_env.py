"""Finite multi-motion residual tracking for T1 soccer skills."""

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
from .soccer_motion_corpus import SoccerMotionCorpus
from .t1_control import apollo_joint_gains


DEFAULT_CONTRACT = (
    Path(__file__).parents[1] / "contracts" / "soccer_motion_policy_v1.yaml"
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=320,
        impl="jax",
        naconmax=2048,
        njmax=256,
        action_clip=1.0,
        action_scale=0.15,
        reset_joint_noise=0.002,
        reset_root_velocity_noise=0.005,
        reset_yaw_range=0.01,
        fixed_motion_index=-1,
        fixed_start_frame_min=-1,
        fixed_start_frame_max=-1,
        action_delay_max_steps=0,
        foot_contact_tolerance=0.0,
        joint_position_scale=4.6,
        joint_velocity_scale=110.0,
        previous_action_scale=10.0,
        angular_velocity_scale=50.0,
        reference_position_scale=4.6,
        reference_linear_velocity_scale=5.0,
        reference_angular_velocity_scale=10.0,
        observation_clip=10.0,
        reward=config_dict.create(
            motion_joint=8.0,
            motion_joint_velocity=1.0,
            motion_root_position=2.0,
            motion_root_orientation=2.0,
            motion_root_velocity=1.0,
            motion_contact=2.0,
            upright=0.5,
            alive=0.5,
            residual=-0.02,
            action_rate=-0.05,
            action_acceleration=-0.01,
            joint_limit=-0.5,
            completion=20.0,
            fall=-100.0,
        ),
    )


class FiniteSoccerMotionTracking(mjx_env.MjxEnv):
    """Track one of many non-periodic kicks with a bounded residual actor."""

    def __init__(
        self,
        corpus: SoccerMotionCorpus,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
    ) -> None:
        self.contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        config = default_config() if config is None else config
        overrides = dict(config_overrides or {})
        if "action_scale" not in overrides and self.contract.action_scale is not None:
            overrides["action_scale"] = self.contract.action_scale
        super().__init__(config, overrides)
        if self.contract.policy_name not in {
            "soccer_motion_policy_v1",
            "soccer_motion_policy_v2",
            "soccer_ball_motion_policy_v1",
        }:
            raise ValueError("finite soccer tracking requires a soccer policy contract")
        expected_observation_size = (
            126
            if self.contract.policy_name == "soccer_ball_motion_policy_v1"
            else 110
        )
        if (
            self.contract.observation_size != expected_observation_size
            or self.contract.action_size != 23
        ):
            raise ValueError("soccer motion policy has an incompatible boundary")
        if self.contract.action_scale is None or not np.isclose(
            self._config.action_scale, self.contract.action_scale
        ):
            raise ValueError("environment action scale differs from policy contract")
        self.corpus = corpus
        if not -1 <= self._config.fixed_motion_index < corpus.motion_count:
            raise ValueError("fixed_motion_index is outside the motion corpus")
        fixed_start_values = (
            self._config.fixed_start_frame_min,
            self._config.fixed_start_frame_max,
        )
        if (fixed_start_values[0] < 0) != (fixed_start_values[1] < 0):
            raise ValueError("fixed start-frame bounds must be enabled together")
        if fixed_start_values[0] >= 0 and fixed_start_values[0] > fixed_start_values[1]:
            raise ValueError("fixed start-frame bounds must be increasing")
        self.prefix = prefix
        self._resource_root = resource_root

        self._mj_model = build_single_t1_soccer_model(
            resource_root, prefix=prefix, robot_x=-10.0, robot_y=0.0
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._configure_pd_actuators()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._joint_qpos = np.asarray(
            [
                self._mj_model.joint(prefix + name).qposadr[0]
                for name in self.contract.joint_order
            ]
        )
        self._joint_dof = np.asarray(
            [
                self._mj_model.joint(prefix + name).dofadr[0]
                for name in self.contract.joint_order
            ]
        )
        root = self._mj_model.joint(prefix + "root")
        self._root_qpos = int(root.qposadr[0])
        self._root_dof = int(root.dofadr[0])
        self._model_root_xy = jp.asarray(
            self._mj_model.qpos0[self._root_qpos : self._root_qpos + 2]
        )
        self._torso_body = self._mj_model.body(prefix + "torso").id
        self._torso_site = self._mj_model.site(prefix + "torso").id
        self._left_foot_geom = self._mj_model.geom(prefix + "left_foot").id
        self._right_foot_geom = self._mj_model.geom(prefix + "right_foot").id
        self._left_foot_half_size = jp.asarray(
            self._mj_model.geom_size[self._left_foot_geom]
        )
        self._right_foot_half_size = jp.asarray(
            self._mj_model.geom_size[self._right_foot_geom]
        )
        self._pitch_height = float(self._mj_model.geom("pitch").pos[2])
        self._pos_actuator = np.asarray(
            [
                self._mj_model.actuator(prefix + name + "_pos").id
                for name in self.contract.effector_order
            ]
        )
        gyro = self._mj_model.sensor(prefix + "torso_gyro")
        self._gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
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

        self._lengths = jp.asarray(corpus.lengths)
        self._root_position = jp.asarray(corpus.root_position)
        self._root_quaternion = jp.asarray(corpus.root_quaternion_wxyz)
        self._root_linear_velocity = jp.asarray(corpus.root_linear_velocity)
        self._root_angular_velocity = jp.asarray(corpus.root_angular_velocity)
        self._joint_position = jp.asarray(corpus.joint_position)
        self._joint_velocity = jp.asarray(corpus.joint_velocity)
        self._foot_contact = jp.asarray(corpus.foot_contact)
        self._kick_leg = jp.asarray(corpus.kick_leg_one_hot)
        reset_weights = np.asarray(corpus.reset_weights, dtype=np.float32)
        self._reset_logits = jp.log(jp.asarray(np.maximum(reset_weights, 1.0e-30)))

    def _configure_pd_actuators(self) -> None:
        for joint_name, effector in zip(
            self.contract.joint_order, self.contract.effector_order, strict=True
        ):
            pos_id = self._mj_model.actuator(self.prefix + effector + "_pos").id
            vel_id = self._mj_model.actuator(self.prefix + effector + "_vel").id
            kp, kd = apollo_joint_gains(joint_name)
            self._mj_model.actuator_gainprm[pos_id, 0] = kp
            self._mj_model.actuator_biasprm[pos_id, 1] = -kp
            self._mj_model.actuator_gainprm[vel_id, 0] = kd
            self._mj_model.actuator_biasprm[vel_id, 2] = -kd

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

    def _reference(self, motion: jax.Array, frame: jax.Array) -> tuple[jax.Array, ...]:
        return (
            self._root_position[motion, frame],
            self._root_quaternion[motion, frame],
            self._root_linear_velocity[motion, frame],
            self._root_angular_velocity[motion, frame],
            self._joint_position[motion, frame],
            self._joint_velocity[motion, frame],
            self._foot_contact[motion, frame],
        )

    @staticmethod
    def _yaw_quaternion_rotate(
        quaternion: jax.Array, yaw: jax.Array
    ) -> jax.Array:
        half = 0.5 * yaw
        yaw_w = jp.cos(half)
        yaw_z = jp.sin(half)
        w, x, y, z = quaternion
        return jp.array(
            [
                yaw_w * w - yaw_z * z,
                yaw_w * x - yaw_z * y,
                yaw_w * y + yaw_z * x,
                yaw_w * z + yaw_z * w,
            ]
        )

    @staticmethod
    def _yaw_vector_rotate(vector: jax.Array, yaw: jax.Array) -> jax.Array:
        cosine, sine = jp.cos(yaw), jp.sin(yaw)
        return jp.array(
            [
                cosine * vector[0] - sine * vector[1],
                sine * vector[0] + cosine * vector[1],
                vector[2],
            ]
        )

    def decode_action_targets(
        self, action: jax.Array, motion: jax.Array, frame: jax.Array
    ) -> jax.Array:
        reference = self._joint_position[motion, frame]
        residual = self._config.action_scale * jp.clip(
            action, -self._config.action_clip, self._config.action_clip
        )
        return jp.clip(reference + residual, self._lowers, self._uppers)

    def _step_targets(
        self,
        state: mjx_env.State,
        applied_action: jax.Array,
        motion: jax.Array,
        frame: jax.Array,
    ) -> jax.Array:
        """Control hook retained by ball-conditioned post-contact recovery."""
        del state
        return self.decode_action_targets(applied_action, motion, frame)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            motion_rng,
            frame_rng,
            joint_rng,
            velocity_rng,
            yaw_rng,
            delay_rng,
        ) = jax.random.split(rng, 7)
        motion = (
            jp.asarray(self._config.fixed_motion_index, dtype=jp.int32)
            if self._config.fixed_motion_index >= 0
            else jax.random.randint(
                motion_rng, (), 0, self.corpus.motion_count, dtype=jp.int32
            )
        )
        frame = (
            jax.random.randint(
                frame_rng,
                (),
                self._config.fixed_start_frame_min,
                self._config.fixed_start_frame_max + 1,
                dtype=jp.int32,
            )
            if self._config.fixed_start_frame_min >= 0
            else jax.random.categorical(frame_rng, self._reset_logits[motion])
        )
        (
            root_position,
            root_quaternion,
            root_linear_velocity,
            root_angular_velocity,
            joint_position,
            joint_velocity,
            _,
        ) = self._reference(motion, frame)
        yaw = jax.random.uniform(
            yaw_rng,
            minval=-self._config.reset_yaw_range,
            maxval=self._config.reset_yaw_range,
        )
        joint_noise = jax.random.uniform(
            joint_rng,
            (self.action_size,),
            minval=-self._config.reset_joint_noise,
            maxval=self._config.reset_joint_noise,
        )
        velocity_noise = jax.random.uniform(
            velocity_rng,
            (6,),
            minval=-self._config.reset_root_velocity_noise,
            maxval=self._config.reset_root_velocity_noise,
        )
        qpos = jp.asarray(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)
        qpos = qpos.at[self._root_qpos : self._root_qpos + 2].set(
            self._model_root_xy
        )
        qpos = qpos.at[self._root_qpos + 2].set(root_position[2])
        qpos = qpos.at[self._root_qpos + 3 : self._root_qpos + 7].set(
            self._yaw_quaternion_rotate(root_quaternion, yaw)
        )
        qpos = qpos.at[self._joint_qpos].set(
            jp.clip(joint_position + joint_noise, self._lowers, self._uppers)
        )
        root_velocity = jp.concatenate(
            [
                self._yaw_vector_rotate(root_linear_velocity, yaw),
                self._yaw_vector_rotate(root_angular_velocity, yaw),
            ]
        )
        qvel = qvel.at[self._root_dof : self._root_dof + 6].set(
            root_velocity + velocity_noise
        )
        qvel = qvel.at[self._joint_dof].set(joint_velocity)
        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._pos_actuator].set(joint_position)
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
        delay_steps = jax.random.randint(
            delay_rng, (), 0, self._config.action_delay_max_steps + 1
        )
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "motion": motion,
            "reference_frame": frame,
            "reference_start_xy": root_position[:2],
            "yaw": yaw,
            "delay_steps": delay_steps,
            "last_action": jp.zeros(self.action_size),
            "last_last_action": jp.zeros(self.action_size),
        }
        metrics = {
            "reward/motion_joint": jp.array(0.0),
            "reward/motion_joint_velocity": jp.array(0.0),
            "reward/motion_root_position": jp.array(0.0),
            "reward/motion_root_orientation": jp.array(0.0),
            "reward/motion_root_velocity": jp.array(0.0),
            "reward/motion_contact": jp.array(0.0),
            "reward/upright": jp.array(0.0),
            "reward/alive": jp.array(0.0),
            "reward/completion": jp.array(0.0),
            "cost/residual": jp.array(0.0),
            "cost/action_rate": jp.array(0.0),
            "cost/action_acceleration": jp.array(0.0),
            "cost/joint_limit": jp.array(0.0),
            "cost/fall": jp.array(0.0),
            "diagnostic/reference_phase": jp.array(0.0),
            "diagnostic/torso_height": jp.array(0.0),
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
        action = jp.clip(action, -self._config.action_clip, self._config.action_clip)
        applied_action = jp.where(
            state.info["delay_steps"] > 0, state.info["last_action"], action
        )
        motion = state.info["motion"]
        length = self._lengths[motion]
        frame = jp.minimum(state.info["reference_frame"] + 1, length - 1)
        targets = self._step_targets(
            state, applied_action, motion, frame
        )
        ctrl = state.data.ctrl.at[self._pos_actuator].set(targets)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        (
            reference_root_position,
            reference_root_quaternion,
            reference_root_linear_velocity,
            reference_root_angular_velocity,
            reference_joint_position,
            reference_joint_velocity,
            reference_contact,
        ) = self._reference(motion, frame)

        joint_error = data.qpos[self._joint_qpos] - reference_joint_position
        joint_velocity_error = data.qvel[self._joint_dof] - reference_joint_velocity
        motion_joint = jp.exp(-4.0 * jp.mean(jp.square(joint_error)))
        motion_joint_velocity = jp.exp(
            -0.02 * jp.mean(jp.square(joint_velocity_error))
        )
        desired_root_position = reference_root_position.at[:2].set(
            self._model_root_xy
            + reference_root_position[:2]
            - state.info["reference_start_xy"]
        )
        root_position_error = jp.sum(
            jp.square(data.qpos[self._root_qpos : self._root_qpos + 3] - desired_root_position)
        )
        motion_root_position = jp.exp(-8.0 * root_position_error)
        desired_quaternion = self._yaw_quaternion_rotate(
            reference_root_quaternion, state.info["yaw"]
        )
        quaternion_dot = jp.dot(
            data.qpos[self._root_qpos + 3 : self._root_qpos + 7],
            desired_quaternion,
        )
        motion_root_orientation = jp.exp(-4.0 * (1.0 - quaternion_dot**2))
        desired_root_velocity = jp.concatenate(
            [
                self._yaw_vector_rotate(
                    reference_root_linear_velocity, state.info["yaw"]
                ),
                self._yaw_vector_rotate(
                    reference_root_angular_velocity, state.info["yaw"]
                ),
            ]
        )
        root_velocity_error = jp.mean(
            jp.square(
                data.qvel[self._root_dof : self._root_dof + 6]
                - desired_root_velocity
            )
        )
        motion_root_velocity = jp.exp(-0.5 * root_velocity_error)
        left_contact, right_contact = self._foot_contacts(data)
        actual_contact = jp.array([left_contact, right_contact])
        motion_contact = jp.mean(
            (actual_contact == reference_contact).astype(jp.float32)
        )
        torso_xmat = data.site_xmat[self._torso_site]
        upright = torso_xmat[2, 2]
        torso_height = data.xpos[self._torso_body, 2]
        fall = (torso_height < 0.35) | (upright < 0.20)
        invalid = (
            jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
            | jp.isnan(action).any()
        )
        completed = (frame >= length - 1) & ~fall & ~invalid
        residual = jp.mean(jp.square(action))
        action_rate = jp.mean(jp.square(action - state.info["last_action"]))
        action_acceleration = jp.mean(
            jp.square(
                action
                - 2.0 * state.info["last_action"]
                + state.info["last_last_action"]
            )
        )
        q = data.qpos[self._joint_qpos]
        span = self._uppers - self._lowers
        soft_lower = self._lowers + 0.03 * span
        soft_upper = self._uppers - 0.03 * span
        joint_limit = jp.mean(jp.square(jp.maximum(soft_lower - q, 0.0)))
        joint_limit += jp.mean(jp.square(jp.maximum(q - soft_upper, 0.0)))

        terms = {
            "motion_joint": self._config.reward.motion_joint * motion_joint,
            "motion_joint_velocity": self._config.reward.motion_joint_velocity
            * motion_joint_velocity,
            "motion_root_position": self._config.reward.motion_root_position
            * motion_root_position,
            "motion_root_orientation": self._config.reward.motion_root_orientation
            * motion_root_orientation,
            "motion_root_velocity": self._config.reward.motion_root_velocity
            * motion_root_velocity,
            "motion_contact": self._config.reward.motion_contact * motion_contact,
            "upright": self._config.reward.upright * jp.clip(upright, 0.0, 1.0),
            "alive": self._config.reward.alive * (~fall).astype(jp.float32),
            "residual": self._config.reward.residual * residual,
            "action_rate": self._config.reward.action_rate * action_rate,
            "action_acceleration": self._config.reward.action_acceleration
            * action_acceleration,
            "joint_limit": self._config.reward.joint_limit * joint_limit,
            "completion": self._config.reward.completion
            / self.dt
            * completed.astype(jp.float32),
            "fall": self._config.reward.fall
            / self.dt
            * fall.astype(jp.float32),
        }
        reward = sum(terms.values()) * self.dt
        state.info["step"] += 1
        state.info["reference_frame"] = frame
        state.info["last_last_action"] = state.info["last_action"]
        state.info["last_action"] = action
        timeout = state.info["step"] >= self._config.episode_length
        done = fall | invalid | completed | timeout
        obs = self._get_obs(data, state.info)
        phase = frame / jp.maximum(length - 1, 1)
        state.metrics.update(
            {
                "reward/motion_joint": motion_joint,
                "reward/motion_joint_velocity": motion_joint_velocity,
                "reward/motion_root_position": motion_root_position,
                "reward/motion_root_orientation": motion_root_orientation,
                "reward/motion_root_velocity": motion_root_velocity,
                "reward/motion_contact": motion_contact,
                "reward/upright": upright,
                "reward/alive": (~fall).astype(jp.float32),
                "reward/completion": completed.astype(jp.float32),
                "cost/residual": residual,
                "cost/action_rate": action_rate,
                "cost/action_acceleration": action_acceleration,
                "cost/joint_limit": joint_limit,
                "cost/fall": fall.astype(jp.float32),
                "diagnostic/reference_phase": phase,
                "diagnostic/torso_height": torso_height,
            }
        )
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _foot_contacts(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        def geometric_contact(geom_id: int, half_size: jax.Array) -> jax.Array:
            rotation = data.geom_xmat[geom_id]
            lowest_z = data.geom_xpos[geom_id, 2] - jp.sum(
                jp.abs(rotation[2, :]) * half_size
            )
            return lowest_z <= (
                self._pitch_height + self._config.foot_contact_tolerance
            )

        return (
            geometric_contact(self._left_foot_geom, self._left_foot_half_size),
            geometric_contact(self._right_foot_geom, self._right_foot_half_size),
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        motion = info["motion"]
        frame = info["reference_frame"]
        length = self._lengths[motion]
        (
            _,
            _,
            root_linear_velocity,
            root_angular_velocity,
            joint_position,
            joint_velocity,
            contact,
        ) = self._reference(motion, frame)
        joint_triplets = jp.stack(
            [
                (data.qpos[self._joint_qpos] - joint_position)
                / self._config.joint_position_scale,
                (data.qvel[self._joint_dof] - joint_velocity)
                / self._config.joint_velocity_scale,
                info["last_action"] / self._config.previous_action_scale,
            ],
            axis=1,
        ).reshape(-1)
        angular_velocity = (
            data.sensordata[self._gyro_slice] / self._config.angular_velocity_scale
        )
        gravity = data.site_xmat[self._torso_site].T @ jp.array([0.0, 0.0, -1.0])
        progress = frame / jp.maximum(length - 1, 1)
        progress_angle = 2.0 * jp.pi * progress
        actor = jp.concatenate(
            [
                joint_triplets,
                angular_velocity,
                gravity,
                joint_position / self._config.reference_position_scale,
                root_linear_velocity
                / self._config.reference_linear_velocity_scale,
                root_angular_velocity
                / self._config.reference_angular_velocity_scale,
                contact.astype(jp.float32),
                jp.array([jp.cos(progress_angle), jp.sin(progress_angle)]),
                self._kick_leg[motion],
            ]
        )
        actor = jp.clip(
            jp.nan_to_num(actor, nan=0.0, posinf=10.0, neginf=-10.0),
            -self._config.observation_clip,
            self._config.observation_clip,
        )
        root_velocity = data.qvel[self._root_dof : self._root_dof + 6]
        torso_xmat = data.site_xmat[self._torso_site]
        privileged = jp.concatenate(
            [
                actor,
                root_velocity,
                jp.array([torso_xmat[2, 2], data.xpos[self._torso_body, 2]]),
            ]
        )
        return {"state": actor, "privileged_state": privileged}
