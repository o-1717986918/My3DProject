"""Velocity-tracking T1 task with the competition walk-policy boundary."""

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


DEFAULT_CONTRACT = Path(__file__).parents[1] / "contracts" / "run_policy_v1.yaml"

# These constants intentionally mirror mujococodebase/skills/walk/walk.py.
NOMINAL_TRAINING_POSE = np.array(
    [
        0.0,
        0.0,
        0.0,
        1.4,
        0.0,
        -0.4,
        0.0,
        -1.4,
        0.0,
        0.4,
        0.0,
        -0.4,
        0.0,
        0.0,
        0.8,
        -0.4,
        0.0,
        0.4,
        0.0,
        0.0,
        -0.8,
        0.4,
        0.0,
    ],
    dtype=np.float32,
)
TRAIN_TO_SERVER_SIGN = np.array(
    [
        1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ],
    dtype=np.float32,
)


def default_config() -> config_dict.ConfigDict:
    """Return the stage-one configuration; trainers override command ranges."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=500,
        impl="jax",
        naconmax=2048,
        njmax=256,
        kp=25.0,
        kd=0.6,
        action_clip=1.0,
        action_scale=0.5,
        joint_position_scale=4.6,
        joint_velocity_scale=110.0,
        previous_action_scale=10.0,
        angular_velocity_scale=50.0,
        observation_clip=10.0,
        lin_vel_x=[0.0, 0.8],
        lin_vel_y=[0.0, 0.0],
        ang_vel_yaw=[0.0, 0.0],
        gait_frequency=[1.0, 2.0],
        swing_period=0.20,
        stand_probability=0.2,
        axis_aligned_command_probability=0.0,
        axis_command_weights=[1.0, 1.0, 1.0],
        minimum_abs_yaw=0.0,
        yaw_negative_probability=0.5,
        command_resample_steps=500,
        fixed_command=[0.0, 0.0, 0.0],
        use_fixed_command=False,
        reset_joint_noise=0.03,
        reset_joint_velocity_noise=0.0,
        reset_policy_action_noise=0.0,
        reset_root_velocity_noise=0.10,
        reset_yaw_range=0.20,
        reference_init_probability=0.0,
        reference_phase_sampling_weights=[],
        push_enable=False,
        push_interval_steps=150,
        push_magnitude=[0.05, 0.25],
        action_delay_max_steps=0,
        foot_contact_tolerance=0.01,
        reward=config_dict.create(
            tracking_linear=3.0,
            tracking_yaw=1.0,
            upright=0.5,
            height=1.0,
            alive=0.2,
            flight=0.0,
            single_support=0.0,
            phase_swing=0.0,
            motion_joint=0.0,
            motion_joint_velocity=0.0,
            motion_contact=0.0,
            motion_action=0.0,
            lateral_tracking=0.0,
            yaw_rate_error=0.0,
            vertical_velocity=-0.25,
            angular_xy=-0.10,
            action_rate=-0.015,
            action_acceleration=-0.005,
            joint_velocity=-0.00005,
            foot_slip=0.0,
            pose=-0.02,
            joint_limit=-0.5,
            fall=-20.0,
            tracking_sigma=0.25,
            yaw_tracking_sigma=0.25,
            height_target=0.65,
            height_sigma=0.02,
        ),
    )


class DirectionalRun(mjx_env.MjxEnv):
    """Train fast locomotion without changing the deployed 78 -> 23 contract."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
        motion_reference: Path | None = None,
    ) -> None:
        self.contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        config = default_config() if config is None else config
        config_overrides = dict(config_overrides or {})
        if "action_scale" not in config_overrides and self.contract.action_scale:
            config_overrides["action_scale"] = self.contract.action_scale
        if "kp" not in config_overrides and self.contract.kp is not None:
            config_overrides["kp"] = self.contract.kp
        if "kd" not in config_overrides and self.contract.kd is not None:
            config_overrides["kd"] = self.contract.kd
        super().__init__(config, config_overrides)
        self.prefix = prefix
        self._resource_root = resource_root
        if not 0.0 <= self._config.stand_probability <= 1.0:
            raise ValueError("stand_probability must be in [0, 1]")
        if not 0.0 <= self._config.axis_aligned_command_probability <= 1.0:
            raise ValueError("axis_aligned_command_probability must be in [0, 1]")
        axis_command_weights = np.asarray(
            self._config.axis_command_weights, dtype=np.float64
        )
        if (
            axis_command_weights.shape != (3,)
            or not np.isfinite(axis_command_weights).all()
            or np.any(axis_command_weights < 0.0)
            or not np.sum(axis_command_weights) > 0.0
        ):
            raise ValueError(
                "axis_command_weights must contain three finite non-negative "
                "values with positive sum"
            )
        self._axis_command_logits = jp.log(
            jp.asarray(
                np.maximum(
                    axis_command_weights / np.sum(axis_command_weights), 1.0e-30
                ),
                dtype=jp.float32,
            )
        )
        if self._config.minimum_abs_yaw < 0.0:
            raise ValueError("minimum_abs_yaw must be non-negative")
        if not 0.0 <= self._config.yaw_negative_probability <= 1.0:
            raise ValueError("yaw_negative_probability must be in [0, 1]")
        if self._config.minimum_abs_yaw > 0.0 and not (
            self._config.ang_vel_yaw[0] <= -self._config.minimum_abs_yaw
            and self._config.ang_vel_yaw[1] >= self._config.minimum_abs_yaw
        ):
            raise ValueError(
                "minimum_abs_yaw requires a yaw range spanning both signs"
            )
        if self._config.command_resample_steps < 1:
            raise ValueError("command_resample_steps must be positive")
        for name in ("reset_joint_velocity_noise", "reset_policy_action_noise"):
            value = float(self._config[name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.contract.action_scale is None:
            raise ValueError("run policy contract must declare action_scale")
        if not np.isclose(self._config.action_scale, self.contract.action_scale):
            raise ValueError("environment action_scale differs from policy contract")
        if self.contract.kp is None or self.contract.kd is None:
            raise ValueError("run policy contract must declare PD gains")
        if not np.isclose(self._config.kp, self.contract.kp) or not np.isclose(
            self._config.kd, self.contract.kd
        ):
            raise ValueError("environment PD gains differ from policy contract")

        # Start far enough from the ball for a ten-second straight rollout.
        self._mj_model = build_single_t1_soccer_model(
            resource_root, prefix=prefix, robot_x=-10.0, robot_y=0.0
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
        self._root_qpos = self._mj_model.joint(prefix + "root").qposadr[0]
        self._root_dof = self._mj_model.joint(prefix + "root").dofadr[0]
        self._torso_body = self._mj_model.body(prefix + "torso").id
        self._torso_site = self._mj_model.site(prefix + "torso").id
        self._left_foot_site = self._mj_model.site(prefix + "lfoot-vismarker").id
        self._right_foot_site = self._mj_model.site(prefix + "rfoot-vismarker").id
        self._left_foot_geom = self._mj_model.geom(prefix + "left_foot").id
        self._right_foot_geom = self._mj_model.geom(prefix + "right_foot").id
        self._left_foot_half_size = jp.asarray(
            self._mj_model.geom_size[self._left_foot_geom]
        )
        self._right_foot_half_size = jp.asarray(
            self._mj_model.geom_size[self._right_foot_geom]
        )
        self._pitch_height = float(self._mj_model.geom("pitch").pos[2])
        self._pos_actuator = np.array(
            [
                self._mj_model.actuator(prefix + name + "_pos").id
                for name in self.contract.effector_order
            ]
        )
        gyro = self._mj_model.sensor(prefix + "torso_gyro")
        self._gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])

        self._nominal_training = jp.asarray(NOMINAL_TRAINING_POSE)
        self._sign = jp.asarray(TRAIN_TO_SERVER_SIGN)
        nominal_physical_unclipped = self._nominal_training * self._sign
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
        # Mirror the runtime motor-limit clamp so every residual target remains
        # valid even when exploration reaches the policy action boundary.
        self._nominal_physical = jp.clip(
            nominal_physical_unclipped, self._lowers, self._uppers
        )

        if (
            self.contract.observation_size not in (78, 80)
            or self.contract.action_size != 23
        ):
            raise ValueError(
                "run policies must preserve 23 actions and use the 78-value "
                "legacy or 80-value phase-aware actor boundary"
            )
        self._phase_observation = self.contract.observation_size == 80
        self._reference_centered = (
            self.contract.control_mode == "motion_reference_residual_joint_position"
        )
        self._motion_reference_path = motion_reference
        self._motion_tracking = motion_reference is not None
        if self._reference_centered and not self._phase_observation:
            raise ValueError(
                "reference-centred control requires gait phase observation"
            )
        if self._reference_centered and not self._motion_tracking:
            raise ValueError("reference-centred control requires a motion reference")
        if self._motion_tracking:
            with np.load(motion_reference, allow_pickle=False) as archive:
                reference_root_position = np.asarray(
                    archive["root_position"], dtype=np.float32
                )
                reference_root_quaternion = np.asarray(
                    archive["root_quaternion_xyzw"], dtype=np.float32
                )[:, [3, 0, 1, 2]]
                reference_root_linear_velocity = np.asarray(
                    archive["root_linear_velocity"], dtype=np.float32
                )
                reference_root_angular_velocity = np.asarray(
                    archive["root_angular_velocity"], dtype=np.float32
                )
                reference_position = np.asarray(
                    archive["joint_position"], dtype=np.float32
                )
                reference_velocity = np.asarray(
                    archive["joint_velocity"], dtype=np.float32
                )
                reference_contact = np.asarray(archive["foot_contact"], dtype=bool)
            if (
                reference_position.ndim != 2
                or reference_position.shape[1] != self.action_size
                or reference_position.shape != reference_velocity.shape
                or reference_contact.shape != (reference_position.shape[0], 2)
                or reference_root_position.shape != (reference_position.shape[0], 3)
                or reference_root_quaternion.shape != (reference_position.shape[0], 4)
                or reference_root_linear_velocity.shape
                != (reference_position.shape[0], 3)
                or reference_root_angular_velocity.shape
                != (reference_position.shape[0], 3)
            ):
                raise ValueError("motion reference has incompatible array shapes")
            if reference_position.shape[0] < 2:
                raise ValueError("motion reference requires at least two frames")
            self._reference_joint_position = jp.asarray(reference_position) * self._sign
            self._reference_joint_velocity = jp.asarray(reference_velocity) * self._sign
            self._reference_contact = jp.asarray(reference_contact)
            self._reference_root_position = jp.asarray(reference_root_position)
            self._reference_root_quaternion = jp.asarray(reference_root_quaternion)
            self._reference_root_linear_velocity = jp.asarray(
                reference_root_linear_velocity
            )
            self._reference_root_angular_velocity = jp.asarray(
                reference_root_angular_velocity
            )
            self._reference_frame_count = reference_position.shape[0]
            self._reference_nominal_frequency = (
                self.contract.frequency_hz / self._reference_frame_count
            )
            self._reference_forward_speed = float(
                np.mean(reference_root_linear_velocity[:, 0])
            )
            if self._reference_centered and self._reference_forward_speed <= 0.0:
                raise ValueError(
                    "reference-centred control requires positive forward speed"
                )
        else:
            self._reference_joint_position = jp.zeros((2, self.action_size))
            self._reference_joint_velocity = jp.zeros((2, self.action_size))
            self._reference_contact = jp.zeros((2, 2), dtype=bool)
            self._reference_root_position = jp.zeros((2, 3))
            self._reference_root_quaternion = jp.tile(
                jp.array([[1.0, 0.0, 0.0, 0.0]]), (2, 1)
            )
            self._reference_root_linear_velocity = jp.zeros((2, 3))
            self._reference_root_angular_velocity = jp.zeros((2, 3))
            self._reference_frame_count = 2
            self._reference_nominal_frequency = 1.0
            self._reference_forward_speed = 1.0

        phase_weights = np.asarray(
            self._config.reference_phase_sampling_weights, dtype=np.float64
        )
        self._weighted_reference_phase_reset = phase_weights.size > 0
        if self._weighted_reference_phase_reset:
            if phase_weights.shape != (self._reference_frame_count,):
                raise ValueError(
                    "reference phase weights must match the reference frame count"
                )
            if (
                not np.isfinite(phase_weights).all()
                or np.any(phase_weights < 0.0)
                or not np.sum(phase_weights) > 0.0
            ):
                raise ValueError(
                    "reference phase weights must be finite and non-negative"
                )
            phase_weights = phase_weights / np.sum(phase_weights)
            self._reference_phase_logits = jp.log(
                jp.asarray(np.maximum(phase_weights, 1.0e-30), dtype=jp.float32)
            )
        else:
            self._reference_phase_logits = jp.zeros(self._reference_frame_count)

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

    def decode_action_targets(
        self, action: jax.Array, gait_phase: jax.Array | None = None
    ) -> jax.Array:
        """Decode one policy action into clamped physical joint targets.

        v1/v2 centre on the fixed nominal pose.  v3 requires a phase and
        centres the bounded residual on the interpolated motion reference.
        """
        clipped_action = jp.clip(
            action, -self._config.action_clip, self._config.action_clip
        )
        if self._reference_centered:
            if gait_phase is None:
                raise ValueError("reference-centred decoder requires gait_phase")
            reference_position = self._reference_at_phase(gait_phase)[0]
        else:
            reference_position = self._nominal_training
        targets_training = (
            reference_position + self._config.action_scale * clipped_action
        )
        return jp.clip(targets_training * self._sign, self._lowers, self._uppers)

    def _phase_frequency_for_command(
        self, command: jax.Array, sampled_frequency: jax.Array
    ) -> jax.Array:
        """Scale the reference cadence by requested forward speed for v3."""
        reference_frequency = (
            self._reference_nominal_frequency
            * jp.abs(command[0])
            / self._reference_forward_speed
        )
        frequency = jp.where(
            self._reference_centered, reference_frequency, sampled_frequency
        )
        # Pure yaw is a real locomotion primitive, not a standing command. The
        # previous linear-only gate froze gait phase for every in-place turn
        # sample and made rapid-turn training internally inconsistent.
        return jp.where(jp.linalg.norm(command) > 1.0e-4, frequency, 0.0)

    def _reference_velocity_scale(self, gait_frequency: jax.Array) -> jax.Array:
        if not self._reference_centered:
            return jp.array(1.0)
        return gait_frequency / self._reference_nominal_frequency

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            joint_rng,
            vel_rng,
            yaw_rng,
            delay_rng,
            phase_rng,
            gait_rng,
            reference_rng,
            entry_action_rng,
            joint_velocity_rng,
        ) = jax.random.split(rng, 10)
        qpos = jp.asarray(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)

        rng, command_rng = jax.random.split(rng)
        command = self._sample_command(command_rng)
        sampled_gait_frequency = jax.random.uniform(
            gait_rng,
            minval=self._config.gait_frequency[0],
            maxval=self._config.gait_frequency[1],
        )
        gait_frequency = self._phase_frequency_for_command(
            command, sampled_gait_frequency
        )
        phase_bin_rng, phase_offset_rng = jax.random.split(phase_rng)
        if self._weighted_reference_phase_reset:
            phase_bin = jax.random.categorical(
                phase_bin_rng, self._reference_phase_logits
            )
            gait_phase = (
                phase_bin + jax.random.uniform(phase_offset_rng)
            ) / self._reference_frame_count
        else:
            gait_phase = jax.random.uniform(phase_rng)
        (
            reference_position,
            reference_velocity,
            _,
            reference_root_position,
            reference_root_quaternion,
            reference_root_linear_velocity,
            reference_root_angular_velocity,
        ) = self._reference_at_phase(
            gait_phase,
            velocity_scale=self._reference_velocity_scale(gait_frequency),
        )
        joint_noise = jax.random.uniform(
            joint_rng,
            (self.action_size,),
            minval=-self._config.reset_joint_noise,
            maxval=self._config.reset_joint_noise,
        )
        reference_init = self._motion_tracking & jax.random.bernoulli(
            reference_rng, self._config.reference_init_probability
        )
        sampled_entry_action = jax.random.uniform(
            entry_action_rng,
            (self.action_size,),
            minval=-self._config.reset_policy_action_noise,
            maxval=self._config.reset_policy_action_noise,
        )
        entry_action = jp.clip(
            jp.where(
                reference_init, jp.zeros(self.action_size), sampled_entry_action
            ),
            -self._config.action_clip,
            self._config.action_clip,
        )
        initial_training = jp.where(
            reference_init,
            reference_position,
            self._nominal_training + self._config.action_scale * entry_action,
        )
        initial_joints = jp.clip(
            initial_training * self._sign + joint_noise,
            self._lowers,
            self._uppers,
        )
        qpos = qpos.at[self._joint_qpos].set(initial_joints)

        yaw = jax.random.uniform(
            yaw_rng,
            minval=-self._config.reset_yaw_range,
            maxval=self._config.reset_yaw_range,
        )
        yaw_cos = jp.cos(0.5 * yaw)
        yaw_sin = jp.sin(0.5 * yaw)
        reference_w, reference_x, reference_y, reference_z = reference_root_quaternion
        motion_root_quaternion = jp.array(
            [
                yaw_cos * reference_w - yaw_sin * reference_z,
                yaw_cos * reference_x - yaw_sin * reference_y,
                yaw_cos * reference_y + yaw_sin * reference_x,
                yaw_cos * reference_z + yaw_sin * reference_w,
            ]
        )
        yaw_quaternion = jp.array([yaw_cos, 0.0, 0.0, yaw_sin])
        root_quat_wxyz = jp.where(
            reference_init, motion_root_quaternion, yaw_quaternion
        )
        qpos = qpos.at[self._root_qpos + 2].set(
            jp.where(
                reference_init,
                reference_root_position[2],
                qpos[self._root_qpos + 2],
            )
        )
        qpos = qpos.at[self._root_qpos + 3 : self._root_qpos + 7].set(root_quat_wxyz)
        root_velocity_noise = jax.random.uniform(
            vel_rng,
            (6,),
            minval=-self._config.reset_root_velocity_noise,
            maxval=self._config.reset_root_velocity_noise,
        )
        yaw_rotation = jp.array(
            [
                [jp.cos(yaw), -jp.sin(yaw), 0.0],
                [jp.sin(yaw), jp.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        reference_root_velocity = jp.concatenate(
            [
                yaw_rotation @ reference_root_linear_velocity,
                yaw_rotation @ reference_root_angular_velocity,
            ]
        )
        root_velocity = root_velocity_noise + jp.where(
            reference_init,
            reference_root_velocity,
            jp.zeros(6),
        )
        qvel = qvel.at[self._root_dof : self._root_dof + 6].set(root_velocity)
        sampled_joint_velocity = jax.random.uniform(
            joint_velocity_rng,
            (self.action_size,),
            minval=-self._config.reset_joint_velocity_noise,
            maxval=self._config.reset_joint_velocity_noise,
        )
        qvel = qvel.at[self._joint_dof].set(jp.where(
            reference_init,
            reference_velocity * self._sign,
            sampled_joint_velocity,
        ))

        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._pos_actuator].set(
            self.decode_action_targets(entry_action, gait_phase)
        )
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
            delay_rng,
            (),
            minval=0,
            maxval=self._config.action_delay_max_steps + 1,
        )
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "command": command,
            "gait_phase": gait_phase,
            "gait_frequency": gait_frequency,
            "last_action": entry_action,
            "last_last_action": entry_action,
            "delay_steps": delay_steps,
            "reference_init": reference_init,
            "last_foot_positions": jp.stack(
                [
                    data.site_xpos[self._left_foot_site],
                    data.site_xpos[self._right_foot_site],
                ]
            ),
        }
        metrics = {
            "reward/tracking_linear": jp.array(0.0),
            "reward/tracking_yaw": jp.array(0.0),
            "reward/upright": jp.array(0.0),
            "reward/height": jp.array(0.0),
            "reward/alive": jp.array(0.0),
            "reward/flight": jp.array(0.0),
            "reward/single_support": jp.array(0.0),
            "reward/phase_swing": jp.array(0.0),
            "reward/motion_joint": jp.array(0.0),
            "reward/motion_joint_velocity": jp.array(0.0),
            "reward/motion_contact": jp.array(0.0),
            "reward/motion_action": jp.array(0.0),
            "cost/vertical_velocity": jp.array(0.0),
            "cost/angular_xy": jp.array(0.0),
            "cost/action_rate": jp.array(0.0),
            "cost/action_acceleration": jp.array(0.0),
            "cost/joint_velocity": jp.array(0.0),
            "cost/foot_slip": jp.array(0.0),
            "cost/pose": jp.array(0.0),
            "cost/joint_limit": jp.array(0.0),
            "cost/fall": jp.array(0.0),
            "diagnostic/local_velocity_x": jp.array(0.0),
            "diagnostic/local_velocity_y": jp.array(0.0),
            "diagnostic/yaw_rate": jp.array(0.0),
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

        state.info["rng"], push_rng, command_rng = jax.random.split(
            state.info["rng"], 3
        )
        push_angle, push_magnitude_rng = jax.random.split(push_rng)
        theta = jax.random.uniform(push_angle, minval=0.0, maxval=2.0 * jp.pi)
        magnitude = jax.random.uniform(
            push_magnitude_rng,
            minval=self._config.push_magnitude[0],
            maxval=self._config.push_magnitude[1],
        )
        apply_push = (
            self._config.push_enable
            & (state.info["step"] > 0)
            & (state.info["step"] % self._config.push_interval_steps == 0)
        )
        push_xy = jp.array([jp.cos(theta), jp.sin(theta)]) * magnitude * apply_push
        qvel = state.data.qvel.at[self._root_dof : self._root_dof + 2].add(push_xy)
        data = state.data.replace(qvel=qvel)

        targets_physical = self.decode_action_targets(
            applied_action, state.info["gait_phase"]
        )
        ctrl = data.ctrl.at[self._pos_actuator].set(targets_physical)
        data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

        local_velocity, yaw_rate, upright, torso_height = self._base_diagnostics(data)
        linear_error = jp.sum(jp.square(state.info["command"][:2] - local_velocity[:2]))
        yaw_error = jp.square(state.info["command"][2] - yaw_rate)
        tracking_linear = jp.exp(-linear_error / self._config.reward.tracking_sigma)
        tracking_yaw = jp.exp(-yaw_error / self._config.reward.yaw_tracking_sigma)
        height_reward = jp.exp(
            -jp.square(torso_height - self._config.reward.height_target)
            / self._config.reward.height_sigma
        )
        left_contact, right_contact = self._foot_contacts(data)
        flight = (~left_contact) & (~right_contact)
        single_support = left_contact ^ right_contact
        gait_phase = state.info["gait_phase"]
        left_phase_error = jp.abs(gait_phase - 0.25)
        right_phase_error = jp.abs(gait_phase - 0.75)
        left_swing = left_phase_error < (0.5 * self._config.swing_period)
        right_swing = right_phase_error < (0.5 * self._config.swing_period)
        moving = state.info["gait_frequency"] > 1.0e-8
        phase_swing = (
            (left_swing & (~left_contact)).astype(jp.float32)
            + (right_swing & (~right_contact)).astype(jp.float32)
        ) * moving.astype(jp.float32)
        reference_position, reference_velocity, reference_contact, *_ = (
            self._reference_at_phase(
                gait_phase,
                velocity_scale=self._reference_velocity_scale(
                    state.info["gait_frequency"]
                ),
            )
        )
        joint_position_training = data.qpos[self._joint_qpos] * self._sign
        joint_velocity_training = data.qvel[self._joint_dof] * self._sign
        motion_joint = jp.exp(
            -2.0 * jp.mean(jp.square(joint_position_training - reference_position))
        )
        motion_joint_velocity = jp.exp(
            -0.01 * jp.mean(jp.square(joint_velocity_training - reference_velocity))
        )
        actual_contact = jp.array([left_contact, right_contact])
        foot_positions = jp.stack(
            [
                data.site_xpos[self._left_foot_site],
                data.site_xpos[self._right_foot_site],
            ]
        )
        foot_velocity_xy = (
            foot_positions[:, :2] - state.info["last_foot_positions"][:, :2]
        ) / self.dt
        contact_weight = actual_contact.astype(jp.float32)
        foot_slip = jp.sum(
            jp.sum(jp.square(foot_velocity_xy), axis=1) * contact_weight
        ) / jp.maximum(jp.sum(contact_weight), 1.0)
        motion_contact = jp.mean(
            (actual_contact == reference_contact).astype(jp.float32)
        )
        reference_action = jp.where(
            self._reference_centered,
            jp.zeros(self.action_size),
            jp.clip(
                (reference_position - self._nominal_training)
                / self._config.action_scale,
                -self._config.action_clip,
                self._config.action_clip,
            ),
        )
        motion_action = jp.exp(-0.25 * jp.mean(jp.square(action - reference_action)))
        tracking_enabled = jp.asarray(self._motion_tracking, dtype=jp.float32)
        motion_joint *= tracking_enabled
        motion_joint_velocity *= tracking_enabled
        motion_contact *= tracking_enabled
        motion_action *= tracking_enabled
        commanded_speed = jp.linalg.norm(state.info["command"][:2])
        locomotion_gate = jp.clip(commanded_speed / 1.2, 0.0, 1.0)
        action_rate = jp.sum(jp.square(action - state.info["last_action"]))
        action_acceleration = jp.sum(
            jp.square(
                action
                - 2.0 * state.info["last_action"]
                + state.info["last_last_action"]
            )
        )
        joint_velocity = jp.sum(jp.square(data.qvel[self._joint_dof]))
        pose_center = jp.where(
            self._reference_centered, reference_position, self._nominal_training
        )
        pose = jp.sum(jp.square(data.qpos[self._joint_qpos] * self._sign - pose_center))
        q = data.qpos[self._joint_qpos]
        span = self._uppers - self._lowers
        soft_lower = self._lowers + 0.03 * span
        soft_upper = self._uppers - 0.03 * span
        joint_limit = jp.sum(jp.square(jp.maximum(soft_lower - q, 0.0)))
        joint_limit += jp.sum(jp.square(jp.maximum(q - soft_upper, 0.0)))
        fall = (torso_height < 0.35) | (upright < 0.20)
        invalid = (
            jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
            | jp.isnan(action).any()
        )

        weighted_terms = {
            "tracking_linear": self._config.reward.tracking_linear * tracking_linear,
            "tracking_yaw": self._config.reward.tracking_yaw * tracking_yaw,
            "upright": self._config.reward.upright * jp.clip(upright, 0.0, 1.0),
            "height": self._config.reward.height * height_reward,
            "alive": self._config.reward.alive * (~fall).astype(jp.float32),
            "flight": self._config.reward.flight
            * flight.astype(jp.float32)
            * locomotion_gate,
            "single_support": self._config.reward.single_support
            * single_support.astype(jp.float32)
            * locomotion_gate,
            "phase_swing": self._config.reward.phase_swing * phase_swing,
            "motion_joint": self._config.reward.motion_joint * motion_joint,
            "motion_joint_velocity": self._config.reward.motion_joint_velocity
            * motion_joint_velocity,
            "motion_contact": self._config.reward.motion_contact * motion_contact,
            "motion_action": self._config.reward.motion_action * motion_action,
            "lateral_tracking": self._config.reward.lateral_tracking
            * jp.square(state.info["command"][1] - local_velocity[1]),
            "yaw_rate_error": self._config.reward.yaw_rate_error * yaw_error,
            "vertical_velocity": self._config.reward.vertical_velocity
            * jp.square(local_velocity[2]),
            "angular_xy": self._config.reward.angular_xy
            * jp.sum(jp.square(data.sensordata[self._gyro_slice][:2])),
            "action_rate": self._config.reward.action_rate * action_rate,
            "action_acceleration": self._config.reward.action_acceleration
            * action_acceleration,
            "joint_velocity": self._config.reward.joint_velocity * joint_velocity,
            "foot_slip": self._config.reward.foot_slip * foot_slip,
            "pose": self._config.reward.pose * pose,
            "joint_limit": self._config.reward.joint_limit * joint_limit,
            "fall": self._config.reward.fall * fall.astype(jp.float32),
        }
        reward = sum(weighted_terms.values()) * self.dt

        state.info["step"] += 1
        state.info["gait_phase"] = jp.mod(
            gait_phase + self.dt * state.info["gait_frequency"], 1.0
        )
        state.info["last_last_action"] = state.info["last_action"]
        state.info["last_action"] = action
        state.info["last_foot_positions"] = foot_positions
        resample = (state.info["step"] % self._config.command_resample_steps == 0) & (
            not self._config.use_fixed_command
        )
        next_command = jp.where(
            resample, self._sample_command(command_rng), state.info["command"]
        )
        state.info["command"] = next_command
        if self._reference_centered:
            state.info["gait_frequency"] = self._phase_frequency_for_command(
                next_command, state.info["gait_frequency"]
            )
        done = fall | invalid | (state.info["step"] >= self._config.episode_length)
        obs = self._get_obs(data, state.info)
        state.metrics.update(
            {
                "reward/tracking_linear": tracking_linear,
                "reward/tracking_yaw": tracking_yaw,
                "reward/upright": upright,
                "reward/height": height_reward,
                "reward/alive": (~fall).astype(jp.float32),
                "reward/flight": flight.astype(jp.float32),
                "reward/single_support": single_support.astype(jp.float32),
                "reward/phase_swing": phase_swing,
                "reward/motion_joint": motion_joint,
                "reward/motion_joint_velocity": motion_joint_velocity,
                "reward/motion_contact": motion_contact,
                "reward/motion_action": motion_action,
                "cost/vertical_velocity": jp.square(local_velocity[2]),
                "cost/angular_xy": jp.sum(
                    jp.square(data.sensordata[self._gyro_slice][:2])
                ),
                "cost/action_rate": action_rate,
                "cost/action_acceleration": action_acceleration,
                "cost/joint_velocity": joint_velocity,
                "cost/foot_slip": foot_slip,
                "cost/pose": pose,
                "cost/joint_limit": joint_limit,
                "cost/fall": fall.astype(jp.float32),
                "diagnostic/local_velocity_x": local_velocity[0],
                "diagnostic/local_velocity_y": local_velocity[1],
                "diagnostic/yaw_rate": yaw_rate,
                "diagnostic/torso_height": torso_height,
            }
        )
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
        )

    def _reference_at_phase(
        self, phase: jax.Array, velocity_scale: jax.Array | float = 1.0
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """Circularly interpolate reference state and use nearest contact phase."""
        frame = jp.mod(phase, 1.0) * self._reference_frame_count
        lower = jp.floor(frame).astype(jp.int32) % self._reference_frame_count
        upper = (lower + 1) % self._reference_frame_count
        fraction = frame - jp.floor(frame)
        position = (1.0 - fraction) * self._reference_joint_position[
            lower
        ] + fraction * self._reference_joint_position[upper]
        velocity = velocity_scale * (
            (1.0 - fraction) * self._reference_joint_velocity[lower]
            + fraction * self._reference_joint_velocity[upper]
        )
        root_position = (1.0 - fraction) * self._reference_root_position[
            lower
        ] + fraction * self._reference_root_position[upper]
        lower_quaternion = self._reference_root_quaternion[lower]
        upper_quaternion = self._reference_root_quaternion[upper]
        upper_quaternion = jp.where(
            jp.dot(lower_quaternion, upper_quaternion) < 0.0,
            -upper_quaternion,
            upper_quaternion,
        )
        root_quaternion = (
            1.0 - fraction
        ) * lower_quaternion + fraction * upper_quaternion
        root_quaternion /= jp.maximum(jp.linalg.norm(root_quaternion), 1.0e-8)
        root_linear_velocity = velocity_scale * (
            (1.0 - fraction) * self._reference_root_linear_velocity[lower]
            + fraction * self._reference_root_linear_velocity[upper]
        )
        root_angular_velocity = velocity_scale * (
            (1.0 - fraction) * self._reference_root_angular_velocity[lower]
            + fraction * self._reference_root_angular_velocity[upper]
        )
        contact_index = jp.where(fraction < 0.5, lower, upper)
        return (
            position,
            velocity,
            self._reference_contact[contact_index],
            root_position,
            root_quaternion,
            root_linear_velocity,
            root_angular_velocity,
        )

    def _foot_contacts(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        # Warp's MJX data surface does not expose contact pairs.  Match the
        # edge-height method used by the current T1 soccer/Booster trainers:
        # project every oriented box half-extent onto world Z and compare the
        # lowest possible foot point with the pitch.  Unlike the former 5 cm
        # site threshold, this does not label a tilted, grounded foot airborne.
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

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        if self._config.use_fixed_command:
            return jp.asarray(self._config.fixed_command)
        (
            rng_x,
            rng_y,
            rng_yaw,
            rng_yaw_magnitude,
            rng_yaw_sign,
            rng_stand,
            rng_axis_gate,
            rng_axis_index,
        ) = jax.random.split(rng, 8)
        sampled_yaw = jax.random.uniform(
            rng_yaw,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )
        yaw_magnitude = jax.random.uniform(
            rng_yaw_magnitude,
            minval=self._config.minimum_abs_yaw,
            maxval=max(
                abs(self._config.ang_vel_yaw[0]),
                abs(self._config.ang_vel_yaw[1]),
            ),
        )
        signed_yaw = jp.where(
            jax.random.bernoulli(
                rng_yaw_sign, self._config.yaw_negative_probability
            ),
            -yaw_magnitude,
            yaw_magnitude,
        )
        sampled_yaw = jp.where(
            self._config.minimum_abs_yaw > 0.0,
            jp.clip(
                signed_yaw,
                self._config.ang_vel_yaw[0],
                self._config.ang_vel_yaw[1],
            ),
            sampled_yaw,
        )
        command = jp.array(
            [
                jax.random.uniform(
                    rng_x,
                    minval=self._config.lin_vel_x[0],
                    maxval=self._config.lin_vel_x[1],
                ),
                jax.random.uniform(
                    rng_y,
                    minval=self._config.lin_vel_y[0],
                    maxval=self._config.lin_vel_y[1],
                ),
                sampled_yaw,
            ]
        )
        axis_index = jax.random.categorical(
            rng_axis_index, self._axis_command_logits
        )
        axis_command = jp.where(
            axis_index == 0,
            jp.array([command[0], 0.0, 0.0]),
            jp.where(
                axis_index == 1,
                jp.array([0.0, command[1], 0.0]),
                jp.array([0.0, 0.0, command[2]]),
            ),
        )
        axis_aligned = (
            jax.random.uniform(rng_axis_gate)
            < self._config.axis_aligned_command_probability
        )
        command = jp.where(axis_aligned, axis_command, command)
        stand = jax.random.uniform(rng_stand) < self._config.stand_probability
        return jp.where(stand, jp.zeros(3), command)

    def _base_diagnostics(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        torso_xmat = data.site_xmat[self._torso_site]
        yaw = jp.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        world_velocity = data.qvel[self._root_dof : self._root_dof + 3]
        local_velocity = jp.array(
            [
                c * world_velocity[0] + s * world_velocity[1],
                -s * world_velocity[0] + c * world_velocity[1],
                world_velocity[2],
            ]
        )
        yaw_rate = data.sensordata[self._gyro_slice][2]
        upright = torso_xmat[2, 2]
        torso_height = data.xpos[self._torso_body, 2]
        return local_velocity, yaw_rate, upright, torso_height

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        joint_positions_training = data.qpos[self._joint_qpos] * self._sign
        joint_velocities_training = data.qvel[self._joint_dof] * self._sign
        if self._reference_centered:
            reference_position, reference_velocity, *_ = self._reference_at_phase(
                info["gait_phase"],
                velocity_scale=self._reference_velocity_scale(info["gait_frequency"]),
            )
        else:
            reference_position = self._nominal_training
            reference_velocity = jp.zeros(self.action_size)
        joint_triplets = jp.stack(
            [
                (joint_positions_training - reference_position)
                / self._config.joint_position_scale,
                (joint_velocities_training - reference_velocity)
                / self._config.joint_velocity_scale,
                info["last_action"] / self._config.previous_action_scale,
            ],
            axis=1,
        ).reshape(-1)
        angular_velocity = (
            data.sensordata[self._gyro_slice] / self._config.angular_velocity_scale
        )
        gravity = data.site_xmat[self._torso_site].T @ jp.array([0.0, 0.0, -1.0])
        actor = jp.concatenate(
            [joint_triplets, angular_velocity, info["command"], gravity]
        )
        if self._phase_observation:
            moving = (info["gait_frequency"] > 1.0e-8).astype(jp.float32)
            phase = 2.0 * jp.pi * info["gait_phase"]
            actor = jp.concatenate(
                [actor, moving * jp.array([jp.cos(phase), jp.sin(phase)])]
            )
        actor = jp.nan_to_num(actor, nan=0.0, posinf=10.0, neginf=-10.0)
        actor = jp.clip(
            actor, -self._config.observation_clip, self._config.observation_clip
        )
        local_velocity, yaw_rate, upright, torso_height = self._base_diagnostics(data)
        privileged = jp.concatenate(
            [actor, local_velocity, jp.array([yaw_rate, upright, torso_height])]
        )
        return {"state": actor, "privileged_state": privileged}
