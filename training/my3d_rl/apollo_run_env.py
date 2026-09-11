"""Apollo-native closed-loop waypoint locomotion task.

The actor boundary, decoder, gains, and command generator mirror the retained
C++ Apollo Walk runner.  Waypoints live only in the environment: the policy
still receives the deployed 78-value observation and can warm-start from the
frozen Apollo ONNX without adding an inference-time dependency.
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

from .apollo_walk_jax import ApolloWalkJax
from .contract import PolicyContract, load_policy_contract
from .rcss_scene import DEFAULT_RESOURCE_ROOT
from .run_env import DirectionalRun, default_config as run_default_config
from .t1_control import APOLLO_DEFAULT_POSE, apollo_joint_gains


DEFAULT_CONTRACT = (
    Path(__file__).parents[1] / "contracts" / "apollo_walk_policy_v1.yaml"
)


def default_config() -> config_dict.ConfigDict:
    config = run_default_config()
    config.use_fixed_command = True
    config.fixed_command = [0.0, 0.0, 0.0]
    config.waypoint_distance_range = [2.0, 6.0]
    config.waypoint_bearing_range = [-float(np.pi), float(np.pi)]
    config.waypoint_arrival_radius = 0.25
    config.reward.waypoint_progress = 4.0
    config.reward.waypoint_success = 10.0
    return config


def apollo_waypoint_command(
    current_xy: jax.Array,
    current_yaw: jax.Array,
    target_xy: jax.Array,
    *,
    stop_radius_m: float = 0.0,
) -> jax.Array:
    """Match C++ WalkRunner's absolute-target command and optional stop gate."""
    delta = target_xy - current_xy
    c = jp.cos(current_yaw)
    s = jp.sin(current_yaw)
    local = jp.array(
        [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
    )
    distance = jp.linalg.norm(delta)
    heading = jp.where(
        jp.linalg.norm(local) > 0.1,
        jp.arctan2(local[1], local[0]),
        0.0,
    )
    command = jp.array(
        [
            jp.clip(local[0], -0.5, 1.0),
            jp.clip(local[1], -0.5, 0.5),
            jp.clip(0.2 * heading, -0.5, 0.5),
        ]
    )
    return jp.where(distance <= stop_radius_m, jp.zeros(3), command)


class ApolloWaypointRun(DirectionalRun):
    """Fine-tune the frozen Apollo actor on its real closed-loop command use."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
    ) -> None:
        contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        if contract.policy_name != "apollo_walk_policy_v1":
            raise ValueError("ApolloWaypointRun requires apollo_walk_policy_v1")
        super().__init__(
            config=default_config() if config is None else config,
            config_overrides=config_overrides,
            contract=contract,
            resource_root=resource_root,
            prefix=prefix,
        )
        self._nominal_training = jp.asarray(APOLLO_DEFAULT_POSE, dtype=jp.float32)
        self._sign = jp.ones(self.action_size, dtype=jp.float32)
        self._nominal_physical = jp.clip(
            self._nominal_training, self._lowers, self._uppers
        )
        distance_range = np.asarray(
            self._config.waypoint_distance_range, dtype=np.float64
        )
        bearing_range = np.asarray(
            self._config.waypoint_bearing_range, dtype=np.float64
        )
        if (
            distance_range.shape != (2,)
            or bearing_range.shape != (2,)
            or not np.isfinite(distance_range).all()
            or not np.isfinite(bearing_range).all()
            or distance_range[0] <= 0.0
            or distance_range[0] > distance_range[1]
            or bearing_range[0] > bearing_range[1]
            or not 0.0 < self._config.waypoint_arrival_radius < distance_range[0]
        ):
            raise ValueError("waypoint curriculum ranges are invalid")

    def _supports_gain_profile(self, profile: str) -> bool:
        return profile == "apollo_runtime_per_joint"

    def _configure_pd_actuators(self) -> None:
        for joint_name, effector in zip(
            self.contract.joint_order, self.contract.effector_order, strict=True
        ):
            position_id = self._mj_model.actuator(
                self.prefix + effector + "_pos"
            ).id
            velocity_id = self._mj_model.actuator(
                self.prefix + effector + "_vel"
            ).id
            kp, kd = apollo_joint_gains(joint_name)
            self._mj_model.actuator_gainprm[position_id, 0] = kp
            self._mj_model.actuator_biasprm[position_id, 1] = -kp
            self._mj_model.actuator_gainprm[velocity_id, 0] = kd
            self._mj_model.actuator_biasprm[velocity_id, 2] = -kd

    def _get_obs(self, data, info):
        joint_position = data.qpos[self._joint_qpos] - self._nominal_training
        joint_velocity = data.qvel[self._joint_dof]
        previous_action = info["last_action"]
        # Apollo's C++ runner deliberately hides the two head joints from the
        # locomotion actor because the head tracker controls them separately.
        joint_position = joint_position.at[:2].set(0.0)
        joint_velocity = joint_velocity.at[:2].set(0.0)
        previous_action = previous_action.at[:2].set(0.0)
        torso_rotation = data.site_xmat[self._torso_site]
        gravity = torso_rotation.T @ jp.array([0.0, 0.0, -1.0])
        actor = jp.concatenate(
            [
                data.sensordata[self._gyro_slice],
                gravity,
                info["command"],
                joint_position,
                joint_velocity,
                previous_action,
            ]
        )
        actor = jp.nan_to_num(actor, nan=0.0, posinf=10.0, neginf=-10.0)
        actor = jp.clip(actor, -10.0, 10.0)
        local_velocity, yaw_rate, upright, torso_height = self._base_diagnostics(data)
        privileged = jp.concatenate(
            [actor, local_velocity[:2], jp.array([yaw_rate, upright, torso_height, data.qvel[self._root_dof + 2]])]
        )
        return {"state": actor, "privileged_state": privileged}

    def _waypoint_command(self, data, waypoint_xy):
        current_yaw = jp.arctan2(
            data.site_xmat[self._torso_site][1, 0],
            data.site_xmat[self._torso_site][0, 0],
        )
        return apollo_waypoint_command(
            data.site_xpos[self._torso_site, :2],
            current_yaw,
            waypoint_xy,
            stop_radius_m=self._config.waypoint_arrival_radius,
        )

    def reset(self, rng: jax.Array):
        state = super().reset(rng)
        return self._initialize_waypoint(state)

    def _initialize_waypoint(self, state):
        """Attach a new waypoint to an already valid Apollo gait state."""
        state.info["rng"], distance_rng, bearing_rng = jax.random.split(
            state.info["rng"], 3
        )
        distance = jax.random.uniform(
            distance_rng,
            minval=self._config.waypoint_distance_range[0],
            maxval=self._config.waypoint_distance_range[1],
        )
        bearing = jax.random.uniform(
            bearing_rng,
            minval=self._config.waypoint_bearing_range[0],
            maxval=self._config.waypoint_bearing_range[1],
        )
        initial_yaw = state.info["initial_yaw"]
        world_heading = initial_yaw + bearing
        waypoint_xy = state.data.site_xpos[self._torso_site, :2] + distance * jp.array(
            [jp.cos(world_heading), jp.sin(world_heading)]
        )
        state.info["waypoint_xy"] = waypoint_xy
        state.info["waypoint_bearing"] = bearing
        state.info["waypoint_distance"] = distance
        state.info["command"] = self._waypoint_command(state.data, waypoint_xy)
        state.info["initial_torso_xy"] = state.data.site_xpos[
            self._torso_site, :2
        ]
        state.info["initial_yaw"] = jp.arctan2(
            state.data.site_xmat[self._torso_site][1, 0],
            state.data.site_xmat[self._torso_site][0, 0],
        )
        state.info["step"] = jp.array(0, dtype=jp.int32)
        state.metrics["diagnostic/waypoint_distance_m"] = distance
        state.metrics["diagnostic/waypoint_progress_m"] = jp.array(0.0)
        state.metrics["diagnostic/waypoint_success"] = jp.array(0.0)
        return state.replace(obs=self._get_obs(state.data, state.info))

    def step(self, state, action):
        waypoint_xy = state.info["waypoint_xy"]
        previous_distance = jp.linalg.norm(
            waypoint_xy - state.data.site_xpos[self._torso_site, :2]
        )
        state.info["command"] = self._waypoint_command(state.data, waypoint_xy)
        state = super().step(state, action)
        distance = jp.linalg.norm(
            waypoint_xy - state.data.site_xpos[self._torso_site, :2]
        )
        progress = previous_distance - distance
        success = distance <= self._config.waypoint_arrival_radius
        state.info["waypoint_distance"] = distance
        state.info["command"] = self._waypoint_command(state.data, waypoint_xy)
        state.metrics["diagnostic/waypoint_distance_m"] = distance
        state.metrics["diagnostic/waypoint_progress_m"] = progress
        state.metrics["diagnostic/waypoint_success"] = success.astype(jp.float32)
        reward = (
            state.reward
            + self._config.reward.waypoint_progress * progress
            + self._config.reward.waypoint_success * success.astype(jp.float32)
        )
        done = state.done.astype(bool) | success
        return state.replace(
            obs=self._get_obs(state.data, state.info),
            reward=reward,
            done=done.astype(jp.float32),
        )


class ApolloHandoffWaypointRun(ApolloWaypointRun):
    """Start waypoint control from states produced by the frozen Apollo Walk.

    This is deliberately a training/evaluation environment only.  It restores
    states collected from the frozen actor under sampled runtime commands, then
    gives the candidate the same 78-value observation and previous-action
    history at the switch.  Near the destination, a small teacher-action cost
    discourages a specialist from creating a large discontinuity when control
    returns to Apollo Walk.
    """

    def __init__(
        self,
        entry_policy: ApolloWalkJax,
        entry_corpus: Path,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "train_",
    ) -> None:
        config = default_config() if config is None else config
        config.handoff_radius = 0.60
        config.reward.handoff_action = -0.10
        super().__init__(
            config=config,
            config_overrides=config_overrides,
            contract=contract,
            resource_root=resource_root,
            prefix=prefix,
        )
        self._entry_policy = entry_policy
        if not 0.0 < self._config.handoff_radius:
            raise ValueError("handoff_radius must be positive")
        if not entry_corpus.is_file():
            raise FileNotFoundError(f"Apollo handoff corpus not found: {entry_corpus}")
        with np.load(entry_corpus, allow_pickle=False) as archive:
            required = {
                "qpos",
                "qvel",
                "last_action",
                "last_last_action",
                "source_command",
                "source_step",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"Apollo handoff corpus is missing arrays: {sorted(missing)}"
                )
            qpos = np.asarray(archive["qpos"], dtype=np.float32)
            qvel = np.asarray(archive["qvel"], dtype=np.float32)
            last_action = np.asarray(archive["last_action"], dtype=np.float32)
            last_last_action = np.asarray(
                archive["last_last_action"], dtype=np.float32
            )
            source_command = np.asarray(
                archive["source_command"], dtype=np.float32
            )
            source_step = np.asarray(archive["source_step"], dtype=np.int32)
        sample_count = qpos.shape[0]
        if (
            sample_count < 1
            or qpos.shape != (sample_count, self._mj_model.nq)
            or qvel.shape != (sample_count, self._mj_model.nv)
            or last_action.shape != (sample_count, self.action_size)
            or last_last_action.shape != (sample_count, self.action_size)
            or source_command.shape != (sample_count, 3)
            or source_step.shape != (sample_count,)
            or not all(
                np.isfinite(array).all()
                for array in (
                    qpos,
                    qvel,
                    last_action,
                    last_last_action,
                    source_command,
                )
            )
        ):
            raise ValueError("Apollo handoff corpus arrays are incompatible")
        self._entry_qpos = jp.asarray(qpos)
        self._entry_qvel = jp.asarray(qvel)
        self._entry_last_action = jp.asarray(last_action)
        self._entry_last_last_action = jp.asarray(last_last_action)
        self._entry_source_command = jp.asarray(source_command)
        self._entry_source_step = jp.asarray(source_step)
        self._entry_sample_count = sample_count

    def reset(self, rng: jax.Array):
        state = self.reset_source_replay(rng)
        state = self._initialize_waypoint(state)
        return state

    def reset_source_replay(self, rng: jax.Array):
        """Restore one corpus state under its original Walk command."""
        rng, entry_index_rng = jax.random.split(rng)
        entry_index = jax.random.randint(
            entry_index_rng,
            (),
            minval=0,
            maxval=self._entry_sample_count,
        )
        last_action = self._entry_last_action[entry_index]
        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._pos_actuator].set(
            self.decode_action_targets(last_action)
        )
        data = mjx_env.make_data(
            self._mj_model,
            qpos=self._entry_qpos[entry_index],
            qvel=self._entry_qvel[entry_index],
            ctrl=ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)
        initial_yaw = jp.arctan2(
            data.site_xmat[self._torso_site][1, 0],
            data.site_xmat[self._torso_site][0, 0],
        )
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "command": self._entry_source_command[entry_index],
            "gait_phase": jp.array(0.0),
            "gait_frequency": jp.array(1.5),
            "last_action": last_action,
            "last_last_action": self._entry_last_last_action[entry_index],
            "delay_steps": jp.array(0, dtype=jp.int32),
            "reference_init": jp.array(False),
            "last_foot_positions": jp.stack(
                [
                    data.site_xpos[self._left_foot_site],
                    data.site_xpos[self._right_foot_site],
                ]
            ),
            "initial_torso_xy": data.site_xpos[self._torso_site, :2],
            "initial_yaw": initial_yaw,
            "entry_sample_index": entry_index,
            "entry_source_command": self._entry_source_command[entry_index],
            "entry_steps": self._entry_source_step[entry_index],
        }
        metrics = self._initial_metrics()
        metrics["cost/handoff_action_mismatch"] = jp.array(0.0)
        return mjx_env.State(
            data=data,
            obs=self._get_obs(data, info),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state, action):
        teacher_action = self._entry_policy(state.obs["state"])
        mismatch = jp.mean(jp.square(action - teacher_action))
        distance = jp.linalg.norm(
            state.info["waypoint_xy"]
            - state.data.site_xpos[self._torso_site, :2]
        )
        handoff_gate = (distance <= self._config.handoff_radius).astype(jp.float32)
        state = super().step(state, action)
        state.metrics["cost/handoff_action_mismatch"] = mismatch * handoff_gate
        reward = (
            state.reward
            + self._config.reward.handoff_action
            * mismatch
            * handoff_gate
            * self.dt
        )
        return state.replace(reward=reward)
