"""Exact CPU MuJoCo reference for the deterministic striker controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from .contract import PolicyContract
from .kick_teacher import KickTeacherEvaluator, KickTeacherSpec


def closed_loop_approach_control_numpy(
    ball_local_xy: np.ndarray,
    target_local: np.ndarray,
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
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """NumPy reference for the deployable geometry controller."""
    target_local = np.asarray(target_local, dtype=np.float64)
    target_local /= max(float(np.linalg.norm(target_local)), 1.0e-6)
    target_left = np.array([-target_local[1], target_local[0]])
    contact_error = (
        np.asarray(ball_local_xy, dtype=np.float64)
        - target_local * standoff
        - target_left * ball_lateral
    )
    distance = float(np.linalg.norm(contact_error))
    heading_error = float(np.arctan2(target_local[1], target_local[0]))
    distance_activation = np.clip(
        (activation_radius - distance) / max(activation_radius - full_radius, 1.0e-6),
        0.0,
        1.0,
    )
    heading_activation = np.clip(
        (heading_radius - abs(heading_error))
        / max(heading_radius - full_heading, 1.0e-6),
        0.0,
        1.0,
    )
    command = np.array(
        [
            np.clip(
                position_gain * contact_error[0],
                -max_backward_speed,
                max_forward_speed,
            ),
            np.clip(
                lateral_gain * contact_error[1],
                -max_lateral_speed,
                max_lateral_speed,
            ),
            np.clip(
                yaw_gain * heading_error,
                -max_yaw_speed,
                max_yaw_speed,
            ),
        ],
        dtype=np.float64,
    )
    return (
        command,
        float(distance_activation * heading_activation),
        contact_error,
        heading_error,
    )


@dataclass(frozen=True)
class StrikerCpuResult:
    seed: int
    initial_robot_distance: float
    initial_robot_lateral: float
    initial_robot_yaw_error: float
    initial_target_angle: float
    initial_target_distance: float
    triggered: bool
    contacted: bool
    succeeded: bool
    fallen: bool
    trigger_count: int
    first_trigger_step: int
    first_contact_step: int
    episode_steps: int
    maximum_directional_speed_mps: float
    final_goal_distance_m: float
    final_contact_distance_m: float
    final_heading_error_rad: float


class StrikerCpuEvaluator:
    """Independent-controller rollout in exact double-precision CPU MuJoCo."""

    def __init__(
        self,
        contract: PolicyContract,
        kick_prior_joint_residuals: np.ndarray,
        config: Mapping[str, Any],
        *,
        prefix: str = "striker_cpu_",
    ) -> None:
        if contract.policy_name != "striker_policy_v1":
            raise ValueError("CPU striker requires striker_policy_v1")
        prior = np.asarray(kick_prior_joint_residuals, dtype=np.float64)
        if (
            prior.ndim != 2
            or prior.shape[0] < 2
            or prior.shape[1] != contract.action_size
            or not np.isfinite(prior).all()
        ):
            raise ValueError("kick prior must be a finite [T, 23] trajectory")
        self.contract = contract
        self.config = dict(config)
        self.prior = prior
        self._teacher = KickTeacherEvaluator(
            KickTeacherSpec(
                target_distance_m=2.0,
                target_angle_deg=0.0,
                requested_ball_speed_mps=1.43,
                desired_arrival_speed_mps=0.8,
                action_mode="pass",
                evaluation_duration_s=max(
                    3.0,
                    float(self.config["episode_length"])
                    * float(self.config["ctrl_dt"]),
                ),
                control_dt_s=float(self.config["ctrl_dt"]),
                simulation_dt_s=float(self.config["sim_dt"]),
            ),
            contract=contract,
            prefix=prefix,
        )
        self._n_substeps = round(
            float(self.config["ctrl_dt"]) / float(self.config["sim_dt"])
        )

    def _features(
        self, data: mujoco.MjData, goal_world: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        teacher = self._teacher
        torso_xmat = data.site_xmat[teacher._torso_site].reshape(3, 3)
        yaw = float(np.arctan2(torso_xmat[1, 0], torso_xmat[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_yaw = np.array([[c, s], [-s, c]])
        torso_pos = data.xpos[teacher._torso_body]
        ball_pos = data.xpos[teacher._ball_body]
        goal_delta = goal_world - ball_pos[:2]
        goal_distance = float(np.linalg.norm(goal_delta))
        target_world = goal_delta / max(goal_distance, 1.0e-6)
        target_local = world_to_yaw @ target_world
        ball_local_xy = world_to_yaw @ (ball_pos[:2] - torso_pos[:2])
        command, activation, contact_error, heading_error = (
            closed_loop_approach_control_numpy(
                ball_local_xy,
                target_local,
                standoff=float(self.config["approach_standoff"]),
                ball_lateral=float(self.config["approach_ball_lateral"]),
                position_gain=float(self.config["approach_position_gain"]),
                lateral_gain=float(self.config["approach_lateral_gain"]),
                yaw_gain=float(self.config["approach_yaw_gain"]),
                max_forward_speed=float(self.config["approach_max_forward_speed"]),
                max_backward_speed=float(
                    self.config["approach_max_backward_speed"]
                ),
                max_lateral_speed=float(self.config["approach_max_lateral_speed"]),
                max_yaw_speed=float(self.config["approach_max_yaw_speed"]),
                activation_radius=float(self.config["kick_activation_radius"]),
                full_radius=float(self.config["kick_full_radius"]),
                heading_radius=float(self.config["kick_heading_radius"]),
                full_heading=float(self.config["kick_full_heading"]),
            )
        )
        return {
            "torso_xmat": torso_xmat,
            "torso_pos": torso_pos,
            "ball_pos": ball_pos,
            "target_world": target_world,
            "goal_distance": goal_distance,
            "command": command,
            "activation": activation,
            "contact_distance": float(np.linalg.norm(contact_error)),
            "heading_error": heading_error,
        }

    def rollout(self, seed: int) -> StrikerCpuResult:
        cfg = self.config
        teacher = self._teacher
        rng = np.random.default_rng(seed)
        target_angle = float(rng.uniform(*cfg["target_angle_range"]))
        target_world = np.array([np.cos(target_angle), np.sin(target_angle)])
        target_left = np.array([-target_world[1], target_world[0]])
        robot_distance = float(rng.uniform(*cfg["robot_distance_range"]))
        robot_lateral = float(rng.uniform(*cfg["robot_lateral_range"]))
        yaw_error = float(rng.uniform(*cfg["robot_yaw_noise_range"]))
        robot_yaw = target_angle + yaw_error
        joint_noise = rng.uniform(
            -float(cfg["reset_joint_noise"]),
            float(cfg["reset_joint_noise"]),
            size=self.contract.action_size,
        )
        root_velocity = rng.uniform(
            -float(cfg["reset_root_velocity_noise"]),
            float(cfg["reset_root_velocity_noise"]),
            size=6,
        )
        target_distance = float(rng.uniform(*cfg["target_distance_range"]))

        data = mujoco.MjData(teacher.model)
        ball_qpos = teacher._ball_qpos
        data.qpos[ball_qpos : ball_qpos + 3] = np.array([0.0, 0.0, 0.11])
        robot_xy = -target_world * robot_distance + target_left * robot_lateral
        data.qpos[teacher._root_qpos : teacher._root_qpos + 2] = robot_xy
        data.qpos[teacher._root_qpos + 3 : teacher._root_qpos + 7] = np.array(
            [np.cos(0.5 * robot_yaw), 0.0, 0.0, np.sin(0.5 * robot_yaw)]
        )
        data.qpos[teacher._joint_qpos] += joint_noise
        data.qvel[teacher._root_dof : teacher._root_dof + 6] = root_velocity
        data.ctrl[teacher._pos_actuator] = teacher._default_pose
        mujoco.mj_forward(teacher.model, data)

        initial_ball_xy = data.xpos[teacher._ball_body, :2].copy()
        goal_world = initial_ball_xy + target_world * target_distance
        walk_last_action = np.zeros(self.contract.action_size)
        kick_step = -1
        kick_cooldown = 0
        settled_steps = 0
        trigger_count = 0
        first_trigger_step = int(cfg["episode_length"])
        first_contact_step = int(cfg["episode_length"])
        contacted = False
        succeeded = False
        fallen = False
        maximum_directional_speed = 0.0
        final_features = self._features(data, goal_world)
        episode_steps = int(cfg["episode_length"])

        for control_step in range(int(cfg["episode_length"])):
            features = self._features(data, goal_world)
            inside_settled = (
                float(features["contact_distance"])
                <= float(cfg["kick_settled_distance"])
                and abs(float(features["heading_error"]))
                <= float(cfg["kick_settled_heading"])
            )
            settled_steps = min(
                settled_steps + 1 if inside_settled else 0,
                int(cfg["kick_settled_confirmation_steps"]),
            )
            trigger = (
                kick_step < 0
                and kick_cooldown <= 0
                and (
                    float(features["activation"])
                    >= float(cfg["kick_trigger_threshold"])
                    or settled_steps >= int(cfg["kick_settled_confirmation_steps"])
                )
            )
            if trigger:
                kick_step = 0
                trigger_count += 1
                if first_trigger_step == int(cfg["episode_length"]):
                    first_trigger_step = control_step
            kick_active = kick_step >= 0
            prior = self.prior[min(max(kick_step, 0), self.prior.shape[0] - 1)]
            if not kick_active:
                prior = np.zeros(self.contract.action_size)
            if kick_active:
                command = (
                    np.asarray(cfg["kick_walk_command"], dtype=np.float64)
                    if kick_step < int(cfg["kick_walk_duration_steps"])
                    else np.zeros(3)
                )
            else:
                command = np.asarray(features["command"], dtype=np.float64)
            walk_target, walk_action = teacher._stable_walk_target(
                data, walk_last_action, command
            )
            targets = np.clip(
                walk_target + prior,
                teacher._lowers,
                teacher._uppers,
            )
            data.ctrl[teacher._pos_actuator] = targets
            for _ in range(self._n_substeps):
                mujoco.mj_step(teacher.model, data)
            walk_last_action = walk_action
            final_features = self._features(data, goal_world)

            ball_velocity = data.qvel[teacher._ball_dof : teacher._ball_dof + 3]
            ball_displacement = (
                np.asarray(final_features["ball_pos"])[:2] - initial_ball_xy
            )
            ball_speed = float(np.linalg.norm(ball_velocity[:2]))
            directional_speed = max(
                0.0,
                float(np.dot(ball_velocity[:2], final_features["target_world"])),
            )
            maximum_directional_speed = max(
                maximum_directional_speed, min(directional_speed, 6.0)
            )
            contact_now = ball_speed >= 0.15 or np.linalg.norm(ball_displacement) >= 0.08
            if contact_now and not contacted:
                first_contact_step = control_step
            contacted = contacted or bool(contact_now)
            torso_xmat = np.asarray(final_features["torso_xmat"])
            torso_height = float(np.asarray(final_features["torso_pos"])[2])
            fallen = bool(torso_height < 0.35 or torso_xmat[2, 2] < 0.0)
            invalid = bool(
                not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            )
            succeeded = bool(
                contacted
                and float(final_features["goal_distance"])
                <= float(cfg["success_radius"])
            )

            if kick_active:
                kick_step += 1
                if kick_step >= self.prior.shape[0]:
                    kick_step = -1
                    kick_cooldown = int(cfg["kick_rearm_steps"])
            else:
                kick_cooldown = max(kick_cooldown - 1, 0)
            settled_steps = 0 if trigger or kick_active else settled_steps
            if succeeded or fallen or invalid:
                episode_steps = control_step + 1
                break

        return StrikerCpuResult(
            seed=seed,
            initial_robot_distance=robot_distance,
            initial_robot_lateral=robot_lateral,
            initial_robot_yaw_error=yaw_error,
            initial_target_angle=target_angle,
            initial_target_distance=target_distance,
            triggered=trigger_count > 0,
            contacted=contacted,
            succeeded=succeeded,
            fallen=fallen,
            trigger_count=trigger_count,
            first_trigger_step=first_trigger_step,
            first_contact_step=first_contact_step,
            episode_steps=episode_steps,
            maximum_directional_speed_mps=maximum_directional_speed,
            final_goal_distance_m=float(final_features["goal_distance"]),
            final_contact_distance_m=float(final_features["contact_distance"]),
            final_heading_error_rad=float(final_features["heading_error"]),
        )
