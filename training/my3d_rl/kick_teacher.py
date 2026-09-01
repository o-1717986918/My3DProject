"""Deterministic low-dimensional kick teacher and CEM optimizer.

The teacher is intentionally independent from PPO.  It produces a bounded,
replayable joint-position trajectory against the exact RCSSServerMJ assets so
later residual learning starts from a physical contact rather than from random
motor exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import onnxruntime as ort

from .contract import PolicyContract, load_policy_contract
from .kick_env import DEFAULT_CONTRACT
from .kick_transition import estimate_locomotion_phase
from .rcss_scene import DEFAULT_RESOURCE_ROOT, build_single_t1_soccer_model
from .t1_control import APOLLO_DEFAULT_POSE, KICK_ACTION_SCALE, apollo_joint_gains


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_WALK_POLICY = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)
PARAMETER_NAMES = (
    "support_hip_roll",
    "support_ankle_roll",
    "backswing_hip_pitch",
    "backswing_knee_pitch",
    "backswing_ankle_pitch",
    "strike_hip_pitch",
    "strike_knee_pitch",
    "strike_ankle_pitch",
    "strike_hip_yaw",
    "waist_yaw",
    "left_arm_pitch",
    "right_arm_pitch",
    "strike_hip_roll",
    "strike_ankle_roll",
)

PARAMETER_LOWER = np.array(
    [
        -0.25,
        -0.20,
        -0.80,
        -1.00,
        -0.60,
        -1.00,
        -1.20,
        -0.80,
        -0.40,
        -0.30,
        -0.50,
        -0.50,
        -0.45,
        -0.35,
    ],
    dtype=np.float64,
)
PARAMETER_UPPER = np.array(
    [
        0.25,
        0.20,
        0.80,
        1.00,
        0.60,
        1.00,
        1.20,
        0.80,
        0.40,
        0.30,
        0.50,
        0.50,
        0.45,
        0.35,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CEMResult:
    parameters: np.ndarray
    score: float
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class KickTransitionEntry:
    """Exact physical state accepted by the deterministic setup controller."""

    qpos: np.ndarray
    qvel: np.ndarray
    joint_position_offset: np.ndarray
    joint_velocity: np.ndarray
    walk_previous_action: np.ndarray
    setup_velocity_command: np.ndarray
    locomotion_phase: np.ndarray
    support_hint: np.ndarray
    phase_magnitude_rad: float
    ball_position_local_m: np.ndarray
    root_velocity: np.ndarray
    torso_height_m: float
    upright: float

    def copy(self) -> "KickTransitionEntry":
        """Return a deep copy so callers cannot mutate evaluator state."""
        return KickTransitionEntry(
            qpos=self.qpos.copy(),
            qvel=self.qvel.copy(),
            joint_position_offset=self.joint_position_offset.copy(),
            joint_velocity=self.joint_velocity.copy(),
            walk_previous_action=self.walk_previous_action.copy(),
            setup_velocity_command=self.setup_velocity_command.copy(),
            locomotion_phase=self.locomotion_phase.copy(),
            support_hint=self.support_hint.copy(),
            phase_magnitude_rad=self.phase_magnitude_rad,
            ball_position_local_m=self.ball_position_local_m.copy(),
            root_velocity=self.root_velocity.copy(),
            torso_height_m=self.torso_height_m,
            upright=self.upright,
        )


@dataclass(frozen=True)
class KickTeacherSpec:
    target_distance_m: float = 2.0
    target_angle_deg: float = 0.0
    requested_ball_speed_mps: float = 1.43
    desired_arrival_speed_mps: float = 1.0
    action_mode: str = "pass"
    duration_s: float = 1.20
    evaluation_duration_s: float = 3.0
    control_dt_s: float = 0.02
    simulation_dt_s: float = 0.005

    def validate(self) -> None:
        values = np.asarray(
            [
                self.target_distance_m,
                self.target_angle_deg,
                self.requested_ball_speed_mps,
                self.desired_arrival_speed_mps,
                self.duration_s,
                self.evaluation_duration_s,
                self.control_dt_s,
                self.simulation_dt_s,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("kick teacher spec must be finite")
        if not 0.25 <= self.target_distance_m <= 10.0:
            raise ValueError("target_distance_m must be in [0.25, 10]")
        if not -30.0 <= self.target_angle_deg <= 30.0:
            raise ValueError("target_angle_deg must be in [-30, 30]")
        if not 0.2 <= self.requested_ball_speed_mps <= 6.0:
            raise ValueError("requested_ball_speed_mps must be in [0.2, 6]")
        if not 0.0 <= self.desired_arrival_speed_mps <= 6.0:
            raise ValueError("desired_arrival_speed_mps must be in [0, 6]")
        if self.action_mode not in {"pass", "shot", "clear"}:
            raise ValueError("action_mode must be pass, shot, or clear")
        if self.duration_s <= 0.0 or self.control_dt_s <= 0.0:
            raise ValueError("durations must be positive")
        if self.evaluation_duration_s < self.duration_s:
            raise ValueError("evaluation_duration_s must cover the action duration")
        ratio = self.control_dt_s / self.simulation_dt_s
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("control_dt_s must be divisible by simulation_dt_s")


def cem_optimize(
    objective: Callable[[np.ndarray], float],
    *,
    initial_mean: np.ndarray,
    initial_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seed: int,
    population: int,
    generations: int,
    elite_fraction: float = 0.2,
    smoothing: float = 0.25,
) -> CEMResult:
    """Maximize a deterministic objective with bounded CEM sampling."""
    mean = np.asarray(initial_mean, dtype=np.float64).copy()
    std = np.asarray(initial_std, dtype=np.float64).copy()
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (mean.shape == std.shape == lower.shape == upper.shape):
        raise ValueError("CEM arrays must have identical shapes")
    if population < 2 or generations < 1:
        raise ValueError("population >= 2 and generations >= 1 are required")
    if not 0.0 < elite_fraction <= 0.5:
        raise ValueError("elite_fraction must be in (0, 0.5]")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    if np.any(lower >= upper) or np.any(std <= 0.0):
        raise ValueError("invalid CEM bounds or standard deviation")

    rng = np.random.default_rng(seed)
    elite_count = max(1, int(np.ceil(population * elite_fraction)))
    best_parameters = np.clip(mean, lower, upper)
    best_score = float(objective(best_parameters))
    history: list[dict[str, float]] = []

    for generation in range(generations):
        samples = rng.normal(mean, std, size=(population, mean.size))
        samples = np.clip(samples, lower, upper)
        scores = np.asarray([objective(sample) for sample in samples])
        finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
        order = np.argsort(finite_scores)[::-1]
        elites = samples[order[:elite_count]]
        elite_scores = finite_scores[order[:elite_count]]
        if elite_scores[0] > best_score:
            best_score = float(elite_scores[0])
            best_parameters = elites[0].copy()

        elite_mean = np.mean(elites, axis=0)
        elite_std = np.std(elites, axis=0)
        mean = smoothing * mean + (1.0 - smoothing) * elite_mean
        std = smoothing * std + (1.0 - smoothing) * np.maximum(elite_std, 1.0e-3)
        history.append(
            {
                "generation": float(generation),
                "best_score": best_score,
                "generation_best_score": float(elite_scores[0]),
                "elite_mean_score": float(np.mean(elite_scores)),
                "mean_std": float(np.mean(std)),
            }
        )

    return CEMResult(best_parameters, best_score, tuple(history))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _blend_keyframes(
    times: np.ndarray, keyframes: np.ndarray, time_s: float
) -> np.ndarray:
    if time_s <= times[0]:
        return keyframes[0].copy()
    if time_s >= times[-1]:
        return keyframes[-1].copy()
    right = int(np.searchsorted(times, time_s, side="right"))
    left = right - 1
    fraction = _smoothstep((time_s - times[left]) / (times[right] - times[left]))
    return keyframes[left] * (1.0 - fraction) + keyframes[right] * fraction


def build_joint_delta_trajectory(
    parameters: np.ndarray,
    contract: PolicyContract,
    times_s: np.ndarray,
) -> np.ndarray:
    """Decode fourteen bounded values into smooth 23-joint kick offsets."""
    parameters = np.asarray(parameters, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)
    if parameters.shape != (len(PARAMETER_NAMES),):
        raise ValueError(f"expected {len(PARAMETER_NAMES)} teacher parameters")
    if np.any(parameters < PARAMETER_LOWER) or np.any(parameters > PARAMETER_UPPER):
        raise ValueError("teacher parameters exceed declared bounds")
    if times_s.ndim != 1 or not np.isfinite(times_s).all():
        raise ValueError("times_s must be a finite one-dimensional array")

    joint = {name: index for index, name in enumerate(contract.joint_order)}
    key_times = np.array([0.0, 0.18, 0.34, 0.54, 0.76, 1.20])
    keys = np.zeros((key_times.size, contract.action_size), dtype=np.float64)

    support_roll, support_ankle = parameters[0:2]
    for key_index in (1, 2, 3):
        keys[key_index, joint["Left_Hip_Roll"]] = support_roll
        keys[key_index, joint["Right_Hip_Roll"]] = support_roll
        keys[key_index, joint["Left_Ankle_Roll"]] = support_ankle
        keys[key_index, joint["Right_Ankle_Roll"]] = support_ankle

    keys[2, joint["Right_Hip_Pitch"]] = parameters[2]
    keys[2, joint["Right_Knee_Pitch"]] = parameters[3]
    keys[2, joint["Right_Ankle_Pitch"]] = parameters[4]
    keys[3, joint["Right_Hip_Pitch"]] = parameters[5]
    keys[3, joint["Right_Knee_Pitch"]] = parameters[6]
    keys[3, joint["Right_Ankle_Pitch"]] = parameters[7]
    keys[3, joint["Right_Hip_Yaw"]] = parameters[8]
    keys[3, joint["Right_Hip_Roll"]] += parameters[12]
    keys[3, joint["Right_Ankle_Roll"]] += parameters[13]
    keys[2:4, joint["Waist"]] = parameters[9]
    keys[2:4, joint["Left_Shoulder_Pitch"]] = parameters[10]
    keys[2:4, joint["Right_Shoulder_Pitch"]] = parameters[11]
    keys[4] = 0.35 * keys[3]

    return np.stack(
        [_blend_keyframes(key_times, keys, float(time_s)) for time_s in times_s]
    )


class KickTeacherEvaluator:
    """Exact-MuJoCo evaluator for a fixed-ball keyframe teacher."""

    def __init__(
        self,
        spec: KickTeacherSpec,
        *,
        contract: PolicyContract | None = None,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "teacher_",
        walk_policy_path: Path = DEFAULT_WALK_POLICY,
    ) -> None:
        spec.validate()
        self.spec = spec
        self.contract = contract or load_policy_contract(DEFAULT_CONTRACT)
        self.prefix = prefix
        self.model = build_single_t1_soccer_model(
            resource_root, prefix=prefix, robot_x=-0.32, robot_y=0.0
        )
        self.model.opt.timestep = spec.simulation_dt_s
        self._joint_qpos = np.array(
            [
                self.model.joint(prefix + name).qposadr[0]
                for name in self.contract.joint_order
            ]
        )
        self._joint_dof = np.array(
            [
                self.model.joint(prefix + name).dofadr[0]
                for name in self.contract.joint_order
            ]
        )
        self._pos_actuator = np.array(
            [
                self.model.actuator(prefix + name + "_pos").id
                for name in self.contract.effector_order
            ]
        )
        for joint_name, effector in zip(
            self.contract.joint_order, self.contract.effector_order, strict=True
        ):
            pos_id = self.model.actuator(prefix + effector + "_pos").id
            vel_id = self.model.actuator(prefix + effector + "_vel").id
            kp, kd = apollo_joint_gains(joint_name)
            self.model.actuator_gainprm[pos_id, 0] = kp
            self.model.actuator_biasprm[pos_id, 1] = -kp
            self.model.actuator_gainprm[vel_id, 0] = kd
            self.model.actuator_biasprm[vel_id, 2] = -kd
        self._default_pose = APOLLO_DEFAULT_POSE.copy()
        self._lowers = np.asarray(
            [
                self.model.joint(prefix + name).range[0]
                for name in self.contract.joint_order
            ]
        )
        self._uppers = np.asarray(
            [
                self.model.joint(prefix + name).range[1]
                for name in self.contract.joint_order
            ]
        )
        self._ball_body = self.model.body("ball").id
        self._ball_qpos = self.model.joint("ball-root").qposadr[0]
        self._ball_dof = self.model.joint("ball-root").dofadr[0]
        root_joint = self.model.joint(prefix + "root")
        self._root_qpos = root_joint.qposadr[0]
        self._root_dof = root_joint.dofadr[0]
        self._torso_body = self.model.body(prefix + "torso").id
        self._torso_site = self.model.site(prefix + "torso").id
        gyro = self.model.sensor(prefix + "torso_gyro")
        self._gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
        if not walk_policy_path.is_file():
            raise FileNotFoundError(f"Apollo walk policy not found: {walk_policy_path}")
        self.walk_policy_path = walk_policy_path
        self._walk_session = ort.InferenceSession(
            str(walk_policy_path), providers=["CPUExecutionProvider"]
        )
        self._walk_input = self._walk_session.get_inputs()[0].name
        if self._walk_session.get_inputs()[0].shape != [1, 78]:
            raise ValueError("Apollo walk teacher must have a [1, 78] input")
        if self._walk_session.get_outputs()[0].shape != [1, 23]:
            raise ValueError("Apollo walk teacher must have a [1, 23] output")
        self._captured_targets = np.empty((0, self.contract.action_size))
        self._captured_observations = np.empty((0, self.contract.observation_size))
        self._captured_actions = np.empty((0, self.contract.action_size))
        self._captured_transition_entry: KickTransitionEntry | None = None

    @property
    def captured_transition_entry(self) -> KickTransitionEntry | None:
        """Last requested transition entry, copied at the ownership boundary."""
        if self._captured_transition_entry is None:
            return None
        return self._captured_transition_entry.copy()

    @property
    def captured_observations(self) -> np.ndarray:
        """Return a defensive copy of observations from the last captured rollout."""
        return self._captured_observations.copy()

    @property
    def captured_actions(self) -> np.ndarray:
        """Return a defensive copy of actions from the last captured rollout."""
        return self._captured_actions.copy()

    @property
    def captured_targets(self) -> np.ndarray:
        """Return a defensive copy of targets from the last captured rollout."""
        return self._captured_targets.copy()

    def _stable_walk_target(
        self,
        data: mujoco.MjData,
        previous_action: np.ndarray,
        velocity_command: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torso_xmat = data.site_xmat[self._torso_site].reshape(3, 3)
        gravity = torso_xmat.T @ np.array([0.0, 0.0, -1.0])
        observation = np.concatenate(
            [
                data.sensordata[self._gyro_slice],
                gravity,
                velocity_command,
                data.qpos[self._joint_qpos] - self._default_pose,
                data.qvel[self._joint_dof],
                previous_action,
            ]
        ).astype(np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        observation = np.clip(observation, -10.0, 10.0)
        action = self._walk_session.run(None, {self._walk_input: observation[None, :]})[
            0
        ][0].astype(np.float64)
        action = np.clip(np.nan_to_num(action), -5.0, 5.0)
        return self._default_pose + 0.25 * action, action

    def _make_transition_entry(
        self,
        data: mujoco.MjData,
        walk_previous_action: np.ndarray,
        setup_velocity_command: np.ndarray,
    ) -> KickTransitionEntry:
        torso_xmat = data.site_xmat[self._torso_site].reshape(3, 3)
        yaw = np.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_yaw = np.array([[c, s], [-s, c]])
        torso_pos = data.xpos[self._torso_body]
        ball_pos = data.xpos[self._ball_body]
        ball_local_xy = world_to_yaw @ (ball_pos[:2] - torso_pos[:2])
        joint_position_offset = (
            data.qpos[self._joint_qpos] - self._default_pose
        ).copy()
        joint_velocity = data.qvel[self._joint_dof].copy()
        locomotion = estimate_locomotion_phase(
            joint_position_offset,
            joint_velocity,
            self.contract.joint_order,
        )
        return KickTransitionEntry(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            joint_position_offset=joint_position_offset,
            joint_velocity=joint_velocity,
            walk_previous_action=np.asarray(
                walk_previous_action, dtype=np.float64
            ).copy(),
            setup_velocity_command=np.asarray(
                setup_velocity_command, dtype=np.float64
            ).copy(),
            locomotion_phase=locomotion.sin_cos.copy(),
            support_hint=locomotion.support_hint.copy(),
            phase_magnitude_rad=locomotion.magnitude_rad,
            ball_position_local_m=np.array(
                [
                    ball_local_xy[0],
                    ball_local_xy[1],
                    ball_pos[2] - torso_pos[2],
                ],
                dtype=np.float64,
            ),
            root_velocity=data.qvel[
                self._root_dof : self._root_dof + 6
            ].copy(),
            torso_height_m=float(torso_pos[2]),
            upright=float(torso_xmat[2, 2]),
        )

    def _kick_actor_observation(
        self,
        data: mujoco.MjData,
        previous_action: np.ndarray,
        action_time_s: float,
    ) -> np.ndarray:
        torso_xmat = data.site_xmat[self._torso_site].reshape(3, 3)
        yaw = np.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_yaw = np.array([[c, s], [-s, c]])
        torso_pos = data.xpos[self._torso_body]
        ball_pos = data.xpos[self._ball_body]
        ball_world_vel = data.qvel[self._ball_dof : self._ball_dof + 3]
        torso_world_vel = data.qvel[self._root_dof : self._root_dof + 3]
        ball_local_xy = world_to_yaw @ (ball_pos[:2] - torso_pos[:2])
        ball_local_vel_xy = world_to_yaw @ (ball_world_vel[:2] - torso_world_vel[:2])
        target_angle = np.deg2rad(self.spec.target_angle_deg)
        target_world = np.array([np.cos(target_angle), np.sin(target_angle)])
        target_local = world_to_yaw @ target_world
        gravity = torso_xmat.T @ np.array([0.0, 0.0, -1.0])
        # A kick is a one-shot motion, not a periodic gait.  Encoding the
        # complete action on one half-circle makes progression injective and
        # clamps to a distinct terminal value after the kick has finished.
        progress = np.clip(action_time_s / self.spec.duration_s, 0.0, 1.0)
        phase = np.pi * progress
        mode_index = {"pass": 0, "shot": 1, "clear": 2}[self.spec.action_mode]
        action_mode = np.eye(3, dtype=np.float64)[mode_index]
        joint_position_offset = data.qpos[self._joint_qpos] - self._default_pose
        joint_velocity = data.qvel[self._joint_dof]
        phase_fields: list[np.ndarray] = [
            np.array([np.sin(phase), np.cos(phase)])
        ]
        if self.contract.policy_name == "kick_policy_v3":
            locomotion = estimate_locomotion_phase(
                joint_position_offset,
                joint_velocity,
                self.contract.joint_order,
            )
            phase_fields.extend([locomotion.sin_cos, locomotion.support_hint])
        elif self.contract.policy_name == "kick_policy_v2":
            phase_fields.append(np.array([0.0, 1.0, 0.0]))
        else:
            raise RuntimeError(
                f"teacher does not support contract {self.contract.policy_name!r}"
            )
        actor = np.concatenate(
            [
                data.sensordata[self._gyro_slice],
                gravity,
                joint_position_offset,
                joint_velocity,
                previous_action,
                np.array(
                    [
                        ball_local_xy[0],
                        ball_local_xy[1],
                        ball_pos[2] - torso_pos[2],
                    ]
                ),
                np.array(
                    [
                        ball_local_vel_xy[0],
                        ball_local_vel_xy[1],
                        ball_world_vel[2],
                    ]
                ),
                target_local,
                np.array([self.spec.target_distance_m]),
                np.array([self.spec.requested_ball_speed_mps]),
                np.array([self.spec.desired_arrival_speed_mps]),
                action_mode,
                np.array([0.0, 1.0]),
                *phase_fields,
            ]
        ).astype(np.float32)
        if actor.shape != (self.contract.observation_size,):
            raise RuntimeError(
                f"teacher observation has shape {actor.shape}, expected "
                f"({self.contract.observation_size},)"
            )
        return actor

    def rollout(
        self,
        parameters: np.ndarray | None,
        *,
        capture_targets: bool = False,
        ball_x_offset_m: float = 0.0,
        ball_y_offset_m: float = 0.0,
        phase_reference_ball_x_offset_m: float | None = None,
        phase_alignment_s_per_m: float = 0.0,
        setup_ball_x_offset_m: float | None = None,
        setup_ball_y_offset_m: float | None = None,
        setup_timeout_s: float = 1.2,
        setup_tolerance_m: float = 0.015,
        setup_confirmation_cycles: int = 5,
        initial_robot_offset_m: tuple[float, float] = (0.0, 0.0),
        initial_robot_yaw_deg: float = 0.0,
        initial_qpos: np.ndarray | None = None,
        initial_qvel: np.ndarray | None = None,
        initial_walk_previous_action: np.ndarray | None = None,
        capture_transition_entry: bool = False,
        stop_after_transition_capture: bool = False,
        kick_policy_session: ort.InferenceSession | None = None,
        kick_correction_session: ort.InferenceSession | None = None,
        kick_correction_scale: float = 0.5,
    ) -> dict[str, float | bool]:
        setup_enabled = (
            setup_ball_x_offset_m is not None and setup_ball_y_offset_m is not None
        )
        if (setup_ball_x_offset_m is None) != (setup_ball_y_offset_m is None):
            raise ValueError("setup ball x/y offsets must be provided together")
        if (initial_qpos is None) != (initial_qvel is None):
            raise ValueError("initial qpos and qvel must be provided together")
        exact_initial_state = initial_qpos is not None
        if initial_walk_previous_action is not None:
            initial_walk_previous_action = np.asarray(
                initial_walk_previous_action, dtype=np.float64
            )
            if initial_walk_previous_action.shape != (self.contract.action_size,):
                raise ValueError("initial walk previous action has incompatible shape")
            if not np.isfinite(initial_walk_previous_action).all():
                raise ValueError("initial walk previous action must be finite")
            if not exact_initial_state:
                raise ValueError(
                    "initial walk previous action requires an exact initial state"
                )
        if stop_after_transition_capture and (
            not capture_transition_entry or not setup_enabled
        ):
            raise ValueError(
                "stopping after transition capture requires setup capture mode"
            )
        if exact_initial_state:
            initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
            initial_qvel = np.asarray(initial_qvel, dtype=np.float64)
            if initial_qpos.shape != (self.model.nq,) or initial_qvel.shape != (
                self.model.nv,
            ):
                raise ValueError("initial qpos/qvel have incompatible shapes")
            if not np.isfinite(initial_qpos).all() or not np.isfinite(
                initial_qvel
            ).all():
                raise ValueError("initial qpos/qvel must be finite")
            if setup_enabled:
                raise ValueError("exact initial state cannot also run setup")
        if phase_reference_ball_x_offset_m is None:
            phase_reference_ball_x_offset_m = ball_x_offset_m
        initial_robot_offset = np.asarray(initial_robot_offset_m, dtype=np.float64)
        if initial_robot_offset.shape != (2,):
            raise ValueError("initial robot offset must contain x and y")
        if not np.isfinite(
            [
                ball_x_offset_m,
                ball_y_offset_m,
                phase_reference_ball_x_offset_m,
                phase_alignment_s_per_m,
                setup_ball_x_offset_m if setup_enabled else 0.0,
                setup_ball_y_offset_m if setup_enabled else 0.0,
                setup_timeout_s,
                setup_tolerance_m,
                initial_robot_offset[0],
                initial_robot_offset[1],
                initial_robot_yaw_deg,
            ]
        ).all():
            raise ValueError("ball and initial robot offsets must be finite")
        if np.max(np.abs(initial_robot_offset)) > 2.0:
            raise ValueError("initial robot offset must stay within two metres")
        if abs(initial_robot_yaw_deg) > 45.0:
            raise ValueError("initial robot yaw must stay within 45 degrees")
        if exact_initial_state and (
            np.any(initial_robot_offset != 0.0)
            or initial_robot_yaw_deg != 0.0
            or ball_x_offset_m != 0.0
            or ball_y_offset_m != 0.0
        ):
            raise ValueError(
                "exact initial state is exclusive with pose and ball offsets"
            )
        if (
            setup_timeout_s <= 0.0
            or setup_tolerance_m <= 0.0
            or setup_confirmation_cycles < 1
        ):
            raise ValueError(
                "setup timing, tolerance and confirmation must be positive"
            )
        if kick_policy_session is not None and kick_correction_session is not None:
            raise ValueError("standalone and correction kick policies are exclusive")
        if kick_correction_session is not None and parameters is None:
            raise ValueError("a kick correction policy requires teacher parameters")
        if not 0.0 < kick_correction_scale <= 0.5:
            raise ValueError("kick_correction_scale must be in (0, 0.5]")
        if (
            parameters is None
            and kick_policy_session is None
            and kick_correction_session is None
        ):
            raise ValueError("provide parameters, kick_policy_session, or both")
        control_times = np.arange(
            0.0,
            self.spec.evaluation_duration_s + 0.5 * self.spec.control_dt_s,
            self.spec.control_dt_s,
        )
        # Preserve every optimized node exactly, but align contact timing
        # inside its Voronoi cell.  Apollo approaches at 0.50 m/s, so the
        # longitudinal difference converts directly into a bounded time shift.
        phase_shift_s = np.clip(
            (ball_x_offset_m - phase_reference_ball_x_offset_m)
            * phase_alignment_s_per_m,
            -0.10,
            0.10,
        )
        action_times = np.clip(control_times - phase_shift_s, 0.0, None)
        data = mujoco.MjData(self.model)
        if exact_initial_state:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
        else:
            data.qpos[self._ball_qpos] += ball_x_offset_m
            data.qpos[self._ball_qpos + 1] += ball_y_offset_m
            data.qpos[self._root_qpos : self._root_qpos + 2] += initial_robot_offset
            yaw_half = 0.5 * np.deg2rad(initial_robot_yaw_deg)
            yaw_quaternion = np.array(
                [np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)], dtype=np.float64
            )
            base_quaternion = data.qpos[
                self._root_qpos + 3 : self._root_qpos + 7
            ].copy()
            rotated_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(rotated_quaternion, yaw_quaternion, base_quaternion)
            data.qpos[self._root_qpos + 3 : self._root_qpos + 7] = (
                rotated_quaternion / np.linalg.norm(rotated_quaternion)
            )
        data.ctrl[self._pos_actuator] = self._default_pose
        mujoco.mj_forward(self.model, data)
        self._captured_transition_entry = None
        initial_ball = data.xpos[self._ball_body, :2].copy()
        target_angle = np.deg2rad(self.spec.target_angle_deg)
        target_direction = np.array([np.cos(target_angle), np.sin(target_angle)])
        lateral_direction = np.array([-target_direction[1], target_direction[0]])
        maximum_directional_speed = 0.0
        minimum_torso_height = float("inf")
        minimum_upright = 1.0
        closest_range_error = float("inf")
        closest_progress = 0.0
        closest_lateral_error = float("inf")
        closest_directional_speed = 0.0
        maximum_progress = 0.0
        previous_action = (
            initial_walk_previous_action.copy()
            if initial_walk_previous_action is not None
            else np.zeros(self.contract.action_size, dtype=np.float64)
        )
        previous_kick_action = np.zeros(self.contract.action_size, dtype=np.float64)
        captured_targets: list[np.ndarray] = []
        captured_observations: list[np.ndarray] = []
        captured_actions: list[np.ndarray] = []
        saturated_action_values = 0
        action_value_count = 0
        kick_started = not setup_enabled
        kick_start_index = 0
        setup_succeeded = not setup_enabled
        setup_timed_out = False
        setup_aligned_cycles = 0
        setup_max_aligned_cycles = 0
        setup_duration_s = 0.0
        setup_position_error_m = 0.0
        setup_initial_position_error_m = 0.0
        minimum_setup_position_error_m = float("inf")
        setup_kick_start_yaw_deg = 0.0
        kick_policy_input = (
            kick_policy_session.get_inputs()[0].name
            if kick_policy_session is not None
            else None
        )
        kick_correction_input = (
            kick_correction_session.get_inputs()[0].name
            if kick_correction_session is not None
            else None
        )
        if capture_transition_entry and not setup_enabled:
            self._captured_transition_entry = self._make_transition_entry(
                data,
                np.zeros(self.contract.action_size, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )

        substeps = int(round(self.spec.control_dt_s / self.spec.simulation_dt_s))
        for control_index, default_action_time_s in enumerate(action_times):
            elapsed = control_times[control_index]
            action_time_s = float(default_action_time_s)
            if setup_enabled and not kick_started and not setup_timed_out:
                torso_xmat = data.site_xmat[self._torso_site].reshape(3, 3)
                yaw = np.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
                c, s = np.cos(yaw), np.sin(yaw)
                world_to_yaw = np.array([[c, s], [-s, c]])
                ball_local_xy = world_to_yaw @ (
                    data.xpos[self._ball_body, :2]
                    - data.site_xpos[self._torso_site, :2]
                )
                desired_ball_local_xy = np.array(
                    [
                        0.32 + float(setup_ball_x_offset_m),
                        float(setup_ball_y_offset_m),
                    ]
                )
                setup_error = ball_local_xy - desired_ball_local_xy
                setup_position_error_m = float(np.linalg.norm(setup_error))
                if control_index == 0:
                    setup_initial_position_error_m = setup_position_error_m
                minimum_setup_position_error_m = min(
                    minimum_setup_position_error_m, setup_position_error_m
                )
                velocity_command = np.array(
                    [
                        np.clip(8.0 * setup_error[0], -0.50, 1.00),
                        np.clip(8.0 * setup_error[1], -0.50, 0.50),
                        np.clip(-2.0 * yaw, -0.50, 0.50),
                    ],
                    dtype=np.float64,
                )
                if np.max(np.abs(setup_error)) <= setup_tolerance_m:
                    setup_aligned_cycles += 1
                    setup_max_aligned_cycles = max(
                        setup_max_aligned_cycles, setup_aligned_cycles
                    )
                else:
                    setup_aligned_cycles = 0
                action_time_s = 0.0
                if setup_aligned_cycles >= setup_confirmation_cycles:
                    kick_started = True
                    setup_succeeded = True
                    setup_duration_s = float(elapsed)
                    kick_start_index = control_index
                    setup_kick_start_yaw_deg = float(np.rad2deg(yaw))
                    if capture_transition_entry:
                        self._captured_transition_entry = (
                            self._make_transition_entry(
                                data, previous_action, velocity_command
                            )
                        )
                        if stop_after_transition_capture:
                            break
                    previous_action.fill(0.0)
                    previous_kick_action.fill(0.0)
                    velocity_command = np.array([0.50, -0.04, 0.0])
                elif elapsed >= setup_timeout_s:
                    setup_timed_out = True
                    setup_duration_s = float(elapsed)
                    velocity_command = np.zeros(3, dtype=np.float64)
            elif setup_enabled and setup_timed_out:
                action_time_s = 0.0
                velocity_command = np.zeros(3, dtype=np.float64)
            else:
                if setup_enabled:
                    action_time_s = float(
                        (control_index - kick_start_index) * self.spec.control_dt_s
                    )
                velocity_command = np.array(
                    ([0.50, -0.04, 0.0] if action_time_s < 0.65 else [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
            residual_target = np.zeros(self.contract.action_size, dtype=np.float64)
            if parameters is not None and kick_started:
                raw_residual_target = build_joint_delta_trajectory(
                    parameters,
                    self.contract,
                    np.asarray([action_time_s], dtype=np.float64),
                )[0]
                residual_target = (
                    np.clip(
                        self._default_pose + raw_residual_target,
                        self._lowers,
                        self._uppers,
                    )
                    - self._default_pose
                )
            kick_observation = self._kick_actor_observation(
                data, previous_kick_action, float(action_time_s)
            )
            stable_target, previous_action = self._stable_walk_target(
                data, previous_action, velocity_command
            )
            teacher_action = residual_target / KICK_ACTION_SCALE
            correction_action = None
            if kick_correction_session is not None and kick_started:
                correction_action = kick_correction_session.run(
                    None,
                    {kick_correction_input: kick_observation[None, :]},
                )[0][0].astype(np.float64)
                correction_action = np.nan_to_num(
                    correction_action, nan=0.0, posinf=1.0, neginf=-1.0
                )
                raw_kick_action = correction_action
                label_action = correction_action
            elif kick_policy_session is not None and kick_started:
                policy_action = kick_policy_session.run(
                    None,
                    {kick_policy_input: kick_observation[None, :]},
                )[0][0].astype(np.float64)
                policy_action = np.nan_to_num(
                    policy_action, nan=0.0, posinf=1.0, neginf=-1.0
                )
                raw_kick_action = policy_action
                label_action = (
                    teacher_action if parameters is not None else policy_action
                )
            else:
                raw_kick_action = teacher_action
                label_action = raw_kick_action
            saturated_action_values += int(
                np.count_nonzero(np.abs(raw_kick_action) > 1.0)
            )
            action_value_count += raw_kick_action.size
            kick_action = np.clip(raw_kick_action, -1.0, 1.0)
            if correction_action is not None:
                target = np.clip(
                    stable_target
                    + residual_target
                    + kick_action * (kick_correction_scale * KICK_ACTION_SCALE),
                    self._lowers,
                    self._uppers,
                )
            else:
                target = np.clip(
                    stable_target + kick_action * KICK_ACTION_SCALE,
                    self._lowers,
                    self._uppers,
                )
            if capture_targets:
                captured_observations.append(kick_observation)
                captured_actions.append(
                    np.clip(label_action, -1.0, 1.0).astype(np.float32)
                )
                captured_targets.append(target.copy())
            previous_kick_action = kick_action
            data.ctrl[self._pos_actuator] = target
            for _ in range(substeps):
                mujoco.mj_step(self.model, data)
                ball_velocity = data.qvel[self._ball_dof : self._ball_dof + 2]
                displacement_now = data.xpos[self._ball_body, :2] - initial_ball
                progress_now = float(np.dot(displacement_now, target_direction))
                lateral_now = float(abs(np.dot(displacement_now, lateral_direction)))
                range_error_now = abs(progress_now - self.spec.target_distance_m)
                directional_speed_now = float(np.dot(ball_velocity, target_direction))
                maximum_progress = max(maximum_progress, progress_now)
                if range_error_now < closest_range_error:
                    closest_range_error = range_error_now
                    closest_progress = progress_now
                    closest_lateral_error = lateral_now
                    closest_directional_speed = directional_speed_now
                maximum_directional_speed = max(
                    maximum_directional_speed,
                    directional_speed_now,
                )
                torso_height = float(data.xpos[self._torso_body, 2])
                torso_upright = float(data.xmat[self._torso_body].reshape(3, 3)[2, 2])
                minimum_torso_height = min(minimum_torso_height, torso_height)
                minimum_upright = min(minimum_upright, torso_upright)
            if (
                kick_policy_session is not None or kick_correction_session is not None
            ) and (minimum_torso_height < 0.35 or minimum_upright < 0.0):
                break

        if capture_targets:
            self._captured_targets = np.asarray(captured_targets)
            self._captured_observations = np.asarray(captured_observations)
            self._captured_actions = np.asarray(captured_actions)

        displacement = data.xpos[self._ball_body, :2] - initial_ball
        final_progress = float(np.dot(displacement, target_direction))
        final_lateral_error = float(abs(np.dot(displacement, lateral_direction)))
        speed_error = abs(
            maximum_directional_speed - self.spec.requested_ball_speed_mps
        )
        arrival_speed_error = abs(
            closest_directional_speed - self.spec.desired_arrival_speed_mps
        )
        fell = minimum_torso_height < 0.35 or minimum_upright < 0.0
        contact = maximum_directional_speed >= 0.15 or maximum_progress >= 0.08
        parameter_cost = (
            float(np.mean(np.square(parameters / PARAMETER_UPPER)))
            if parameters is not None
            else 0.0
        )
        score = (
            3.0 * min(max(maximum_progress, 0.0), self.spec.target_distance_m)
            - 2.0 * closest_range_error
            - 4.0 * speed_error
            - 2.0 * arrival_speed_error
            - 6.0 * closest_lateral_error
            + (1.0 if contact else -2.0)
            - (12.0 if fell else 0.0)
            - 0.05 * parameter_cost
        )
        if not np.isfinite(score):
            score = -1.0e9
        return {
            "score": float(score),
            "progress_m": final_progress,
            "maximum_progress_m": maximum_progress,
            "closest_target_progress_m": closest_progress,
            "lateral_error_m": closest_lateral_error,
            "final_lateral_error_m": final_lateral_error,
            "range_error_m": closest_range_error,
            "maximum_directional_speed_mps": maximum_directional_speed,
            "closest_target_speed_mps": closest_directional_speed,
            "speed_error_mps": speed_error,
            "arrival_speed_error_mps": arrival_speed_error,
            "minimum_torso_height_m": minimum_torso_height,
            "minimum_upright": minimum_upright,
            "contact": contact,
            "fell": fell,
            "setup_enabled": setup_enabled,
            "setup_succeeded": setup_succeeded,
            "setup_timed_out": setup_timed_out,
            "setup_duration_s": setup_duration_s,
            "setup_position_error_m": setup_position_error_m,
            "setup_initial_position_error_m": setup_initial_position_error_m,
            "minimum_setup_position_error_m": (
                minimum_setup_position_error_m if setup_enabled else 0.0
            ),
            "setup_max_aligned_cycles": setup_max_aligned_cycles,
            "setup_kick_start_yaw_deg": setup_kick_start_yaw_deg,
            "action_saturation_fraction": (
                saturated_action_values / action_value_count
                if action_value_count
                else 0.0
            ),
        }

    def objective(self, parameters: np.ndarray) -> float:
        return float(self.rollout(parameters)["score"])

    def rollout_policy(
        self,
        session: ort.InferenceSession,
        *,
        ball_x_offset_m: float = 0.0,
        ball_y_offset_m: float = 0.0,
    ) -> dict[str, float | bool]:
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        if input_meta.shape != [1, self.contract.observation_size]:
            raise ValueError("kick ONNX input does not match the v2 contract")
        if output_meta.shape != [1, self.contract.action_size]:
            raise ValueError("kick ONNX output does not match the v2 contract")
        return self.rollout(
            None,
            ball_x_offset_m=ball_x_offset_m,
            ball_y_offset_m=ball_y_offset_m,
            kick_policy_session=session,
        )

    def dagger_demonstration(
        self,
        teacher_parameters: np.ndarray,
        session: ort.InferenceSession,
        *,
        ball_x_offset_m: float = 0.0,
        ball_y_offset_m: float = 0.0,
        initial_qpos: np.ndarray | None = None,
        initial_qvel: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool]]:
        """Execute the learner and label every visited state with the teacher."""
        metrics = self.rollout(
            teacher_parameters,
            capture_targets=True,
            ball_x_offset_m=ball_x_offset_m,
            ball_y_offset_m=ball_y_offset_m,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
            kick_policy_session=session,
        )
        sample_count = self._captured_observations.shape[0]
        times = np.arange(sample_count, dtype=np.float64) * self.spec.control_dt_s
        return (
            times,
            self._captured_observations.copy(),
            self._captured_actions.copy(),
            metrics,
        )

    def optimize(
        self,
        *,
        seed: int,
        population: int,
        generations: int,
        robust_samples: int = 1,
        ball_x_offset_m: float = 0.0,
        ball_y_offset_m: float = 0.0,
        initial_parameters: np.ndarray | None = None,
        ball_x_range_m: tuple[float, float] | None = None,
        ball_y_range_m: tuple[float, float] | None = None,
        setup_ball_x_offset_m: float | None = None,
        setup_ball_y_offset_m: float | None = None,
        setup_timeout_s: float = 1.2,
        setup_tolerance_m: float = 0.015,
        setup_confirmation_cycles: int = 5,
    ) -> CEMResult:
        if robust_samples < 1:
            raise ValueError("robust_samples must be positive")
        if (ball_x_range_m is None) != (ball_y_range_m is None):
            raise ValueError("robust ball x/y ranges must be provided together")
        if ball_x_range_m is not None:
            if (
                len(ball_x_range_m) != 2
                or len(ball_y_range_m) != 2
                or ball_x_range_m[0] > ball_x_range_m[1]
                or ball_y_range_m[0] > ball_y_range_m[1]
            ):
                raise ValueError("robust ball ranges must be ordered pairs")
        perturbation_rng = np.random.default_rng(seed + 1_000_003)
        perturbations = [(ball_x_offset_m, ball_y_offset_m)]
        if ball_x_range_m is None:
            perturbations.extend(
                (
                    float(ball_x_offset_m + perturbation_rng.uniform(-0.02, 0.02)),
                    float(ball_y_offset_m + perturbation_rng.uniform(-0.03, 0.03)),
                )
                for _ in range(robust_samples - 1)
            )
        else:
            perturbations.extend(
                (
                    float(
                        perturbation_rng.uniform(ball_x_range_m[0], ball_x_range_m[1])
                    ),
                    float(
                        perturbation_rng.uniform(ball_y_range_m[0], ball_y_range_m[1])
                    ),
                )
                for _ in range(robust_samples - 1)
            )

        def robust_objective(parameters: np.ndarray) -> float:
            scores = np.asarray(
                [
                    self.rollout(
                        parameters,
                        ball_x_offset_m=ball_x,
                        ball_y_offset_m=ball_y,
                        setup_ball_x_offset_m=setup_ball_x_offset_m,
                        setup_ball_y_offset_m=setup_ball_y_offset_m,
                        setup_timeout_s=setup_timeout_s,
                        setup_tolerance_m=setup_tolerance_m,
                        setup_confirmation_cycles=setup_confirmation_cycles,
                    )["score"]
                    for ball_x, ball_y in perturbations
                ],
                dtype=np.float64,
            )
            # Deployment needs a floor, not a high average hiding one bad ball
            # placement.  Keep some pressure on mean quality while making the
            # worst deterministic perturbation dominate the search objective.
            return float(0.35 * np.mean(scores) + 0.65 * np.min(scores))

        if initial_parameters is None:
            initial_mean = np.zeros(len(PARAMETER_NAMES), dtype=np.float64)
            initial_std = 0.35 * (PARAMETER_UPPER - PARAMETER_LOWER)
        else:
            initial_mean = np.asarray(initial_parameters, dtype=np.float64)
            if initial_mean.shape != (len(PARAMETER_NAMES),):
                raise ValueError("initial_parameters has the wrong shape")
            initial_mean = np.clip(initial_mean, PARAMETER_LOWER, PARAMETER_UPPER)
            initial_std = 0.18 * (PARAMETER_UPPER - PARAMETER_LOWER)
        return cem_optimize(
            robust_objective,
            initial_mean=initial_mean,
            initial_std=initial_std,
            lower=PARAMETER_LOWER,
            upper=PARAMETER_UPPER,
            seed=seed,
            population=population,
            generations=generations,
        )

    def trajectory(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.arange(
            0.0,
            self.spec.evaluation_duration_s + 0.5 * self.spec.control_dt_s,
            self.spec.control_dt_s,
        )
        self.rollout(parameters, capture_targets=True)
        return times, self._captured_targets.copy()

    def demonstration(
        self,
        parameters: np.ndarray,
        *,
        ball_x_offset_m: float = 0.0,
        ball_y_offset_m: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool]]:
        """Return exact observations/actions/targets for one optimized condition."""
        times = np.arange(
            0.0,
            self.spec.evaluation_duration_s + 0.5 * self.spec.control_dt_s,
            self.spec.control_dt_s,
        )
        metrics = self.rollout(
            parameters,
            capture_targets=True,
            ball_x_offset_m=ball_x_offset_m,
            ball_y_offset_m=ball_y_offset_m,
        )
        return (
            times,
            self._captured_observations.copy(),
            self._captured_actions.copy(),
            self._captured_targets.copy(),
            metrics,
        )


def kick_trial_success(
    metrics: dict[str, float | bool],
    *,
    range_tolerance_m: float = 0.5,
    corridor_half_width_m: float = 0.5,
    launch_speed_tolerance_mps: float = 1.0,
) -> bool:
    """Classify one teacher trial using explicit R1 pre-promotion tolerances."""
    return bool(
        metrics["contact"]
        and not metrics["fell"]
        and float(metrics["range_error_m"]) <= range_tolerance_m
        and float(metrics["lateral_error_m"]) <= corridor_half_width_m
        and float(metrics["speed_error_mps"]) <= launch_speed_tolerance_mps
    )
