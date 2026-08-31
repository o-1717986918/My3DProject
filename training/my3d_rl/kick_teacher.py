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
from .rcss_scene import DEFAULT_RESOURCE_ROOT, build_single_t1_soccer_model


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
APOLLO_DEFAULT_POSE = np.array(
    [
        0.0,
        0.0,
        0.0,
        -1.4,
        0.0,
        -0.4,
        0.0,
        1.4,
        0.0,
        0.4,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
    ],
    dtype=np.float64,
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
class KickTeacherSpec:
    target_distance_m: float = 2.0
    target_angle_deg: float = 0.0
    requested_ball_speed_mps: float = 1.43
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
            kp, kd = self._apollo_joint_gains(joint_name)
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

    @staticmethod
    def _apollo_joint_gains(name: str) -> tuple[float, float]:
        if name == "AAHead_yaw":
            return 10.0, 1.0
        if name == "Head_pitch":
            return 20.0, 1.0
        if name == "Waist":
            return 85.0, 5.0
        if "Shoulder" in name:
            return 45.0, 2.5
        if "Elbow" in name:
            return 30.0, 1.2
        if "Hip_Pitch" in name:
            return 130.0, 10.0
        if "Hip_Roll" in name:
            return 90.0, 8.0
        if "Hip_Yaw" in name:
            return 70.0, 3.0
        if "Knee" in name:
            return 140.0, 6.0
        if "Ankle_Pitch" in name:
            return 45.0, 2.0
        if "Ankle_Roll" in name:
            return 40.0, 1.8
        return 10.0, 0.1

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

    def rollout(
        self,
        parameters: np.ndarray,
        *,
        capture_targets: bool = False,
        ball_x_offset_m: float = 0.0,
        ball_y_offset_m: float = 0.0,
    ) -> dict[str, float | bool]:
        if not np.isfinite([ball_x_offset_m, ball_y_offset_m]).all():
            raise ValueError("ball offsets must be finite")
        control_times = np.arange(
            0.0,
            self.spec.evaluation_duration_s + 0.5 * self.spec.control_dt_s,
            self.spec.control_dt_s,
        )
        deltas = build_joint_delta_trajectory(parameters, self.contract, control_times)
        targets = np.clip(
            self._default_pose[None, :] + deltas, self._lowers, self._uppers
        )

        data = mujoco.MjData(self.model)
        data.qpos[self._ball_qpos] += ball_x_offset_m
        data.qpos[self._ball_qpos + 1] += ball_y_offset_m
        data.ctrl[self._pos_actuator] = self._default_pose
        mujoco.mj_forward(self.model, data)
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
        previous_action = np.zeros(self.contract.action_size, dtype=np.float64)
        captured_targets: list[np.ndarray] = []

        substeps = int(round(self.spec.control_dt_s / self.spec.simulation_dt_s))
        for control_index, residual_target in enumerate(
            targets - self._default_pose[None, :]
        ):
            elapsed = control_times[control_index]
            velocity_command = np.array(
                [0.50, -0.04, 0.0] if elapsed < 0.65 else [0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            stable_target, previous_action = self._stable_walk_target(
                data, previous_action, velocity_command
            )
            target = np.clip(
                stable_target + residual_target, self._lowers, self._uppers
            )
            if capture_targets:
                captured_targets.append(target.copy())
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
            if control_index + 1 >= targets.shape[0]:
                break

        if capture_targets:
            self._captured_targets = np.asarray(captured_targets)

        displacement = data.xpos[self._ball_body, :2] - initial_ball
        final_progress = float(np.dot(displacement, target_direction))
        final_lateral_error = float(abs(np.dot(displacement, lateral_direction)))
        speed_error = abs(
            maximum_directional_speed - self.spec.requested_ball_speed_mps
        )
        fell = minimum_torso_height < 0.35 or minimum_upright < 0.0
        contact = maximum_directional_speed >= 0.15 or maximum_progress >= 0.08
        parameter_cost = float(np.mean(np.square(parameters / PARAMETER_UPPER)))
        score = (
            3.0 * min(max(maximum_progress, 0.0), self.spec.target_distance_m)
            - 2.0 * closest_range_error
            - 2.0 * speed_error
            - 3.0 * closest_lateral_error
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
            "minimum_torso_height_m": minimum_torso_height,
            "minimum_upright": minimum_upright,
            "contact": contact,
            "fell": fell,
        }

    def objective(self, parameters: np.ndarray) -> float:
        return float(self.rollout(parameters)["score"])

    def optimize(
        self,
        *,
        seed: int,
        population: int,
        generations: int,
        robust_samples: int = 1,
    ) -> CEMResult:
        if robust_samples < 1:
            raise ValueError("robust_samples must be positive")
        perturbation_rng = np.random.default_rng(seed + 1_000_003)
        perturbations = [(0.0, 0.0)]
        perturbations.extend(
            (
                float(perturbation_rng.uniform(-0.01, 0.08)),
                float(perturbation_rng.uniform(-0.08, 0.08)),
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
                    )["score"]
                    for ball_x, ball_y in perturbations
                ],
                dtype=np.float64,
            )
            return float(np.mean(scores) - 0.25 * np.std(scores))

        initial_mean = np.zeros(len(PARAMETER_NAMES), dtype=np.float64)
        initial_std = 0.35 * (PARAMETER_UPPER - PARAMETER_LOWER)
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
