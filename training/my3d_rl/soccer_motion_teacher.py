"""Exact-CPU phase correction teachers for finite soccer motions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contract import PolicyContract
from .rcss_scene import DEFAULT_RESOURCE_ROOT, build_single_t1_soccer_model
from .reference_dynamics import configure_pd_actuators
from .soccer_motion_corpus import SoccerMotionCorpus
from .soccer_motion_policy import SoccerMotionPolicy, soccer_motion_actor_observation
from .soccer_motion_reset import (
    derive_case_seed,
    deterministic_reset_perturbation,
    yaw_quaternion_rotate,
    yaw_vector_rotate,
)
from .t1_control import apollo_joint_gains


TEACHER_JOINT_CANDIDATES = (
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)


def decode_phase_correction(
    parameters: np.ndarray,
    *,
    phases: np.ndarray,
    action_size: int,
    joint_indices: Sequence[int],
    knot_count: int,
    maximum_abs_correction: float,
) -> np.ndarray:
    """Decode bounded knots into a smooth correction over global clip phase."""
    parameters = np.asarray(parameters, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    indices = np.asarray(joint_indices, dtype=np.int64)
    if phases.ndim != 1 or not np.isfinite(phases).all():
        raise ValueError("phases must be a finite vector")
    if np.any(phases < 0.0) or np.any(phases > 1.0):
        raise ValueError("phases must lie in [0, 1]")
    if action_size < 1 or knot_count < 2:
        raise ValueError("action size and knot count are too small")
    if indices.ndim != 1 or indices.size < 1 or len(set(indices.tolist())) != indices.size:
        raise ValueError("joint indices must be a non-empty unique vector")
    if np.any(indices < 0) or np.any(indices >= action_size):
        raise ValueError("joint index lies outside the action")
    expected = knot_count * indices.size
    if parameters.shape != (expected,) or not np.isfinite(parameters).all():
        raise ValueError(f"expected {expected} finite correction parameters")
    if maximum_abs_correction <= 0.0 or np.any(
        np.abs(parameters) > maximum_abs_correction + 1.0e-12
    ):
        raise ValueError("correction exceeds its declared bound")

    knots = parameters.reshape(knot_count, indices.size)
    knot_phases = np.linspace(0.0, 1.0, knot_count)
    right = np.searchsorted(knot_phases, phases, side="right")
    right = np.clip(right, 1, knot_count - 1)
    left = right - 1
    width = knot_phases[right] - knot_phases[left]
    fraction = np.clip((phases - knot_phases[left]) / width, 0.0, 1.0)
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    active = knots[left] * (1.0 - fraction[:, None]) + knots[right] * fraction[:, None]
    correction = np.zeros((phases.size, action_size), dtype=np.float64)
    correction[:, indices] = active
    return correction


def robust_teacher_objective(
    scores: np.ndarray, *, minimum_weight: float = 0.35
) -> float:
    """Prefer typical improvement while protecting the weakest train restart."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("teacher scores must be a non-empty finite vector")
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum weight must lie in [0, 1]")
    return float(
        (1.0 - minimum_weight) * np.mean(values)
        + minimum_weight * np.min(values)
    )


def select_dagger_action(
    base_action: np.ndarray,
    teacher_action: np.ndarray,
    *,
    teacher_probability: float,
    action_clip: tuple[float, float],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool]:
    """Choose the executed action while retaining an expert label."""
    base_action = np.asarray(base_action, dtype=np.float64)
    teacher_action = np.asarray(teacher_action, dtype=np.float64)
    if base_action.shape != teacher_action.shape or base_action.ndim != 1:
        raise ValueError("DAgger actions must be equal one-dimensional vectors")
    if not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher probability must lie in [0, 1]")
    if 0.0 < teacher_probability < 1.0 and rng is None:
        raise ValueError("mixed teacher execution requires a random generator")
    use_teacher = teacher_probability == 1.0 or (
        0.0 < teacher_probability < 1.0
        and rng is not None
        and bool(rng.random() < teacher_probability)
    )
    selected = teacher_action if use_teacher else base_action
    return np.clip(selected, *action_clip), use_teacher


def state_feedback_action_candidates(
    student_action: np.ndarray,
    nominal_teacher_action: np.ndarray,
    position_error: np.ndarray,
    velocity_error: np.ndarray,
    *,
    active_joint_indices: Sequence[int],
    action_scale: float,
    control_period: float,
    action_clip: tuple[float, float],
    maximum_delta_from_student: float = 1.0,
) -> np.ndarray:
    """Build bounded, deterministic actions for short-horizon teacher search."""
    student_action = np.asarray(student_action, dtype=np.float64)
    nominal_teacher_action = np.asarray(nominal_teacher_action, dtype=np.float64)
    position_error = np.asarray(position_error, dtype=np.float64)
    velocity_error = np.asarray(velocity_error, dtype=np.float64)
    if (
        student_action.ndim != 1
        or nominal_teacher_action.shape != student_action.shape
        or position_error.shape != student_action.shape
        or velocity_error.shape != student_action.shape
    ):
        raise ValueError("state-feedback teacher vectors are incompatible")
    if (
        action_scale <= 0.0
        or control_period <= 0.0
        or maximum_delta_from_student <= 0.0
    ):
        raise ValueError("state-feedback decoder scales must be positive")
    active = np.asarray(active_joint_indices, dtype=np.int64)
    if active.ndim != 1 or active.size < 1 or np.any(active < 0) or np.any(
        active >= student_action.size
    ):
        raise ValueError("state-feedback active joints are invalid")
    feedback = np.zeros(student_action.shape, dtype=np.float64)
    feedback[active] = (
        position_error[active] + control_period * velocity_error[active]
    ) / action_scale
    feedback = np.clip(feedback, -0.5, 0.5)
    nodes = [
        student_action,
        nominal_teacher_action,
        0.5 * (student_action + nominal_teacher_action),
        nominal_teacher_action + 0.25 * feedback,
        nominal_teacher_action + 0.5 * feedback,
        nominal_teacher_action + feedback,
    ]
    lower = np.maximum(action_clip[0], student_action - maximum_delta_from_student)
    upper = np.minimum(action_clip[1], student_action + maximum_delta_from_student)
    return np.stack([np.clip(node, lower, upper) for node in nodes])


def cross_fitted_label_selected(
    student_action: np.ndarray,
    teacher_action: np.ndarray,
    *,
    search_improvement: float,
    minimum_search_improvement: float,
    validation_improvement: float | None = None,
    minimum_validation_improvement: float = 0.0,
) -> bool:
    """Accept a label only when search and independent validation agree."""
    student_action = np.asarray(student_action, dtype=np.float64)
    teacher_action = np.asarray(teacher_action, dtype=np.float64)
    if student_action.shape != teacher_action.shape or student_action.ndim != 1:
        raise ValueError("cross-fitted actions are incompatible")
    if min(minimum_search_improvement, minimum_validation_improvement) < 0.0:
        raise ValueError("cross-fitted improvement thresholds must be non-negative")
    if not np.isfinite(search_improvement) or (
        validation_improvement is not None
        and not np.isfinite(validation_improvement)
    ):
        raise ValueError("cross-fitted improvements must be finite")
    return bool(
        search_improvement >= minimum_search_improvement
        and (
            validation_improvement is None
            or validation_improvement >= minimum_validation_improvement
        )
        and not np.allclose(teacher_action, student_action, atol=1.0e-6)
    )


class SoccerMotionCorrectionEvaluator:
    """Evaluate a base actor plus an open-loop phase correction in MuJoCo."""

    def __init__(
        self,
        corpus: SoccerMotionCorpus,
        contract: PolicyContract,
        base_policy: SoccerMotionPolicy,
        *,
        resource_root: Path = DEFAULT_RESOURCE_ROOT,
        prefix: str = "soccer_teacher_",
    ) -> None:
        if contract.observation_size != 110 or contract.action_size != 23:
            raise ValueError("soccer-motion teacher requires the 110 -> 23 contract")
        self.corpus = corpus
        self.contract = contract
        self.base_policy = base_policy
        self.prefix = prefix
        self.model = build_single_t1_soccer_model(
            resource_root, prefix=prefix, robot_x=-10.0, robot_y=0.0
        )
        self.model.opt.timestep = 0.005
        gains = np.asarray(
            [apollo_joint_gains(name) for name in contract.joint_order],
            dtype=np.float64,
        )
        self._tau_actuator, self._pos_actuator, self._vel_actuator = (
            configure_pd_actuators(
                self.model,
                contract,
                kp=gains[:, 0],
                kd=gains[:, 1],
                prefix=prefix,
            )
        )
        self._joint_qpos = np.asarray(
            [self.model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
        )
        self._joint_dof = np.asarray(
            [self.model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
        )
        root = self.model.joint(prefix + "root")
        self._root_qpos = int(root.qposadr[0])
        self._root_dof = int(root.dofadr[0])
        self._torso_body = self.model.body(prefix + "torso").id
        self._torso_site = self.model.site(prefix + "torso").id
        gyro = self.model.sensor(prefix + "torso_gyro")
        self._gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
        self._pitch_geom = self.model.geom("pitch").id
        self._foot_geoms = (
            self.model.geom(prefix + "left_foot").id,
            self.model.geom(prefix + "right_foot").id,
        )
        self._lower = self.model.jnt_range[
            self.model.dof_jntid[self._joint_dof], 0
        ]
        self._upper = self.model.jnt_range[
            self.model.dof_jntid[self._joint_dof], 1
        ]
        self._substeps = round(
            (1.0 / contract.frequency_hz) / self.model.opt.timestep
        )
        if not np.isclose(
            self._substeps * self.model.opt.timestep,
            1.0 / contract.frequency_hz,
        ):
            raise ValueError("physics step must divide the actor control period")
        self._model_root_xy = self.model.qpos0[
            self._root_qpos : self._root_qpos + 2
        ].copy()

    def _contacts(self, data: mujoco.MjData) -> np.ndarray:
        pairs = np.asarray(data.contact.geom[: data.ncon], dtype=np.int32)
        return np.asarray(
            [
                np.any(
                    ((pairs[:, 0] == self._pitch_geom) & (pairs[:, 1] == foot))
                    | ((pairs[:, 1] == self._pitch_geom) & (pairs[:, 0] == foot))
                )
                for foot in self._foot_geoms
            ],
            dtype=bool,
        )

    def _candidate_rollout_cost(
        self,
        data: mujoco.MjData,
        *,
        first_action: np.ndarray,
        nominal_teacher_action: np.ndarray,
        previous_action: np.ndarray,
        motion: int,
        current_frame: int,
        correction: np.ndarray,
        teacher_base_policy: SoccerMotionPolicy,
        horizon: int,
    ) -> float:
        length = int(self.corpus.lengths[motion])
        trial = mujoco.MjData(self.model)
        mujoco.mj_copyData(trial, self.model, data)
        trial_previous_action = np.asarray(previous_action, dtype=np.float64)
        cost = 0.0
        for offset in range(min(horizon, length - 1 - current_frame)):
            frame = current_frame + offset
            if offset == 0:
                action = first_action
            else:
                observation = soccer_motion_actor_observation(
                    trial,
                    joint_qpos=self._joint_qpos,
                    joint_dof=self._joint_dof,
                    gyro_slice=self._gyro_slice,
                    torso_site=self._torso_site,
                    reference_joint_position=(
                        self.corpus.joint_position[motion, frame]
                    ),
                    reference_joint_velocity=(
                        self.corpus.joint_velocity[motion, frame]
                    ),
                    reference_root_linear_velocity=(
                        self.corpus.root_linear_velocity[motion, frame]
                    ),
                    reference_root_angular_velocity=(
                        self.corpus.root_angular_velocity[motion, frame]
                    ),
                    reference_contact=self.corpus.foot_contact[motion, frame],
                    previous_action=trial_previous_action,
                    progress=frame / max(length - 1, 1),
                    kick_leg_one_hot=self.corpus.kick_leg_one_hot[motion],
                )
                action = np.clip(
                    np.asarray(teacher_base_policy(observation), dtype=np.float64)
                    + correction[frame],
                    *self.contract.action_clip,
                )
            target = np.clip(
                self.corpus.joint_position[motion, frame + 1]
                + self.contract.action_scale * action,
                self._lower,
                self._upper,
            )
            trial.ctrl[self._pos_actuator] = target
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, trial)
            trial_previous_action = action
            joint_error = (
                trial.qpos[self._joint_qpos]
                - self.corpus.joint_position[motion, frame + 1]
            )
            velocity_error = (
                trial.qvel[self._joint_dof]
                - self.corpus.joint_velocity[motion, frame + 1]
            )
            rotation = trial.site_xmat[self._torso_site].reshape(3, 3)
            upright = float(rotation[2, 2])
            height = float(trial.xpos[self._torso_body, 2])
            contact_error = float(
                np.mean(
                    self._contacts(trial)
                    != self.corpus.foot_contact[motion, frame + 1]
                )
            )
            stage_cost = (
                10.0 * float(np.mean(np.square(joint_error)))
                + 0.02 * float(np.mean(np.square(velocity_error)))
                + 2.0 * (1.0 - np.clip(upright, -1.0, 1.0)) ** 2
                + 20.0 * max(0.45 - height, 0.0) ** 2
                + 0.1 * contact_error
            )
            if (
                not np.isfinite(trial.qpos).all()
                or not np.isfinite(trial.qvel).all()
                or height < 0.35
                or upright < 0.20
            ):
                stage_cost += 100.0
                cost += (0.9**offset) * stage_cost
                break
            cost += (0.9**offset) * stage_cost
        return cost + 0.01 * float(
            np.mean(np.square(first_action - nominal_teacher_action))
        )

    def short_horizon_teacher_action(
        self,
        data: mujoco.MjData,
        *,
        motion: int,
        current_frame: int,
        previous_action: np.ndarray,
        student_action: np.ndarray,
        nominal_teacher_action: np.ndarray,
        correction: np.ndarray,
        teacher_base_policy: SoccerMotionPolicy,
        horizon: int,
        maximum_action_delta: float,
        validation_horizon: int = 0,
        validation_seed: int | None = None,
        validation_joint_noise: float = 0.0,
        validation_joint_velocity_noise: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Select a state-feedback label using exact copied MuJoCo states."""
        length = int(self.corpus.lengths[motion])
        if (
            horizon < 1
            or validation_horizon < 0
            or min(validation_joint_noise, validation_joint_velocity_noise) < 0.0
            or not 0 <= current_frame < length - 1
        ):
            raise ValueError("short-horizon teacher request is invalid")
        if validation_horizon and validation_seed is None:
            raise ValueError("cross-fitted validation requires a validation seed")
        if self.contract.action_scale is None:
            raise ValueError("short-horizon teacher requires residual action scale")
        active = np.flatnonzero(np.any(np.abs(correction) > 1.0e-12, axis=0))
        candidates = state_feedback_action_candidates(
            student_action,
            nominal_teacher_action,
            self.corpus.joint_position[motion, current_frame]
            - data.qpos[self._joint_qpos],
            self.corpus.joint_velocity[motion, current_frame]
            - data.qvel[self._joint_dof],
            active_joint_indices=active,
            action_scale=self.contract.action_scale,
            control_period=1.0 / self.contract.frequency_hz,
            action_clip=self.contract.action_clip,
            maximum_delta_from_student=maximum_action_delta,
        )
        costs = [
            self._candidate_rollout_cost(
                data,
                first_action=first_action,
                nominal_teacher_action=nominal_teacher_action,
                previous_action=previous_action,
                motion=motion,
                current_frame=current_frame,
                correction=correction,
                teacher_base_policy=teacher_base_policy,
                horizon=horizon,
            )
            for first_action in candidates
        ]
        best_index = int(np.argmin(costs))
        validation_student_cost = float("nan")
        validation_selected_cost = float("nan")
        validation_improvement = float("nan")
        if validation_horizon:
            validation = mujoco.MjData(self.model)
            mujoco.mj_copyData(validation, self.model, data)
            validation_rng = np.random.default_rng(validation_seed)
            validation.qpos[self._joint_qpos] = np.clip(
                validation.qpos[self._joint_qpos]
                + validation_rng.uniform(
                    -validation_joint_noise,
                    validation_joint_noise,
                    size=self.contract.action_size,
                ),
                self._lower,
                self._upper,
            )
            validation.qvel[self._joint_dof] += validation_rng.uniform(
                -validation_joint_velocity_noise,
                validation_joint_velocity_noise,
                size=self.contract.action_size,
            )
            mujoco.mj_forward(self.model, validation)
            validation_student_cost = self._candidate_rollout_cost(
                validation,
                first_action=candidates[0],
                nominal_teacher_action=nominal_teacher_action,
                previous_action=previous_action,
                motion=motion,
                current_frame=current_frame,
                correction=correction,
                teacher_base_policy=teacher_base_policy,
                horizon=validation_horizon,
            )
            validation_selected_cost = self._candidate_rollout_cost(
                validation,
                first_action=candidates[best_index],
                nominal_teacher_action=nominal_teacher_action,
                previous_action=previous_action,
                motion=motion,
                current_frame=current_frame,
                correction=correction,
                teacher_base_policy=teacher_base_policy,
                horizon=validation_horizon,
            )
            validation_improvement = (
                validation_student_cost - validation_selected_cost
            )
        return candidates[best_index], {
            "student_cost": costs[0],
            "nominal_teacher_cost": costs[1],
            "selected_cost": costs[best_index],
            "improvement_over_student": costs[0] - costs[best_index],
            "improvement_over_nominal_teacher": costs[1] - costs[best_index],
            "validation_student_cost": validation_student_cost,
            "validation_selected_cost": validation_selected_cost,
            "validation_improvement_over_student": validation_improvement,
            "candidate_count": float(len(candidates)),
        }

    def rollout(
        self,
        motion: int,
        start_frame: int,
        correction: np.ndarray | None = None,
        *,
        capture: bool = False,
        teacher_execution_probability: float = 1.0,
        teacher_base_policy: SoccerMotionPolicy | None = None,
        state_feedback_horizon: int = 0,
        minimum_state_feedback_improvement: float = 0.0,
        maximum_state_feedback_action_delta: float = 1.0,
        state_feedback_validation_horizon: int = 0,
        minimum_state_feedback_validation_improvement: float = 0.0,
        state_feedback_validation_base_seed: int | None = None,
        validation_joint_noise: float = 0.0,
        validation_joint_velocity_noise: float = 0.0,
        reset_perturbation_base_seed: int | None = None,
        reset_joint_noise: float = 0.0,
        reset_root_velocity_noise: float = 0.0,
        reset_yaw_range: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Run one deterministic exact-state restart to the finite endpoint."""
        if not 0 <= motion < self.corpus.motion_count:
            raise ValueError("motion index lies outside the corpus")
        length = int(self.corpus.lengths[motion])
        if not 0 <= start_frame < length - 1:
            raise ValueError("start frame must leave one transition")
        if correction is None:
            correction = np.zeros((length, self.contract.action_size))
        correction = np.asarray(correction, dtype=np.float64)
        if correction.shape != (length, self.contract.action_size):
            raise ValueError("correction must have one action per finite frame")
        if not np.isfinite(correction).all():
            raise ValueError("correction contains non-finite values")
        if (
            state_feedback_horizon < 0
            or state_feedback_validation_horizon < 0
            or minimum_state_feedback_improvement < 0.0
            or minimum_state_feedback_validation_improvement < 0.0
            or maximum_state_feedback_action_delta <= 0.0
            or min(
                validation_joint_noise,
                validation_joint_velocity_noise,
                reset_joint_noise,
                reset_root_velocity_noise,
                reset_yaw_range,
            )
            < 0.0
        ):
            raise ValueError("state-feedback teacher settings are invalid")
        if state_feedback_validation_horizon and (
            not state_feedback_horizon
            or state_feedback_validation_base_seed is None
        ):
            raise ValueError(
                "state-feedback cross-validation requires search and a base seed"
            )
        if not state_feedback_validation_horizon and (
            minimum_state_feedback_validation_improvement > 0.0
            or validation_joint_noise > 0.0
            or validation_joint_velocity_noise > 0.0
            or state_feedback_validation_base_seed is not None
        ):
            raise ValueError(
                "state-feedback validation settings require a validation horizon"
            )
        if reset_perturbation_base_seed is None and any(
            value > 0.0
            for value in (
                reset_joint_noise,
                reset_root_velocity_noise,
                reset_yaw_range,
            )
        ):
            raise ValueError("reset noise requires a perturbation base seed")

        reset_perturbation = (
            deterministic_reset_perturbation(
                base_seed=reset_perturbation_base_seed,
                motion=motion,
                start_frame=start_frame,
                action_size=self.contract.action_size,
                joint_noise=reset_joint_noise,
                root_velocity_noise=reset_root_velocity_noise,
                yaw_range=reset_yaw_range,
            )
            if reset_perturbation_base_seed is not None
            else None
        )
        reset_yaw = reset_perturbation.yaw if reset_perturbation else 0.0
        reset_joint_offset = (
            reset_perturbation.joint_position_noise
            if reset_perturbation
            else np.zeros(self.contract.action_size, dtype=np.float64)
        )
        reset_velocity_offset = (
            reset_perturbation.root_velocity_noise
            if reset_perturbation
            else np.zeros(6, dtype=np.float64)
        )

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        data.qvel[:] = 0.0
        data.qpos[self._root_qpos : self._root_qpos + 2] = self._model_root_xy
        data.qpos[self._root_qpos + 2] = self.corpus.root_position[
            motion, start_frame, 2
        ]
        data.qpos[self._root_qpos + 3 : self._root_qpos + 7] = (
            yaw_quaternion_rotate(
                self.corpus.root_quaternion_wxyz[motion, start_frame],
                reset_yaw,
            )
        )
        data.qpos[self._joint_qpos] = np.clip(
            self.corpus.joint_position[motion, start_frame] + reset_joint_offset,
            self._lower,
            self._upper,
        )
        data.qvel[self._root_dof : self._root_dof + 3] = (
            yaw_vector_rotate(
                self.corpus.root_linear_velocity[motion, start_frame],
                reset_yaw,
            )
            + reset_velocity_offset[:3]
        )
        data.qvel[self._root_dof + 3 : self._root_dof + 6] = (
            yaw_vector_rotate(
                self.corpus.root_angular_velocity[motion, start_frame],
                reset_yaw,
            )
            + reset_velocity_offset[3:]
        )
        data.qvel[self._joint_dof] = self.corpus.joint_velocity[motion, start_frame]
        data.ctrl[self._tau_actuator] = 0.0
        data.ctrl[self._vel_actuator] = 0.0
        data.ctrl[self._pos_actuator] = self.corpus.joint_position[
            motion, start_frame
        ]
        mujoco.mj_forward(self.model, data)

        previous_action = np.zeros(self.contract.action_size, dtype=np.float64)
        squared_error: list[float] = []
        absolute_error_by_joint: list[np.ndarray] = []
        root_position_error: list[float] = []
        contact_match = 0
        contact_count = 0
        action_energy: list[float] = []
        correction_energy: list[float] = []
        minimum_height = float("inf")
        minimum_upright = float("inf")
        terminal_frame: int | None = None
        reason = "completed"
        observations: list[np.ndarray] = []
        base_actions: list[np.ndarray] = []
        teacher_base_actions: list[np.ndarray] = []
        teacher_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        teacher_interventions: list[bool] = []
        teacher_label_selected: list[bool] = []
        teacher_cost_improvement: list[float] = []
        teacher_validation_cost_improvement: list[float] = []
        teacher_validation_seeds: list[int] = []
        reference_frames: list[int] = []
        qpos_states: list[np.ndarray] = []

        for frame in range(start_frame + 1, length):
            current = frame - 1
            observation = soccer_motion_actor_observation(
                data,
                joint_qpos=self._joint_qpos,
                joint_dof=self._joint_dof,
                gyro_slice=self._gyro_slice,
                torso_site=self._torso_site,
                reference_joint_position=self.corpus.joint_position[motion, current],
                reference_joint_velocity=self.corpus.joint_velocity[motion, current],
                reference_root_linear_velocity=(
                    self.corpus.root_linear_velocity[motion, current]
                ),
                reference_root_angular_velocity=(
                    self.corpus.root_angular_velocity[motion, current]
                ),
                reference_contact=self.corpus.foot_contact[motion, current],
                previous_action=previous_action,
                progress=current / max(length - 1, 1),
                kick_leg_one_hot=self.corpus.kick_leg_one_hot[motion],
            )
            base_action = np.asarray(self.base_policy(observation), dtype=np.float64)
            if base_action.shape != (self.contract.action_size,):
                raise ValueError("base policy returned an incompatible action")
            teacher_base_action = np.asarray(
                (
                    teacher_base_policy(observation)
                    if teacher_base_policy is not None
                    else base_action
                ),
                dtype=np.float64,
            )
            if teacher_base_action.shape != (self.contract.action_size,):
                raise ValueError("teacher base policy returned an incompatible action")
            teacher_action = np.clip(
                teacher_base_action + correction[current], *self.contract.action_clip
            )
            cost_improvement = float("nan")
            validation_cost_improvement: float | None = None
            validation_seed: int | None = None
            label_selected = True
            if state_feedback_horizon:
                if teacher_base_policy is None:
                    raise ValueError(
                        "state-feedback teacher requires a separate teacher base"
                    )
                if state_feedback_validation_horizon:
                    validation_seed = derive_case_seed(
                        state_feedback_validation_base_seed,
                        motion,
                        start_frame,
                        current,
                    )
                teacher_action, feedback = self.short_horizon_teacher_action(
                    data,
                    motion=motion,
                    current_frame=current,
                    previous_action=previous_action,
                    student_action=base_action,
                    nominal_teacher_action=teacher_action,
                    correction=correction,
                    teacher_base_policy=teacher_base_policy,
                    horizon=state_feedback_horizon,
                    maximum_action_delta=maximum_state_feedback_action_delta,
                    validation_horizon=state_feedback_validation_horizon,
                    validation_seed=validation_seed,
                    validation_joint_noise=validation_joint_noise,
                    validation_joint_velocity_noise=(
                        validation_joint_velocity_noise
                    ),
                )
                cost_improvement = feedback["improvement_over_student"]
                validation_cost_improvement = (
                    feedback["validation_improvement_over_student"]
                    if state_feedback_validation_horizon
                    else None
                )
                label_selected = cross_fitted_label_selected(
                    base_action,
                    teacher_action,
                    search_improvement=cost_improvement,
                    minimum_search_improvement=(
                        minimum_state_feedback_improvement
                    ),
                    validation_improvement=validation_cost_improvement,
                    minimum_validation_improvement=(
                        minimum_state_feedback_validation_improvement
                    ),
                )
            action, use_teacher = select_dagger_action(
                base_action,
                teacher_action,
                teacher_probability=teacher_execution_probability,
                action_clip=self.contract.action_clip,
                rng=rng,
            )
            target = np.clip(
                self.corpus.joint_position[motion, frame]
                + self.contract.action_scale * action,
                self._lower,
                self._upper,
            )
            data.ctrl[self._pos_actuator] = target
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, data)
            previous_action = action
            error = data.qpos[self._joint_qpos] - self.corpus.joint_position[
                motion, frame
            ]
            squared_error.extend(np.square(error).tolist())
            absolute_error_by_joint.append(np.abs(error))
            desired_root = self.corpus.root_position[motion, frame].copy()
            desired_xy_displacement = np.asarray(
                [
                    *(
                        self.corpus.root_position[motion, frame, :2]
                        - self.corpus.root_position[motion, start_frame, :2]
                    ),
                    0.0,
                ],
                dtype=np.float64,
            )
            desired_root[:2] = (
                self._model_root_xy
                + yaw_vector_rotate(desired_xy_displacement, reset_yaw)[:2]
            )
            root_position_error.append(
                float(
                    np.linalg.norm(
                        data.qpos[self._root_qpos : self._root_qpos + 3]
                        - desired_root
                    )
                )
            )
            contacts = self._contacts(data)
            contact_match += int(
                np.sum(contacts == self.corpus.foot_contact[motion, frame])
            )
            contact_count += 2
            action_energy.append(float(np.mean(np.square(action))))
            correction_energy.append(float(np.mean(np.square(correction[current]))))
            rotation = data.site_xmat[self._torso_site].reshape(3, 3)
            upright = float(rotation[2, 2])
            torso_height = float(data.xpos[self._torso_body, 2])
            minimum_height = min(minimum_height, torso_height)
            minimum_upright = min(minimum_upright, upright)
            if capture:
                observations.append(observation)
                base_actions.append(base_action)
                teacher_base_actions.append(teacher_base_action)
                teacher_actions.append(teacher_action.copy())
                executed_actions.append(action.copy())
                teacher_interventions.append(use_teacher)
                teacher_label_selected.append(label_selected)
                teacher_cost_improvement.append(cost_improvement)
                teacher_validation_cost_improvement.append(
                    float("nan")
                    if validation_cost_improvement is None
                    else validation_cost_improvement
                )
                teacher_validation_seeds.append(
                    -1 if validation_seed is None else validation_seed
                )
                reference_frames.append(current)
                qpos_states.append(data.qpos.copy())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                terminal_frame = frame
                reason = "non_finite_state"
                break
            if torso_height < 0.35 or upright < 0.20:
                terminal_frame = frame
                reason = "fall"
                break

        evaluated = len(absolute_error_by_joint)
        completed = terminal_frame is None
        survival = evaluated / (length - 1 - start_frame)
        joint_rmse = float(np.sqrt(np.mean(squared_error)))
        contact_agreement = contact_match / max(contact_count, 1)
        mean_root_error = float(np.mean(root_position_error))
        mean_correction_energy = float(np.mean(correction_energy))
        score = (
            1000.0 * survival
            + 250.0 * float(completed)
            + 20.0 * contact_agreement
            + 10.0 * np.clip(minimum_upright, -1.0, 1.0)
            - 25.0 * joint_rmse
            - 10.0 * mean_root_error
            - 2.0 * mean_correction_energy
        )
        result: dict[str, Any] = {
            "motion": motion,
            "relative_path": self.corpus.relative_paths[motion],
            "start_frame": start_frame,
            "length": length,
            "evaluated_frames": evaluated,
            "completed": completed,
            "terminal_frame": terminal_frame,
            "termination_reason": reason,
            "survival_fraction": float(survival),
            "joint_tracking_rmse_rad": joint_rmse,
            "mean_joint_abs_error_by_joint": np.mean(
                np.stack(absolute_error_by_joint), axis=0
            ).tolist(),
            "mean_root_position_error_m": mean_root_error,
            "foot_contact_agreement": float(contact_agreement),
            "minimum_torso_height_m": minimum_height,
            "minimum_upright": minimum_upright,
            "mean_action_energy": float(np.mean(action_energy)),
            "mean_correction_energy": mean_correction_energy,
            "teacher_execution_probability": teacher_execution_probability,
            "teacher_intervention_fraction": (
                float(np.mean(teacher_interventions))
                if teacher_interventions
                else teacher_execution_probability
            ),
            "selected_teacher_label_fraction": (
                float(np.mean(teacher_label_selected))
                if teacher_label_selected
                else 1.0
            ),
            "reset_perturbation_seed": (
                reset_perturbation.case_seed if reset_perturbation else None
            ),
            "teacher_score": float(score),
        }
        if capture:
            result["trajectory"] = {
                "observations": np.asarray(observations, dtype=np.float32),
                "base_actions": np.asarray(base_actions, dtype=np.float32),
                "teacher_base_actions": np.asarray(
                    teacher_base_actions,
                    dtype=np.float32,
                ),
                "teacher_actions": np.asarray(teacher_actions, dtype=np.float32),
                "executed_actions": np.asarray(executed_actions, dtype=np.float32),
                "teacher_interventions": np.asarray(
                    teacher_interventions, dtype=bool
                ),
                "teacher_label_selected": np.asarray(
                    teacher_label_selected, dtype=bool
                ),
                "teacher_cost_improvement": np.asarray(
                    teacher_cost_improvement, dtype=np.float32
                ),
                "teacher_validation_cost_improvement": np.asarray(
                    teacher_validation_cost_improvement, dtype=np.float32
                ),
                "teacher_validation_seed": np.asarray(
                    teacher_validation_seeds, dtype=np.int64
                ),
                "reference_frames": np.asarray(reference_frames, dtype=np.int32),
                "qpos": np.asarray(qpos_states, dtype=np.float64),
            }
        return result
