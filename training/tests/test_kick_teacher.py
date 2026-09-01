import numpy as np
import pytest

from my3d_rl import load_policy_contract
from my3d_rl.kick_env import DEFAULT_CONTRACT
from my3d_rl.kick_teacher import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    KickTeacherEvaluator,
    KickTeacherSpec,
    build_joint_delta_trajectory,
    cem_optimize,
    kick_trial_success,
)


def test_cem_reproducibly_solves_bounded_quadratic():
    target = np.array([0.25, -0.5, 0.75])

    def objective(candidate: np.ndarray) -> float:
        return -float(np.sum(np.square(candidate - target)))

    kwargs = dict(
        initial_mean=np.zeros(3),
        initial_std=np.ones(3),
        lower=-np.ones(3),
        upper=np.ones(3),
        seed=17,
        population=80,
        generations=10,
    )
    first = cem_optimize(objective, **kwargs)
    second = cem_optimize(objective, **kwargs)

    np.testing.assert_allclose(first.parameters, second.parameters)
    np.testing.assert_allclose(first.parameters, target, atol=0.06)
    assert first.score > -0.01


def test_teacher_trajectory_is_bounded_smooth_and_returns_to_neutral():
    contract = load_policy_contract(DEFAULT_CONTRACT)
    parameters = 0.25 * PARAMETER_UPPER
    times = np.arange(0.0, 1.201, 0.02)
    trajectory = build_joint_delta_trajectory(parameters, contract, times)

    assert trajectory.shape == (61, contract.action_size)
    assert np.isfinite(trajectory).all()
    np.testing.assert_allclose(trajectory[0], 0.0)
    np.testing.assert_allclose(trajectory[-1], 0.0)
    assert np.max(np.abs(np.diff(trajectory, axis=0))) < 0.25
    assert len(PARAMETER_NAMES) == PARAMETER_LOWER.size == PARAMETER_UPPER.size


def test_teacher_spec_rejects_unsupported_requests():
    with pytest.raises(ValueError, match="target_angle_deg"):
        KickTeacherSpec(target_angle_deg=31.0).validate()
    with pytest.raises(ValueError, match="divisible"):
        KickTeacherSpec(control_dt_s=0.02, simulation_dt_s=0.006).validate()


def test_walk_history_requires_exact_state_and_matching_action_shape():
    contract = load_policy_contract(DEFAULT_CONTRACT)
    evaluator = KickTeacherEvaluator(KickTeacherSpec(), contract=contract)
    parameters = np.zeros(len(PARAMETER_NAMES))

    with pytest.raises(ValueError, match="exact initial state"):
        evaluator.rollout(
            parameters,
            initial_walk_previous_action=np.zeros(contract.action_size),
        )
    with pytest.raises(ValueError, match="incompatible shape"):
        evaluator.rollout(
            parameters,
            initial_qpos=np.asarray(evaluator.model.qpos0),
            initial_qvel=np.zeros(evaluator.model.nv),
            initial_walk_previous_action=np.zeros(contract.action_size - 1),
        )


def test_capture_only_stop_requires_setup_transition_capture():
    evaluator = KickTeacherEvaluator(KickTeacherSpec())
    parameters = np.zeros(len(PARAMETER_NAMES))

    with pytest.raises(ValueError, match="requires setup capture mode"):
        evaluator.rollout(parameters, stop_after_transition_capture=True)


def test_capture_only_stop_preserves_the_captured_transition_state():
    evaluator = KickTeacherEvaluator(KickTeacherSpec(evaluation_duration_s=1.2))
    parameters = np.zeros(len(PARAMETER_NAMES))
    kwargs = {
        "setup_ball_x_offset_m": 0.0,
        "setup_ball_y_offset_m": 0.0,
        "setup_confirmation_cycles": 1,
        "capture_transition_entry": True,
    }

    evaluator.rollout(parameters, **kwargs)
    complete_entry = evaluator.captured_transition_entry
    evaluator.rollout(parameters, stop_after_transition_capture=True, **kwargs)
    stopped_entry = evaluator.captured_transition_entry

    assert complete_entry is not None and stopped_entry is not None
    np.testing.assert_array_equal(stopped_entry.qpos, complete_entry.qpos)
    np.testing.assert_array_equal(stopped_entry.qvel, complete_entry.qvel)
    np.testing.assert_array_equal(
        stopped_entry.walk_previous_action, complete_entry.walk_previous_action
    )


def test_kick_trial_gate_requires_contact_range_direction_speed_and_upright():
    accepted = {
        "contact": True,
        "fell": False,
        "range_error_m": 0.2,
        "lateral_error_m": 0.1,
        "speed_error_mps": 0.3,
    }
    assert kick_trial_success(accepted)
    for rejected_field, rejected_value in (
        ("contact", False),
        ("fell", True),
        ("range_error_m", 0.51),
        ("lateral_error_m", 0.51),
        ("speed_error_mps", 1.01),
    ):
        rejected = dict(accepted)
        rejected[rejected_field] = rejected_value
        assert not kick_trial_success(rejected)
