from __future__ import annotations

import numpy as np
import pytest

from my3d_rl.soccer_motion_teacher import (
    decode_phase_correction,
    robust_teacher_objective,
    select_dagger_action,
    state_feedback_action_candidates,
)
from my3d_rl.soccer_motion_dagger import load_selected_teacher_corrections


def test_phase_correction_is_bounded_smooth_and_sparse():
    parameters = np.array([0.0, 0.2, 0.4, -0.2, 0.0, 0.2])
    phases = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    correction = decode_phase_correction(
        parameters,
        phases=phases,
        action_size=5,
        joint_indices=[1, 3],
        knot_count=3,
        maximum_abs_correction=0.5,
    )

    assert correction.shape == (5, 5)
    np.testing.assert_array_equal(correction[:, [0, 2, 4]], 0.0)
    np.testing.assert_allclose(correction[0, [1, 3]], [0.0, 0.2])
    np.testing.assert_allclose(correction[2, [1, 3]], [0.4, -0.2])
    np.testing.assert_allclose(correction[-1, [1, 3]], [0.0, 0.2])


def test_phase_correction_rejects_bound_violation():
    with pytest.raises(ValueError, match="bound"):
        decode_phase_correction(
            np.array([0.0, 0.6, 0.0, 0.0]),
            phases=np.array([0.0, 1.0]),
            action_size=3,
            joint_indices=[1, 2],
            knot_count=2,
            maximum_abs_correction=0.5,
        )


def test_robust_teacher_objective_protects_minimum():
    assert robust_teacher_objective(np.array([10.0, 20.0]), minimum_weight=0.5) == 12.5


def test_selected_teacher_loader_rejects_incomplete_manifest(tmp_path):
    manifest = tmp_path / "selection.json"
    manifest.write_text('{"status": "incomplete"}', encoding="utf-8")

    class Corpus:
        motion_count = 1

    with pytest.raises(ValueError, match="incomplete"):
        load_selected_teacher_corrections(
            manifest,
            corpus=Corpus(),  # type: ignore[arg-type]
            contract=object(),  # type: ignore[arg-type]
            contract_sha256="0" * 64,
        )


def test_dagger_action_can_query_teacher_without_executing_it():
    selected, used_teacher = select_dagger_action(
        np.array([1.2, -0.3]),
        np.array([0.4, -0.8]),
        teacher_probability=0.0,
        action_clip=(-1.0, 1.0),
    )

    np.testing.assert_allclose(selected, [1.0, -0.3])
    assert not used_teacher


def test_dagger_action_executes_teacher_at_beta_one():
    selected, used_teacher = select_dagger_action(
        np.array([0.1, 0.2]),
        np.array([0.4, -0.8]),
        teacher_probability=1.0,
        action_clip=(-1.0, 1.0),
    )

    np.testing.assert_allclose(selected, [0.4, -0.8])
    assert used_teacher


def test_state_feedback_candidates_are_bounded_and_keep_student_and_teacher():
    student = np.array([0.1, -0.2, 0.3])
    teacher = np.array([0.2, -0.1, 0.4])
    candidates = state_feedback_action_candidates(
        student,
        teacher,
        position_error=np.array([0.0, 0.5, -0.5]),
        velocity_error=np.array([0.0, 1.0, -1.0]),
        active_joint_indices=[1, 2],
        action_scale=0.35,
        control_period=0.02,
        action_clip=(-1.0, 1.0),
        maximum_delta_from_student=0.2,
    )

    assert candidates.shape == (6, 3)
    np.testing.assert_allclose(candidates[0], student)
    np.testing.assert_allclose(candidates[1], teacher)
    assert np.max(np.abs(candidates)) <= 1.0
    np.testing.assert_allclose(candidates[3:, 0], teacher[0])
    assert np.max(np.abs(candidates - student)) <= 0.2 + 1.0e-12
