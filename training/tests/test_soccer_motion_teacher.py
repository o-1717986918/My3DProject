from __future__ import annotations

import numpy as np
import pytest

from my3d_rl.soccer_motion_teacher import (
    decode_phase_correction,
    robust_teacher_objective,
)


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
