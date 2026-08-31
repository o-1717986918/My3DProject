import numpy as np

from my3d_rl.reference_dynamics import (
    circular_interpolate,
    circular_smooth,
    failure_phase_sampling_weights,
)


def test_circular_interpolation_wraps_without_a_seam_discontinuity():
    values = np.array([[0.0], [1.0], [2.0], [3.0]])

    np.testing.assert_allclose(circular_interpolate(values, 0.125), [0.5])
    np.testing.assert_allclose(circular_interpolate(values, 0.875), [1.5])
    np.testing.assert_allclose(circular_interpolate(values, 1.125), [0.5])


def test_circular_smoothing_preserves_constant_and_rejects_bad_pass_count():
    constant = np.full((8, 3), 2.5)
    np.testing.assert_allclose(circular_smooth(constant, 4), constant)

    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        circular_smooth(constant, -1)


def test_failure_phase_weights_focus_before_failure_and_keep_uniform_support():
    weights = failure_phase_sampling_weights(
        np.array([0.5, 0.5, 0.5]),
        bin_count=8,
        kernel_size=3,
        kernel_decay=0.8,
        uniform_ratio=0.1,
    )

    np.testing.assert_allclose(np.sum(weights), 1.0)
    assert np.all(weights > 0.0)
    assert np.argmax(weights) == 4
    assert weights[3] > weights[2] > weights[1]
