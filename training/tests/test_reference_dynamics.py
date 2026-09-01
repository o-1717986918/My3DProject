import numpy as np

from my3d_rl.reference_dynamics import (
    circular_interpolate,
    circular_smooth,
    failure_phase_sampling_weights,
    nonperiodic_failure_frame_sampling_weights,
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


def test_nonperiodic_failure_sampling_does_not_wrap_clip_boundary():
    weights = nonperiodic_failure_frame_sampling_weights(
        np.array([0, 5]), frame_count=8, kernel_size=3, uniform_ratio=0.1
    )

    np.testing.assert_allclose(np.sum(weights), 1.0)
    np.testing.assert_allclose(weights[7], 0.1 / 8)
    assert weights[5] > weights[4] > weights[3] > weights[7]


def test_nonperiodic_failure_sampling_is_uniform_without_failures():
    weights = nonperiodic_failure_frame_sampling_weights(
        np.array([], dtype=np.int64), frame_count=4
    )
    np.testing.assert_allclose(weights, np.full(4, 0.25))


def test_nonperiodic_failure_sampling_can_leave_recovery_lead_time():
    weights = nonperiodic_failure_frame_sampling_weights(
        np.array([8]),
        frame_count=10,
        kernel_size=3,
        lead_frames=2,
        uniform_ratio=0.1,
    )
    uniform = 0.1 / 10
    np.testing.assert_allclose(weights[8], uniform)
    np.testing.assert_allclose(weights[7], uniform)
    assert weights[6] > weights[5] > weights[4] > uniform
