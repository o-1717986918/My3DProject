import numpy as np

from my3d_rl.reference_dynamics import circular_interpolate, circular_smooth


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
