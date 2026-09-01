import numpy as np
import pytest

from my3d_rl.kick_transition import (
    estimate_locomotion_phase,
    estimate_locomotion_phase_jax,
)


JOINT_ORDER = (
    "AAHead_yaw",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
)


def test_neutral_phase_is_deterministic_double_support():
    phase = estimate_locomotion_phase(np.zeros(3), np.zeros(3), JOINT_ORDER)

    np.testing.assert_array_equal(phase.sin_cos, [0.0, 1.0])
    np.testing.assert_array_equal(phase.support_hint, [0.0, 1.0, 0.0])
    assert phase.magnitude_rad == 0.0


@pytest.mark.parametrize(
    ("right_hip", "expected_support"),
    [
        (0.25, [1.0, 0.0, 0.0]),
        (-0.25, [0.0, 0.0, 1.0]),
    ],
)
def test_position_signal_selects_a_deployable_support_hint(
    right_hip: float, expected_support: list[float]
):
    positions = np.array([0.0, 0.0, right_hip])
    phase = estimate_locomotion_phase(positions, np.zeros(3), JOINT_ORDER)

    assert np.isclose(np.linalg.norm(phase.sin_cos), 1.0)
    np.testing.assert_array_equal(phase.support_hint, expected_support)


def test_velocity_signal_produces_quadrature_phase_and_double_support():
    velocities = np.array([0.0, -1.0, 1.0])
    phase = estimate_locomotion_phase(np.zeros(3), velocities, JOINT_ORDER)

    np.testing.assert_allclose(phase.sin_cos, [0.0, 1.0], atol=1.0e-7)
    np.testing.assert_array_equal(phase.support_hint, [0.0, 1.0, 0.0])


def test_phase_rejects_non_finite_or_mismatched_joint_state():
    with pytest.raises(ValueError, match="match"):
        estimate_locomotion_phase(np.zeros(2), np.zeros(3), JOINT_ORDER)
    with pytest.raises(ValueError, match="finite"):
        estimate_locomotion_phase(
            np.array([0.0, np.nan, 0.0]), np.zeros(3), JOINT_ORDER
        )


def test_jax_phase_estimator_matches_exact_numpy_runtime_formula():
    positions = np.array([0.0, -0.12, 0.21], dtype=np.float32)
    velocities = np.array([0.0, 0.4, -0.7], dtype=np.float32)
    expected = estimate_locomotion_phase(positions, velocities, JOINT_ORDER)

    phase, support, magnitude = estimate_locomotion_phase_jax(
        positions, velocities, 1, 2
    )

    np.testing.assert_allclose(phase, expected.sin_cos, atol=1.0e-6)
    np.testing.assert_array_equal(support, expected.support_hint)
    assert float(magnitude) == pytest.approx(expected.magnitude_rad, abs=1.0e-6)
