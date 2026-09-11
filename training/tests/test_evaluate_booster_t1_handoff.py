import numpy as np
import pytest

from tools.evaluate_booster_t1_handoff import (
    _initial_phase,
    _initial_previous_action,
    _waypoint_command,
)


def test_waypoint_command_uses_body_frame_and_clips() -> None:
    command = _waypoint_command(
        np.array([1.0, 2.0]),
        np.pi / 2.0,
        np.array([3.0, 2.0]),
    )

    np.testing.assert_allclose(command, [0.0, -0.5, -np.pi / 10.0], atol=1e-6)


def test_initial_phase_modes_are_explicit() -> None:
    assert _initial_phase(13, "zero", 0.02) == 0.0
    assert _initial_phase(63, "source-step", 0.02) == pytest.approx(0.26)
    with pytest.raises(ValueError, match="unsupported phase mode"):
        _initial_phase(0, "unknown", 0.02)


def test_previous_action_zero_matches_official_session_initialization() -> None:
    joint_position = np.linspace(-2.0, 2.0, 23)
    default_pose = np.zeros(23)

    np.testing.assert_array_equal(
        _initial_previous_action(joint_position, default_pose, "zero"),
        np.zeros(12, dtype=np.float32),
    )
    np.testing.assert_allclose(
        _initial_previous_action(joint_position, default_pose, "pose-offset"),
        np.clip(joint_position[11:], -1.0, 1.0),
    )
