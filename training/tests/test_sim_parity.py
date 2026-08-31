import numpy as np

from my3d_rl.sim_parity import (
    ParityThresholds,
    generate_action_sequence,
    quaternion_angle_error,
    step_errors,
    summarize_trace,
)


def _snapshot(*, root_x=0.0, target_offset=0.0, contact=(True, True)):
    return {
        "joint_target_rad": (np.zeros(23) + target_offset).tolist(),
        "joint_position_rad": np.zeros(23).tolist(),
        "root_position_m": [root_x, 0.0, 0.65],
        "root_yaw_rad": 0.0,
        "torso_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "foot_lowest_height_m": [0.0, 0.0],
        "contact_proxy": list(contact),
    }


def test_quaternion_angle_error_is_sign_invariant():
    assert np.isclose(
        quaternion_angle_error([1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]),
        0.0,
    )
    assert np.isclose(
        quaternion_angle_error(
            [1.0, 0.0, 0.0, 0.0],
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        ),
        np.pi / 2,
    )


def test_action_sequence_is_deterministic_and_bounded():
    first = generate_action_sequence(
        pattern="random", steps=5, action_size=23, amplitude=0.2, seed=17
    )
    second = generate_action_sequence(
        pattern="random", steps=5, action_size=23, amplitude=0.2, seed=17
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 23)
    assert np.max(np.abs(first)) <= 0.2


def test_trace_summary_locates_first_threshold_crossing():
    cpu = _snapshot()
    close = _snapshot(root_x=0.01)
    far = _snapshot(root_x=0.04, contact=(False, True))
    trace = [
        {"step": 0, "errors": step_errors(cpu, close)},
        {"step": 1, "errors": step_errors(cpu, far)},
    ]
    summary = summarize_trace(trace, ParityThresholds(root_position_norm_m=0.03))

    assert summary["first_divergence_step"]["root_position_norm_m"] == 1
    assert summary["contact_proxy_mismatch_frames"] == 1
    assert not summary["parity_gate_passed"]
