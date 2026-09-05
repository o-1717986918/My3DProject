from __future__ import annotations

import pytest

from tools.evaluate_run_command_suite import soccer_command_suite, summarize_reports


def _report(name: str, passed: bool, completion: float, planar: float, yaw: float):
    return {
        "suite_command_name": name,
        "soccer_command_gate_passed": passed,
        "upright_completion_rate": completion,
        "planar_velocity_tracking_rmse": {"median_m_s": planar},
        "yaw_rate_tracking_rmse": {"median_rad_s": yaw},
    }


def test_soccer_command_suite_covers_stationary_and_bidirectional_axes():
    commands = dict(soccer_command_suite())

    assert commands["stand"] == (0.0, 0.0, 0.0)
    assert commands["fast_forward"][0] >= 1.5
    assert commands["reverse"][0] < 0.0
    assert commands["pure_left_strafe"] == (0.0, 0.30, 0.0)
    assert commands["pure_right_strafe"] == (0.0, -0.30, 0.0)
    assert commands["pure_left_turn"] == (0.0, 0.0, 0.75)
    assert commands["pure_right_turn"] == (0.0, 0.0, -0.75)
    assert commands["curve_left"][0] > 0.0
    assert commands["curve_left"][2] > 0.0
    assert commands["curve_right"][2] < 0.0


def test_suite_summary_preserves_worst_case_and_failed_names():
    summary = summarize_reports(
        [
            _report("stand", True, 1.0, 0.1, 0.1),
            _report("left_turn", False, 0.75, 0.5, 0.6),
        ]
    )

    assert summary["command_count"] == 2
    assert summary["passed_command_count"] == 1
    assert summary["all_commands_passed"] is False
    assert summary["minimum_upright_completion_rate"] == 0.75
    assert summary["maximum_median_planar_velocity_rmse_m_s"] == 0.5
    assert summary["maximum_median_yaw_rate_rmse_rad_s"] == 0.6
    assert summary["failed_commands"] == ["left_turn"]


def test_suite_summary_rejects_missing_or_duplicate_reports():
    with pytest.raises(ValueError, match="at least one"):
        summarize_reports([])
    duplicate = _report("stand", True, 1.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="repeats"):
        summarize_reports([duplicate, duplicate])
