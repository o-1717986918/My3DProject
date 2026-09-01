from __future__ import annotations

import pytest

from tools.optimize_soccer_motion_teacher import (
    fixed_phase_starts,
    paired_summary,
    parse_frame_list,
)


def _result(start, survival, completed, score):
    return {
        "start_frame": start,
        "survival_fraction": survival,
        "completed": completed,
        "teacher_score": score,
    }


def test_frame_parsing_and_fixed_grid_are_finite():
    assert parse_frame_list("0, 10,20") == [0, 10, 20]
    assert fixed_phase_starts(101, samples=3, minimum_remaining_frames=10) == [0, 45, 91]
    with pytest.raises(ValueError):
        parse_frame_list("1,1")


def test_paired_summary_counts_improvements_and_regressions():
    baseline = [_result(0, 0.5, False, 500), _result(10, 1.0, True, 1200)]
    candidate = [_result(0, 0.8, False, 800), _result(10, 0.9, False, 900)]

    summary = paired_summary(baseline, candidate)

    assert summary["survival_improvements"] == 1
    assert summary["survival_regressions"] == 1
    assert summary["baseline_only_completions"] == 1
    assert summary["candidate_only_completions"] == 0
