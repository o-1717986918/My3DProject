from __future__ import annotations

from pathlib import Path

import pytest

from tools.assemble_soccer_motion_teacher_corpus import (
    aggregate_split,
    parse_override,
)


def _report(trials, base_survival, candidate_survival):
    summary = {
        "trials": trials,
        "baseline_completions": 1,
        "candidate_completions": 2,
        "baseline_only_completions": 0,
        "candidate_only_completions": 1,
        "baseline_mean_survival_fraction": base_survival,
        "candidate_mean_survival_fraction": candidate_survival,
        "mean_survival_fraction_delta": candidate_survival - base_survival,
        "baseline_mean_teacher_score": 500.0,
        "candidate_mean_teacher_score": 600.0,
        "mean_teacher_score_delta": 100.0,
        "survival_improvements": 2,
        "survival_regressions": 0,
    }
    return {"validation": {"summary": summary}}


def test_parse_override_requires_motion_and_path():
    assert parse_override("12=/tmp/report.json") == (12, Path("/tmp/report.json"))
    with pytest.raises(ValueError):
        parse_override("missing")


def test_aggregate_split_weights_unequal_phase_counts():
    summary = aggregate_split(
        [_report(2, 0.5, 0.7), _report(6, 0.8, 0.9)], "validation"
    )

    assert summary["trials"] == 8
    assert summary["baseline_completions"] == 2
    assert summary["candidate_completions"] == 4
    assert summary["baseline_mean_survival_fraction"] == pytest.approx(0.725)
    assert summary["candidate_mean_survival_fraction"] == pytest.approx(0.85)
    assert summary["mean_survival_fraction_delta"] == pytest.approx(0.125)
