from __future__ import annotations

from tools.collect_server_kick_dagger import select_successful_teacher


def test_select_successful_teacher_ignores_failure_and_fall() -> None:
    candidates = [
        {"success": False, "fell": False, "metrics": {"score": 100.0}},
        {"success": True, "fell": True, "metrics": {"score": 90.0}},
        {"success": True, "fell": False, "metrics": {"score": 2.0}},
        {"success": True, "fell": False, "metrics": {"score": 3.0}},
    ]

    assert select_successful_teacher(candidates) is candidates[3]


def test_select_successful_teacher_returns_none_without_valid_success() -> None:
    candidates = [
        {"success": False, "fell": False, "metrics": {"score": 1.0}},
        {"success": True, "fell": True, "metrics": {"score": 2.0}},
    ]

    assert select_successful_teacher(candidates) is None
