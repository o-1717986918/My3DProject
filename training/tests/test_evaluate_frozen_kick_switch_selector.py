import numpy as np

from tools.evaluate_frozen_kick_switch_selector import (
    fixed_cycle_policy_metrics,
    paired_success_counts,
)


def test_fixed_cycle_policy_uses_only_the_predeclared_exact_cycle() -> None:
    metrics = fixed_cycle_policy_metrics(
        np.array([[0, 1, 1, 0, 1]], dtype=np.uint8),
        np.array([[0, 0, 0, 0, 1]], dtype=np.uint8),
        np.array([1, 1, 1, 2, 2]),
        np.array([1, 3, 5, 1, 3]),
        np.ones(5, dtype=bool),
        prototype_index=0,
        release_cycle=3,
    )

    assert metrics["rollouts"] == 2
    assert metrics["releases"] == 2
    assert metrics["successes"] == 2
    assert metrics["falls"] == 1
    assert [node["row"] for node in metrics["decisions"]] == [1, 4]


def test_paired_success_counts_treats_abstention_as_not_success() -> None:
    comparison = paired_success_counts(
        [
            {"rollout_id": 1, "success": True},
            {"rollout_id": 2, "success": False},
            {"rollout_id": 4, "success": True},
        ],
        [
            {"rollout_id": 1, "success": True},
            {"rollout_id": 2, "success": True},
            {"rollout_id": 3, "success": True},
        ],
    )

    assert comparison == {
        "compared_rollouts": 4,
        "both_succeed": 1,
        "selector_only_succeeds": 1,
        "baseline_only_succeeds": 2,
        "neither_succeeds": 0,
        "net_success_advantage": -1,
    }


def test_paired_success_counts_includes_rollouts_where_both_abstain() -> None:
    comparison = paired_success_counts([], [], np.array([1, 2, 2, 3]))

    assert comparison["compared_rollouts"] == 3
    assert comparison["neither_succeeds"] == 3
