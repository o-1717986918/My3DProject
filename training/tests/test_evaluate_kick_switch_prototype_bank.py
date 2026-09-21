import numpy as np

from tools.evaluate_kick_switch_prototype_bank import (
    greedy_rollout_cover,
    greedy_utility_bank,
    rollout_coverage,
)


def test_greedy_rollout_cover_uses_only_eligible_whole_rollouts() -> None:
    rollout_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    train = rollout_ids < 3
    success = np.array(
        [
            [1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ]
    )
    fall = np.zeros_like(success)

    selected = greedy_rollout_cover(
        success, fall, rollout_ids, train, maximum_prototypes=2
    )

    assert selected == [0, 1]
    assert rollout_coverage(success, rollout_ids, train, selected) == (3, 3)
    assert rollout_coverage(success, rollout_ids, ~train, selected) == (1, 1)


def test_greedy_rollout_cover_prefers_safer_prototype_on_equal_gain() -> None:
    rollout_ids = np.array([0, 1])
    success = np.ones((2, 2), dtype=np.uint8)
    fall = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    selected = greedy_rollout_cover(
        success,
        fall,
        rollout_ids,
        np.ones(2, dtype=bool),
        maximum_prototypes=1,
    )

    assert selected == [1]


def test_greedy_utility_bank_keeps_falls_as_a_finite_cost() -> None:
    rollout_ids = np.array([0, 1, 2])
    score = np.array(
        [
            [3.0, 3.0, 3.0],
            [4.0, 2.0, 4.0],
            [2.0, 5.0, 2.0],
        ]
    )
    fall = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    selected = greedy_utility_bank(
        score,
        fall,
        rollout_ids,
        np.ones(3, dtype=bool),
        maximum_prototypes=2,
    )

    # Prototype 1 has the best average despite one finite-cost fall.  The
    # second action then improves the remaining weak state.
    assert selected == [1, 2]


def test_greedy_utility_bank_uses_falls_to_break_equal_score_ties() -> None:
    rollout_ids = np.array([0, 1])
    score = np.ones((2, 2), dtype=np.float32)
    fall = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    selected = greedy_utility_bank(
        score,
        fall,
        rollout_ids,
        np.ones(2, dtype=bool),
        maximum_prototypes=1,
    )

    assert selected == [1]
