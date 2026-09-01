import numpy as np
import pytest

from training.tools.optimize_phase_kick_teacher import (
    representative_indices,
    robust_phase_objective,
)


def test_representative_indices_are_deterministic_and_span_sorted_rows():
    ids = np.array([50, 10, 40, 20, 30])

    selected = representative_indices(ids, 3)

    assert ids[selected].tolist() == [10, 30, 50]


def test_representative_indices_keep_all_small_bucket_rows():
    ids = np.array([9, 3])
    selected = representative_indices(ids, 3)
    assert ids[selected].tolist() == [3, 9]


def test_robust_phase_objective_rewards_gate_and_worst_case():
    baseline = robust_phase_objective(
        np.array([1.0, 1.0]),
        np.array([False, False]),
        success_bonus=50.0,
        minimum_weight=0.5,
    )
    one_success = robust_phase_objective(
        np.array([1.0, 1.0]),
        np.array([True, False]),
        success_bonus=50.0,
        minimum_weight=0.5,
    )
    all_success = robust_phase_objective(
        np.array([1.0, 1.0]),
        np.array([True, True]),
        success_bonus=50.0,
        minimum_weight=0.5,
    )

    assert baseline < one_success < all_success


def test_robust_phase_objective_rejects_invalid_weights():
    with pytest.raises(ValueError, match="weights"):
        robust_phase_objective(
            np.array([1.0]),
            np.array([True]),
            success_bonus=-1.0,
            minimum_weight=0.5,
        )
