import numpy as np
import pytest

from tools.generate_kick_switch_window_corpus import (
    rollout_group_split,
    successful_rollout_coverage,
)


def test_rollout_group_split_is_deterministic_and_has_no_frame_leakage() -> None:
    rollout_ids = np.repeat(np.arange(10), [2, 3, 4, 2, 5, 3, 2, 4, 3, 2])
    first = rollout_group_split(rollout_ids, seed=19, validation_fraction=0.2)
    second = rollout_group_split(rollout_ids, seed=19, validation_fraction=0.2)

    np.testing.assert_array_equal(first, second)
    assert set(rollout_ids[first == 0]).isdisjoint(set(rollout_ids[first == 1]))
    assert len(set(rollout_ids[first == 1])) == 2


def test_successful_rollout_coverage_counts_windows_not_frames() -> None:
    rollout_ids = np.array([0, 0, 1, 1, 1, 2])
    success = np.array([0, 1, 0, 0, 0, 1])

    assert successful_rollout_coverage(rollout_ids, success) == (2, 3)


@pytest.mark.parametrize(
    "rollout_ids,success",
    [
        (np.array([0]), np.array([[1]])),
        (np.array([0, 1]), np.array([1])),
    ],
)
def test_successful_rollout_coverage_rejects_misaligned_rows(
    rollout_ids: np.ndarray, success: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="aligned vectors"):
        successful_rollout_coverage(rollout_ids, success)
