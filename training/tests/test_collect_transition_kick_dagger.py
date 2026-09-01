import numpy as np
import pytest

from tools.collect_transition_kick_dagger import training_episode_ids


def test_training_episode_ids_returns_only_complete_training_groups() -> None:
    assert training_episode_ids(
        np.array([1, 1, 2, 2, 3]), np.array([0, 0, 1, 1, 0])
    ) == (1, 3)


def test_training_episode_ids_rejects_partition_leakage() -> None:
    with pytest.raises(ValueError, match="leaks an episode"):
        training_episode_ids(np.array([1, 1, 2]), np.array([0, 1, 0]))
