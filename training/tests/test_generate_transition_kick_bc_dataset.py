import numpy as np
import pytest

from tools.generate_transition_kick_bc_dataset import expand_episode_split


def test_expand_episode_split_keeps_complete_episodes_together() -> None:
    episode_ids = np.array([4, 4, 4, 8, 8, 9])

    split = expand_episode_split(episode_ids, {4: 0, 8: 1, 9: 0})

    np.testing.assert_array_equal(split, [0, 0, 0, 1, 1, 0])


def test_expand_episode_split_rejects_missing_episode() -> None:
    with pytest.raises(ValueError, match="missing IDs"):
        expand_episode_split(np.array([1, 2]), {1: 0})
