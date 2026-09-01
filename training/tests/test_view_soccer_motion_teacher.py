from __future__ import annotations

import numpy as np
import pytest

from tools.view_soccer_motion_teacher import select_episode


def test_select_episode_filters_split_and_start_frame():
    dataset = {
        "split": np.array([0, 0, 1, 1, 1]),
        "start_frame": np.array([0, 0, 5, 5, 9]),
        "qpos": np.arange(15, dtype=np.float64).reshape(5, 3),
    }

    trajectory, start = select_episode(dataset, split=1, start_frame=5)

    assert start == 5
    np.testing.assert_array_equal(trajectory, dataset["qpos"][2:4])


def test_select_episode_rejects_unavailable_start():
    dataset = {
        "split": np.array([0]),
        "start_frame": np.array([0]),
        "qpos": np.zeros((1, 3)),
    }
    with pytest.raises(ValueError, match="no split"):
        select_episode(dataset, split=1, start_frame=None)
