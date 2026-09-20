from __future__ import annotations

import numpy as np
import pytest

from tools.collect_match_kick_handoff import (
    choose_windows,
    grouped_split,
    release_like_geometry,
)


def test_choose_windows_uses_only_complete_near_ball_starts() -> None:
    expanded = [
        (
            "player-a",
            np.zeros((6, 3), dtype=np.float32),
            np.zeros((6, 2), dtype=np.float32),
            np.array([False, True, False, False, True, True]),
        ),
        (
            "player-b",
            np.zeros((5, 3), dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
            np.array([False, False, True, False, False]),
        ),
    ]

    windows = choose_windows(expanded, episodes=4, steps=3, seed=19)

    assert len(windows) == 4
    assert {trace_id for trace_id, _ in windows} == {0, 1}
    assert all(expanded[trace_id][3][start] for trace_id, start in windows)
    assert all(start + 3 <= expanded[trace_id][1].shape[0] for trace_id, start in windows)


def test_choose_windows_rejects_absent_near_ball() -> None:
    expanded = [
        (
            "player-a",
            np.zeros((2, 3)),
            np.zeros((2, 2)),
            np.array([False, True]),
        )
    ]
    with pytest.raises(ValueError, match="near-ball"):
        choose_windows(expanded, episodes=1, steps=3, seed=0)


def test_grouped_split_keeps_each_source_trace_intact() -> None:
    ids = np.array([7, 7, 7, 8, 8, 9, 9, 9, 10, 10])

    split = grouped_split(ids, seed=11)

    assert np.any(split == 0) and np.any(split == 1)
    assert all(np.unique(split[ids == trace_id]).size == 1 for trace_id in np.unique(ids))
    np.testing.assert_array_equal(grouped_split(ids, seed=11), split)
    np.testing.assert_array_equal(grouped_split(np.array([7, 7]), seed=11), [0, 0])


def test_release_like_geometry_is_a_diagnostic_not_a_release_gate() -> None:
    positions = np.array(
        [
            [0.20, 0.25, -0.50],
            [0.65, -0.25, -0.50],
            [0.70, 0.0, -0.50],
            [0.40, 0.26, -0.50],
        ]
    )

    np.testing.assert_array_equal(
        release_like_geometry(positions), [True, True, False, False]
    )
    with pytest.raises(ValueError, match="shape"):
        release_like_geometry(np.zeros((2, 2)))
