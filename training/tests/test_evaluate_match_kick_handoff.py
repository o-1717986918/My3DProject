from __future__ import annotations

import numpy as np
import pytest

from tools.evaluate_match_kick_handoff import representative_rows


def test_representative_rows_never_counts_correlated_frames_twice() -> None:
    ids = np.array([1, 1, 1, 2, 2, 3])
    ball = np.array(
        [
            [0.60, 0.00, -0.5],
            [0.35, 0.01, -0.5],
            [0.30, 0.10, -0.5],
            [0.40, 0.02, -0.5],
            [0.34, 0.01, -0.5],
            [0.34, 0.01, -0.5],
        ]
    )
    eligible = np.array([True, True, True, True, True, False])

    np.testing.assert_array_equal(
        representative_rows(ids, ball, eligible, max_rollouts=2), [1, 4]
    )
    with pytest.raises(ValueError, match="release-like"):
        representative_rows(ids, ball, np.zeros(6, dtype=bool), max_rollouts=2)
