from __future__ import annotations

import numpy as np
import pytest

from tools.render_soccer_motion_replay import frame_indices


def test_frame_indices_include_endpoints_without_duplicate_frames() -> None:
    np.testing.assert_array_equal(frame_indices(10, 3), [0, 4, 9])
    np.testing.assert_array_equal(frame_indices(2, 8), [0, 1])
    with pytest.raises(ValueError, match="positive"):
        frame_indices(0, 3)
