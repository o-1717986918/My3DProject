from __future__ import annotations

import numpy as np
import pytest

from tools.evaluate_server_motion_parity import eligible_pairs, quaternion_angle_deg


def test_rounded_quaternion_does_not_create_phantom_orientation_error() -> None:
    true = np.array([0.9238795325, 0.0, 0.0, 0.3826834324])
    rounded = np.round(true, 3)

    assert quaternion_angle_deg(true, true * 0.999) < 1e-5
    assert quaternion_angle_deg(true, rounded) < 0.1
    assert quaternion_angle_deg(true, -true) < 1e-5
    with pytest.raises(ValueError, match="quaternion"):
        quaternion_angle_deg(true, np.zeros(4))


def test_eligible_pairs_excludes_gap_teleport_and_partial_controls() -> None:
    arrays = {
        "time_s": np.array([1.0, 1.02, 1.04, 1.20, 1.22, 1.24]),
        "player_number": np.ones(6, dtype=np.int32),
        "match_id": np.zeros(6, dtype=np.int32),
        "self_xyz": np.array([
            [0.0, 0.0, 0.8], [0.02, 0.0, 0.8], [0.04, 0.0, 0.8],
            [5.0, 0.0, 0.8], [5.02, 0.0, 0.8], [5.04, 0.0, 0.8],
        ]),
        "target_mask": np.ones((6, 23), dtype=np.uint8),
    }
    arrays["target_mask"][4, 2] = 0

    np.testing.assert_array_equal(eligible_pairs(arrays), [0, 1, 3])
