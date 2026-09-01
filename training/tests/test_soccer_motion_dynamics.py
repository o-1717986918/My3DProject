from __future__ import annotations

import numpy as np

from my3d_rl.soccer_motion_dynamics import quaternion_distance_rad


def test_quaternion_distance_handles_sign_equivalence():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quaternion_distance_rad(identity, -identity), 0.0)


def test_quaternion_distance_returns_shortest_rotation():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    quarter_turn = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(
        quaternion_distance_rad(identity, quarter_turn), np.pi / 2
    )
