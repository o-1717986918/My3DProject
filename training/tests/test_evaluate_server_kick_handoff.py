from __future__ import annotations

import numpy as np

from tools.evaluate_server_kick_handoff import (
    ball_local_xy,
    nearest_teacher_records,
    normalize_teacher_records,
    representative_approaches,
)


def test_representative_approaches_deduplicates_adjacent_frames() -> None:
    # Four release-like frames: two from one approach, one after a time gap,
    # and one from another match. Invalid/stale ball rows cannot enter.
    arrays = {
        "self_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (6, 1)),
        "self_xyz": np.tile([0.0, 0.0, 0.8], (6, 1)),
        "ball_xyz": np.tile([0.34, 0.01, 0.11], (6, 1)),
        "ball_valid": np.array([1, 1, 1, 0, 1, 1]),
        "ball_position_age_s": np.array([0.02, 0.02, 0.02, 0.02, 0.2, 0.02]),
        "match_id": np.array([0, 0, 0, 0, 0, 1]),
        "player_number": np.array([7, 7, 7, 7, 7, 7]),
        "time_s": np.array([1.0, 1.02, 2.0, 2.02, 2.04, 3.0]),
    }

    np.testing.assert_allclose(ball_local_xy(arrays), [[0.34, 0.01]] * 6)
    rows, candidate_frames = representative_approaches(arrays)

    assert candidate_frames == 4
    np.testing.assert_array_equal(rows, [0, 2, 5])


def test_representative_approaches_rejects_turned_away() -> None:
    arrays = {
        "self_quat_wxyz": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "self_xyz": np.array([[0.0, 0.0, 0.8]]),
        "ball_xyz": np.array([[0.34, 0.01, 0.11]]),
        "ball_valid": np.array([1]),
        "ball_position_age_s": np.array([0.0]),
        "match_id": np.array([0]),
        "player_number": np.array([7]),
        "time_s": np.array([1.0]),
    }

    rows, candidate_frames = representative_approaches(arrays)

    assert candidate_frames == 0
    assert rows.size == 0


def test_nearest_teacher_bank_uses_geometry_not_outcome() -> None:
    records = [
        {"condition_index": 2, "ball_x_offset_m": -0.05, "ball_y_offset_m": 0.04,
         "score": 100.0},
        {"condition_index": 1, "ball_x_offset_m": 0.02, "ball_y_offset_m": 0.0,
         "score": -100.0},
        {"condition_index": 3, "ball_x_offset_m": 0.02, "ball_y_offset_m": 0.0,
         "score": 100.0},
    ]

    chosen = nearest_teacher_records(records, np.array([0.34, 0.0]), 2)

    assert [record["condition_index"] for record in chosen] == [1, 3]


def test_normalize_single_teacher_manifest() -> None:
    records = normalize_teacher_records(
        {
            "purpose": "r1_low_dimensional_kick_teacher",
            "spec": {
                "target_distance_m": 3.5,
                "target_angle_deg": 0.0,
                "requested_ball_speed_mps": 2.2,
                "desired_arrival_speed_mps": 0.8,
                "action_mode": "pass",
            },
            "ball_offset_m": {"x": -0.01, "y": -0.04},
            "parameters": [0.1] * 14,
            "metrics": {"contact": True},
        }
    )

    assert len(records) == 1
    assert records[0]["condition_index"] == 0
    assert records[0]["distance_m"] == 3.5
    assert records[0]["ball_y_offset_m"] == -0.04
    assert records[0]["accepted"] is True
