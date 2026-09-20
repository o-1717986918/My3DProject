from __future__ import annotations

import numpy as np
import pytest

from tools.evaluate_onnx_run import select_server_handoff_rows


def _corpus() -> dict[str, np.ndarray]:
    count = 400
    xyz = np.zeros((count, 3), dtype=np.float32)
    xyz[:, 2] = 0.72
    ball = xyz.copy()
    ball[200:, 0] = 10.0
    velocity = np.zeros((count, 3), dtype=np.float32)
    velocity[:, 0] = 0.4
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32)
    split = np.zeros(count, dtype=np.uint8)
    split[200:] = 1
    return {
        "match_id": np.zeros(count, dtype=np.int32),
        "player_number": np.full(count, 2, dtype=np.int32),
        "time_s": np.arange(count, dtype=np.float64) * 0.02,
        "motion": np.full(count, "Walk", dtype="U64"),
        "self_xyz": xyz,
        "self_velocity_body": velocity,
        "self_quat_wxyz": quaternion,
        "gyro_deg_s": np.zeros((count, 3), dtype=np.float32),
        "target_mask": np.ones((count, 23), dtype=np.uint8),
        "ball_valid": np.ones(count, dtype=np.int32),
        "ball_xyz": ball,
        "ball_position_age_s": np.zeros(count, dtype=np.float64),
        "split": split,
    }


def test_server_handoff_rows_are_balanced_separated_and_deterministic():
    arrays = _corpus()
    rows = select_server_handoff_rows(
        arrays, episodes=4, seed=7, min_speed_m_s=0.2
    )
    repeat = select_server_handoff_rows(
        arrays, episodes=4, seed=7, min_speed_m_s=0.2
    )
    assert rows.tolist() == repeat.tolist()
    assert len(set((arrays["time_s"][rows] // 2.0).astype(int))) == 4
    assert int(np.sum(rows < 200)) == 2
    assert int(np.sum(rows >= 200)) == 2


def test_server_handoff_excludes_non_walk_and_too_slow_entries():
    arrays = _corpus()
    arrays["motion"][:] = "GetUp"
    with pytest.raises(ValueError, match="no eligible"):
        select_server_handoff_rows(
            arrays, episodes=1, seed=7, min_speed_m_s=0.2
        )
    arrays["motion"][:] = "Walk"
    with pytest.raises(ValueError, match="not enough"):
        select_server_handoff_rows(
            arrays, episodes=5, seed=7, min_speed_m_s=0.2
        )


def test_server_handoff_can_probe_held_out_forward_entries_only():
    arrays = _corpus()
    arrays["self_velocity_body"][200:, 0] = 0.7
    rows = select_server_handoff_rows(
        arrays, episodes=2, seed=7, min_speed_m_s=0.2,
        validation_only=True, forward_entry_only=True,
    )
    assert np.all(rows >= 200)
    assert len(set((arrays["time_s"][rows] // 2.0).astype(int))) == 2
