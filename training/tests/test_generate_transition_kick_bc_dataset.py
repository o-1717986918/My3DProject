import numpy as np
import pytest

from tools.generate_transition_kick_bc_dataset import (
    _accepted_condition,
    expand_episode_split,
)


def test_expand_episode_split_keeps_complete_episodes_together() -> None:
    episode_ids = np.array([4, 4, 4, 8, 8, 9])

    split = expand_episode_split(episode_ids, {4: 0, 8: 1, 9: 0})

    np.testing.assert_array_equal(split, [0, 0, 0, 1, 1, 0])


def test_expand_episode_split_rejects_missing_episode() -> None:
    with pytest.raises(ValueError, match="missing IDs"):
        expand_episode_split(np.array([1, 2]), {1: 0})


def test_accepted_condition_normalizes_single_teacher_manifest() -> None:
    record = _accepted_condition(
        {
            "purpose": "r1_low_dimensional_kick_teacher",
            "spec": {
                "target_distance_m": 5.0,
                "target_angle_deg": 0.0,
                "requested_ball_speed_mps": 3.0,
                "desired_arrival_speed_mps": 0.8,
                "action_mode": "pass",
            },
            "ball_offset_m": {"x": 0.0, "y": 0.0},
            "parameters": [0.0] * 14,
            "metrics": {"contact": True},
        },
        60,
    )

    assert record["condition_index"] == 0
    assert record["distance_m"] == 5.0
