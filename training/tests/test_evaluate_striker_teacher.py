import numpy as np

from tools.evaluate_striker_teacher import (
    _apply_optional_trigger_threshold,
    _first_event,
)


def test_first_event_returns_horizon_when_event_never_occurs():
    events = np.array(
        [
            [False, False, True],
            [True, False, False],
            [False, False, False],
        ]
    )

    assert _first_event(events, 3).tolist() == [1, 3, 0]


def test_optional_trigger_threshold_does_not_replace_the_stage_default():
    stage = {"kick_trigger_threshold": 0.8, "episode_length": 1000}

    assert _apply_optional_trigger_threshold(stage, None) == stage
    assert _apply_optional_trigger_threshold(stage, 0.9) == {
        "kick_trigger_threshold": 0.9,
        "episode_length": 1000,
    }
