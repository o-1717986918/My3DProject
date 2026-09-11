import numpy as np

from tools.evaluate_striker_teacher import (
    _apply_optional_trigger_threshold,
    _first_event,
    _setup_ready_mask,
    _stage_success_mask,
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


def test_chase_stages_use_setup_arrival_while_kick_stage_uses_ball_result():
    setup = np.array([True, False, True])
    ball_result = np.array([False, True, False])

    assert _stage_success_mask("ball_chase", setup, ball_result).tolist() == [
        True,
        False,
        True,
    ]
    assert _stage_success_mask(
        "ball_reposition", setup, ball_result
    ).tolist() == [True, False, True]
    assert _stage_success_mask(
        "walk_clone_pre_kick", setup, ball_result
    ).tolist() == [True, False, True]
    assert _stage_success_mask(
        "walk_clone_ball_chase", setup, ball_result
    ).tolist() == [True, False, True]
    assert _stage_success_mask(
        "directional_kick", setup, ball_result
    ).tolist() == [False, True, False]


def test_setup_requires_stable_dwell_before_any_ball_contact():
    settled_steps = np.array([4, 5, 8])
    contacted = np.array([False, False, True])

    assert _setup_ready_mask(
        settled_steps, contacted, confirmation_steps=5
    ).tolist() == [False, True, False]
