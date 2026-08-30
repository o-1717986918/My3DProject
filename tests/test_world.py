from types import SimpleNamespace

import numpy as np

from mujococodebase.world.world import World


def make_world() -> World:
    return World(
        agent=SimpleNamespace(),
        team_name="TestTeam",
        number=7,
        field_name="my_field",
    )


def test_goals_are_canonical_for_both_sides():
    world = make_world()
    world.is_left_team = True
    assert world.field.get_our_goal_position() == (-27.5, 0)
    assert world.field.get_their_goal_position() == (27.5, 0)

    world.is_left_team = False
    assert world.field.get_our_goal_position() == (-27.5, 0)
    assert world.field.get_their_goal_position() == (27.5, 0)


def test_right_team_pose_is_converted_back_to_simulator_frame():
    world = make_world()
    world.is_left_team = False
    position, yaw = world.to_simulator_pose(np.array([-4.0, 2.0]), 15.0)
    np.testing.assert_allclose(position, [4.0, -2.0])
    assert yaw == -165.0


def test_ball_freshness_has_an_explicit_expiry():
    world = make_world()
    world.server_time = 10.0
    assert not world.is_ball_fresh()
    world.ball_last_seen_time = 9.7
    assert world.is_ball_fresh(max_age=0.5)
    world.ball_last_seen_time = 9.0
    assert not world.is_ball_fresh(max_age=0.5)


def test_landmark_lookup_uses_the_landmark_store():
    world = make_world()
    expected = np.array([1.0, 2.0, 3.0])
    world.field.field_landmarks.landmarks["F1L"] = expected
    assert world.field.field_landmarks.get_landmark_position("F1L") is expected
