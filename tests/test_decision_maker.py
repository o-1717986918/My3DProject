from types import SimpleNamespace

import numpy as np
import pytest

from mujococodebase.decision_maker import AttackPhase, DecisionMaker
from mujococodebase.world.play_mode import PlayModeGroupEnum
from mujococodebase.world.world import World


class SkillRecorder:
    def __init__(self):
        self.calls = []
        self.finish_kick = False

    def execute(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return self.finish_kick if name == "KickRight" else False


class BeamRecorder:
    def __init__(self):
        self.beams = []

    def commit_beam(self, pos2d, rotation):
        self.beams.append((pos2d, rotation))


def make_decision_maker():
    skills = SkillRecorder()
    server = BeamRecorder()
    robot = SimpleNamespace(global_orientation_euler=np.zeros(3))
    agent = SimpleNamespace(skills_manager=skills, server=server, robot=robot)
    world = World(agent=agent, team_name="Team", number=7, field_name="my_field")
    agent.world = world
    return DecisionMaker(agent), world, skills, server


def test_aligned_attacker_enters_kick_phase():
    decision, world, skills, _ = make_decision_maker()
    world.server_time = 1.0
    world.ball_last_seen_time = 1.0
    world.ball_pos = np.array([0.0, 0.0, 0.11])
    world.global_position = np.array([-0.62, 0.04, 0.62])

    decision.run_attack()

    assert decision.attack_phase is AttackPhase.KICK
    assert skills.calls[-1][0] == "KickRight"


def test_finished_kick_enters_timed_recovery():
    decision, world, skills, _ = make_decision_maker()
    decision.attack_phase = AttackPhase.KICK
    world.server_time = 4.0
    skills.finish_kick = True

    decision.run_attack()

    assert decision.attack_phase is AttackPhase.RECOVER
    assert decision.kick_cooldown_until == 4.8


def test_stale_ball_triggers_search_rotation():
    decision, world, skills, _ = make_decision_maker()
    world.server_time = 2.0

    decision.run_attack()

    assert decision.attack_phase is AttackPhase.SEARCH
    name, _, kwargs = skills.calls[-1]
    assert name == "Walk"
    assert kwargs["is_orientation_absolute"] is False


def test_right_side_beam_is_mirrored_with_degree_yaw():
    decision, world, skills, server = make_decision_maker()
    world.is_left_team = False
    world.playmode_group = PlayModeGroupEnum.ACTIVE_BEAM
    decision._run_beam_state()

    assert server.beams == [([0.9, 0.4], -180.0)]
    assert skills.calls == []


def test_passive_kickoff_keeps_all_field_players_outside_centre_circle():
    decision, world, _, _ = make_decision_maker()
    world.playmode_group = PlayModeGroupEnum.PASSIVE_BEAM

    poses = decision.MY_FIELD_PASSIVE_KICKOFF_POSES

    assert all(x < 0.0 for number, (x, _, _) in poses.items() if number != 1)
    assert all(
        np.hypot(x, y) > 5.5 for number, (x, y, _) in poses.items() if number != 1
    )


@pytest.mark.parametrize(
    ("ball", "owner"),
    [
        ((-12.0, -2.0), 2),
        ((-12.0, 2.0), 3),
        ((0.0, -7.0), 5),
        ((-3.0, 0.0), 4),
        ((0.0, 0.0), 7),
        ((0.0, 7.0), 6),
        ((9.0, 0.0), 7),
    ],
)
def test_ball_owner_covers_defensive_midfield_and_attacking_zones(ball, owner):
    decision, world, _, _ = make_decision_maker()
    world.ball_pos[:2] = ball

    assert decision._select_ball_owner() == owner
