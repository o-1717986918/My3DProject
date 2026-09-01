import jax.numpy as jp

from my3d_rl.kick_env import kick_gate_success


def test_training_gate_matches_the_central_physical_acceptance_envelope():
    accepted = dict(
        contact=jp.array(True),
        fallen=jp.array(False),
        remaining_distance=jp.array(0.4),
        lateral_error=jp.array(0.3),
        maximum_directional_speed=jp.array(1.8),
        requested_ball_speed=jp.array(1.43),
    )
    assert bool(kick_gate_success(**accepted))

    for key, value in (
        ("contact", jp.array(False)),
        ("fallen", jp.array(True)),
        ("remaining_distance", jp.array(0.51)),
        ("lateral_error", jp.array(0.51)),
        ("maximum_directional_speed", jp.array(2.44)),
    ):
        rejected = dict(accepted)
        rejected[key] = value
        assert not bool(kick_gate_success(**rejected))
