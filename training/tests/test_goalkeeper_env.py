import jax
import jax.numpy as jp
import numpy as np
import pytest

from my3d_rl.goalkeeper_env import ApolloGoalkeeperBlock


def _fixed_environment(side: int) -> ApolloGoalkeeperBlock:
    return ApolloGoalkeeperBlock(
        config_overrides={
            "fixed_shot_side": side,
            "shot_speed_range": [5.0, 5.0],
            "shot_lateral_range": [1.35, 1.35],
            "shot_start_distance_range": [8.0, 8.0],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
        }
    )


@pytest.fixture(scope="module")
def positive_env() -> ApolloGoalkeeperBlock:
    return _fixed_environment(1)


@pytest.fixture(scope="module")
def negative_env() -> ApolloGoalkeeperBlock:
    return _fixed_environment(-1)


def test_goalkeeper_reset_preserves_actor_contract_and_exposes_task_to_critic(
    positive_env,
):
    env = positive_env
    state = jax.jit(env.reset)(jax.random.PRNGKey(20_261_501))

    assert env.observation_size == {"state": (84,), "privileged_state": (90,)}
    np.testing.assert_allclose(state.info["command"], [0.0, 0.5, 0.0])
    assert np.isclose(float(state.info["shot_lateral_m"]), 1.35)
    assert np.isclose(float(state.info["shot_speed_mps"]), 5.0)
    assert float(state.data.qvel[env._ball_dof]) < -4.9
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(np.asarray(state.obs["privileged_state"])).all()
    assert float(state.obs["state"][-6]) > 0.7
    assert float(state.obs["state"][-5]) > 0.6
    assert float(state.obs["state"][-3]) < -0.5


def test_goalkeeper_shot_side_mirrors_ball_and_command(positive_env, negative_env):
    key = jax.random.PRNGKey(20_261_502)
    positive = positive_env.reset(key)
    negative = negative_env.reset(key)

    positive_ball_y = float(positive.data.xpos[positive_env._ball_body, 1])
    negative_ball_y = float(negative.data.xpos[negative_env._ball_body, 1])
    assert positive_ball_y > 1.3
    assert negative_ball_y < -1.3
    assert np.isclose(positive_ball_y, -negative_ball_y, atol=1.0e-5)
    assert float(positive.info["command"][1]) == 0.5
    assert float(negative.info["command"][1]) == -0.5


def test_goalkeeper_step_advances_incoming_ball_and_stays_finite(positive_env):
    env = positive_env
    state = jax.jit(env.reset)(jax.random.PRNGKey(20_261_503))
    previous_ball_x = float(state.data.xpos[env._ball_body, 0])
    state = jax.jit(env.step)(state, jp.zeros(23))

    assert float(state.data.xpos[env._ball_body, 0]) < previous_ball_x
    assert np.isfinite(float(state.reward))
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert float(state.metrics["diagnostic/goalkeeper_shot_lateral_m"]) == pytest.approx(
        1.35
    )
