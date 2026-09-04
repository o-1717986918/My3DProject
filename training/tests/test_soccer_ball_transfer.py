from __future__ import annotations

from brax.training import types as brax_types
from brax.training.acme import running_statistics
import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.soccer_ball_transfer import (
    expand_privileged_observation,
    expand_soccer_motion_params,
)


def _networks(profile_name: str, actor_size: int, critic_size: int):
    return get_ppo_profile(profile_name).network_factory()(
        {"state": actor_size, "privileged_state": critic_size},
        23,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )


def _assert_zero_row_transfer_preserves_actor_and_critic_outputs():
    old_networks = _networks("soccer_motion_residual_v3", 110, 118)
    new_networks = _networks("soccer_ball_motion_residual_v1", 126, 134)
    actor = old_networks.policy_network.init(jax.random.PRNGKey(1))
    critic = old_networks.value_network.init(jax.random.PRNGKey(2))
    actor["params"]["Dense_0"]["kernel"] = jax.random.normal(
        jax.random.PRNGKey(3), (128, 23)
    )
    normalizer = running_statistics.init_state(
        {"state": jp.zeros(110), "privileged_state": jp.zeros(118)}
    )
    expanded = expand_soccer_motion_params([normalizer, actor, critic])

    old_actor_kernel = actor["params"]["MLP_0"]["hidden_0"]["kernel"]
    new_actor_kernel = expanded[1]["params"]["MLP_0"]["hidden_0"]["kernel"]
    np.testing.assert_array_equal(new_actor_kernel[:110], old_actor_kernel)
    np.testing.assert_array_equal(new_actor_kernel[110:], 0.0)
    old_critic_kernel = critic["params"]["hidden_0"]["kernel"]
    new_critic_kernel = expanded[2]["params"]["hidden_0"]["kernel"]
    np.testing.assert_array_equal(new_critic_kernel[:110], old_critic_kernel[:110])
    np.testing.assert_array_equal(new_critic_kernel[110:126], 0.0)
    np.testing.assert_array_equal(new_critic_kernel[126:], old_critic_kernel[110:])

    old_actor_obs = jax.random.normal(jax.random.PRNGKey(4), (32, 110))
    new_actor_obs = jp.concatenate([old_actor_obs, jp.zeros((32, 16))], axis=1)
    old_critic_obs = np.asarray(
        jax.random.normal(jax.random.PRNGKey(5), (32, 118)), np.float32
    )
    new_critic_obs = expand_privileged_observation(old_critic_obs)
    old_policy = old_networks.policy_network.apply(
        normalizer, actor, {"state": old_actor_obs}
    )[0]
    new_policy = new_networks.policy_network.apply(
        expanded[0], expanded[1], {"state": new_actor_obs}
    )[0]
    old_value = old_networks.value_network.apply(
        normalizer, critic, {"privileged_state": old_critic_obs}
    )
    new_value = new_networks.value_network.apply(
        expanded[0], expanded[2], {"privileged_state": new_critic_obs}
    )

    np.testing.assert_allclose(new_policy, old_policy, atol=5.0e-7, rtol=0.0)
    np.testing.assert_allclose(new_value, old_value, atol=5.0e-7, rtol=0.0)


def test_zero_row_transfer_preserves_actor_and_critic_outputs():
    # Inserting zero input rows changes the GEMM shape. GPU autotuning may then
    # select a different FP32 accumulation order, obscuring the exact transfer
    # invariant with backend-dependent roundoff. This structural parity test
    # deliberately runs on CPU; GPU behavior is covered by training smoke tests.
    with jax.default_device(jax.devices("cpu")[0]):
        _assert_zero_row_transfer_preserves_actor_and_critic_outputs()
