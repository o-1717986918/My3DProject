"""Brax actor compatible with the competition walk ONNX teacher."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from brax.training import distribution
from brax.training import networks as brax_networks
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen
from flax.core import unfreeze
from jax import lax
import jax.numpy as jp
import numpy as np
import onnx
from onnx import numpy_helper


class LegacyPolicyWithStd(linen.Module):
    """Exact deterministic teacher graph plus a small train-time exploration std."""

    init_noise_std: float = float(np.exp(-2.0))

    @linen.compact
    def __call__(self, observations: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
        hidden = linen.Dense(512, precision=lax.Precision.HIGHEST, name="fc1")(
            observations
        )
        hidden = linen.LayerNorm(
            epsilon=1.0e-6,
            use_fast_variance=False,
            name="layer_norm",
        )(hidden)
        hidden = linen.elu(hidden)
        hidden = linen.elu(
            linen.Dense(256, precision=lax.Precision.HIGHEST, name="fc2")(hidden)
        )
        hidden = linen.elu(
            linen.Dense(128, precision=lax.Precision.HIGHEST, name="fc3")(hidden)
        )
        mean = jp.clip(
            linen.Dense(23, precision=lax.Precision.HIGHEST, name="fc4")(hidden),
            -10.0,
            10.0,
        )
        std = self.param(
            "std",
            lambda _: jp.full((23,), self.init_noise_std, dtype=jp.float32),
        )
        return mean, jp.broadcast_to(std, mean.shape)


def make_legacy_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    **unused: Any,
) -> ppo_networks.PPONetworks:
    """Build the teacher-compatible actor and a privileged-state critic."""
    del unused
    if action_size != 23:
        raise ValueError(f"legacy policy expects 23 actions, got {action_size}")
    if not isinstance(observation_size, Mapping):
        raise ValueError(
            "legacy policy requires state and privileged_state observations"
        )
    actor_observation_size = observation_size["state"][-1]
    if actor_observation_size not in (78, 80):
        raise ValueError(
            "legacy warm-start expects a 78-dimensional actor observation or "
            "the 80-dimensional phase extension"
        )

    module = LegacyPolicyWithStd()

    def apply(processor_params: Any, policy_params: Any, observation: Any):
        actor_observation = observation["state"]
        actor_observation = preprocess_observations_fn(
            actor_observation,
            brax_networks.normalizer_select(processor_params, "state"),
        )
        return module.apply(policy_params, actor_observation)

    policy_network = brax_networks.FeedForwardNetwork(
        init=lambda key: module.init(
            key, jp.zeros((1, actor_observation_size), dtype=jp.float32)
        ),
        apply=apply,
    )
    value_network = brax_networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=(512, 256, 128),
        activation=linen.elu,
        obs_key="privileged_state",
    )
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=distribution.NormalDistribution(
            event_size=action_size
        ),
    )


def load_onnx_teacher_params(initial_params: Any, model_path: Path) -> Any:
    """Replace an initialized actor's deterministic layers with ONNX tensors."""
    model = onnx.load(str(model_path))
    arrays = {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in model.graph.initializer
    }
    expected = {
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
        "fc4.weight",
        "fc4.bias",
        "layer_norm.weight",
        "layer_norm.bias",
    }
    missing = expected - arrays.keys()
    if missing:
        raise ValueError(f"teacher ONNX is missing tensors: {sorted(missing)}")

    params = unfreeze(initial_params)
    learned = params["params"]
    for layer in ("fc1", "fc2", "fc3", "fc4"):
        teacher_kernel = jp.asarray(arrays[f"{layer}.weight"].T)
        initialized_kernel = learned[layer]["kernel"]
        if layer == "fc1" and initialized_kernel.shape[0] > teacher_kernel.shape[0]:
            initialized_kernel = initialized_kernel.at[: teacher_kernel.shape[0]].set(
                teacher_kernel
            )
            initialized_kernel = initialized_kernel.at[teacher_kernel.shape[0] :].set(
                0.0
            )
            learned[layer]["kernel"] = initialized_kernel
        else:
            if initialized_kernel.shape != teacher_kernel.shape:
                raise ValueError(
                    f"teacher {layer} shape {teacher_kernel.shape} does not match "
                    f"initialized shape {initialized_kernel.shape}"
                )
            learned[layer]["kernel"] = teacher_kernel
        learned[layer]["bias"] = jp.asarray(arrays[f"{layer}.bias"])
    learned["layer_norm"]["scale"] = jp.asarray(arrays["layer_norm.weight"])
    learned["layer_norm"]["bias"] = jp.asarray(arrays["layer_norm.bias"])
    return params
