"""Shape-safe zero-row transfer from K1-D into the K2 ball actor."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from .soccer_ball_policy import (
    SOCCER_BALL_ACTOR_SIZE,
    SOCCER_BALL_FEATURE_SIZE,
    SOCCER_BALL_PRIVILEGED_SIZE,
    SOCCER_MOTION_ACTOR_SIZE,
)
from .soccer_motion_policy import SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE


def _copied_tree(value: Any) -> Any:
    return jax.tree.map(lambda leaf: jp.asarray(leaf), value)


def _append_statistics_prefix(value: Any, prefix_size: int) -> Any:
    array = jp.asarray(value)
    if array.shape != (prefix_size,):
        raise ValueError(
            f"actor statistic shape {array.shape} != ({prefix_size},)"
        )
    return jp.concatenate(
        [array, jp.zeros((SOCCER_BALL_FEATURE_SIZE,), dtype=array.dtype)]
    )


def _insert_statistics_before_privileged_suffix(value: Any) -> Any:
    array = jp.asarray(value)
    expected = SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE
    if array.shape != (expected,):
        raise ValueError(f"critic statistic shape {array.shape} != ({expected},)")
    return jp.concatenate(
        [
            array[:SOCCER_MOTION_ACTOR_SIZE],
            jp.zeros((SOCCER_BALL_FEATURE_SIZE,), dtype=array.dtype),
            array[SOCCER_MOTION_ACTOR_SIZE:],
        ]
    )


def _expand_normalizer(normalizer: Any) -> Any:
    mean = dict(normalizer.mean)
    std = dict(normalizer.std)
    variance = dict(normalizer.summed_variance)
    mean["state"] = _append_statistics_prefix(
        mean["state"], SOCCER_MOTION_ACTOR_SIZE
    )
    mean["privileged_state"] = _insert_statistics_before_privileged_suffix(
        mean["privileged_state"]
    )
    std["state"] = _append_statistics_prefix(
        std["state"], SOCCER_MOTION_ACTOR_SIZE
    ).at[SOCCER_MOTION_ACTOR_SIZE:].set(1.0)
    std["privileged_state"] = _insert_statistics_before_privileged_suffix(
        std["privileged_state"]
    ).at[
        SOCCER_MOTION_ACTOR_SIZE:
        SOCCER_MOTION_ACTOR_SIZE + SOCCER_BALL_FEATURE_SIZE
    ].set(1.0)
    variance["state"] = _append_statistics_prefix(
        variance["state"], SOCCER_MOTION_ACTOR_SIZE
    )
    variance["privileged_state"] = _insert_statistics_before_privileged_suffix(
        variance["privileged_state"]
    )
    return normalizer.replace(mean=mean, std=std, summed_variance=variance)


def expand_soccer_motion_params(params: Any) -> list[Any]:
    """Expand compatible 110/118 PPO params to 126/134 with zero new rows."""
    if not isinstance(params, (list, tuple)) or len(params) != 3:
        raise ValueError("PPO checkpoint must contain normalizer, actor and critic")
    normalizer, actor_source, critic_source = params
    actor = _copied_tree(actor_source)
    critic = _copied_tree(critic_source)

    actor_kernel = jp.asarray(
        actor["params"]["MLP_0"]["hidden_0"]["kernel"]
    )
    if actor_kernel.shape[0] != SOCCER_MOTION_ACTOR_SIZE:
        raise ValueError("source actor does not use the 110-value K1 boundary")
    actor["params"]["MLP_0"]["hidden_0"]["kernel"] = jp.concatenate(
        [
            actor_kernel,
            jp.zeros(
                (SOCCER_BALL_FEATURE_SIZE, actor_kernel.shape[1]),
                dtype=actor_kernel.dtype,
            ),
        ],
        axis=0,
    )

    critic_kernel = jp.asarray(critic["params"]["hidden_0"]["kernel"])
    if critic_kernel.shape[0] != SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE:
        raise ValueError("source critic does not use the 118-value K1 boundary")
    critic["params"]["hidden_0"]["kernel"] = jp.concatenate(
        [
            critic_kernel[:SOCCER_MOTION_ACTOR_SIZE],
            jp.zeros(
                (SOCCER_BALL_FEATURE_SIZE, critic_kernel.shape[1]),
                dtype=critic_kernel.dtype,
            ),
            critic_kernel[SOCCER_MOTION_ACTOR_SIZE:],
        ],
        axis=0,
    )
    expanded = [_expand_normalizer(normalizer), actor, critic]
    if expanded[1]["params"]["MLP_0"]["hidden_0"]["kernel"].shape[0] != (
        SOCCER_BALL_ACTOR_SIZE
    ):
        raise AssertionError("expanded actor size mismatch")
    if expanded[2]["params"]["hidden_0"]["kernel"].shape[0] != (
        SOCCER_BALL_PRIVILEGED_SIZE
    ):
        raise AssertionError("expanded critic size mismatch")
    return expanded


def expand_privileged_observation(old_privileged: np.ndarray) -> np.ndarray:
    """Insert neutral ball features before K1's final eight critic-only values."""
    value = np.asarray(old_privileged, dtype=np.float32)
    if value.shape[-1] != SOCCER_MOTION_PRIVILEGED_OBSERVATION_SIZE:
        raise ValueError("old privileged observation has an incompatible shape")
    zeros = np.zeros(value.shape[:-1] + (SOCCER_BALL_FEATURE_SIZE,), np.float32)
    return np.concatenate(
        [
            value[..., :SOCCER_MOTION_ACTOR_SIZE],
            zeros,
            value[..., SOCCER_MOTION_ACTOR_SIZE:],
        ],
        axis=-1,
    )
