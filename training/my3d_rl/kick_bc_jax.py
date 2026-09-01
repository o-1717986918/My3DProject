"""Dependency-light JAX inference for exported kick behavior clones."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jp
from onnx import numpy_helper
import onnx
import numpy as np


def load_kick_behavior_clone_jax(
    path: Path, *, observation_size: int, action_size: int
) -> Callable[[jax.Array], jax.Array]:
    """Load the repository's normalized three-layer tanh ONNX into JAX."""
    if observation_size < 1 or action_size < 1:
        raise ValueError("kick policy dimensions must be positive")
    model = onnx.load(str(path))
    arrays = {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), np.float32)
        for initializer in model.graph.initializer
    }
    expected = {"observation_mean", "observation_std"}
    expected.update(
        f"dense_{index}_{suffix}"
        for index in range(3)
        for suffix in ("kernel", "bias")
    )
    missing = expected - set(arrays)
    if missing:
        raise ValueError(f"kick behavior clone is missing tensors {sorted(missing)}")
    mean = arrays["observation_mean"]
    std = arrays["observation_std"]
    layers = [
        (arrays[f"dense_{index}_kernel"], arrays[f"dense_{index}_bias"])
        for index in range(3)
    ]
    if mean.shape != (observation_size,) or std.shape != mean.shape:
        raise ValueError("kick behavior clone normalizer shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("kick behavior clone normalizer is invalid")
    previous = observation_size
    for index, (kernel, bias) in enumerate(layers):
        if kernel.ndim != 2 or kernel.shape[0] != previous:
            raise ValueError(f"kick behavior clone dense layer {index} shape mismatch")
        if bias.shape != (kernel.shape[1],):
            raise ValueError(f"kick behavior clone dense bias {index} shape mismatch")
        if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
            raise ValueError("kick behavior clone contains non-finite parameters")
        previous = kernel.shape[1]
    if previous != action_size:
        raise ValueError("kick behavior clone action width mismatch")

    jax_mean = jp.asarray(mean)
    jax_std = jp.asarray(std)
    jax_layers = tuple((jp.asarray(kernel), jp.asarray(bias)) for kernel, bias in layers)

    @jax.jit
    def policy(observation: jax.Array) -> jax.Array:
        hidden = (jp.asarray(observation) - jax_mean) / jax_std
        for kernel, bias in jax_layers:
            hidden = jp.tanh(
                jp.matmul(hidden, kernel, precision=jax.lax.Precision.HIGHEST) + bias
            )
        return hidden

    return policy
