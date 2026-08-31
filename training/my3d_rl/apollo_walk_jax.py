"""Pure-JAX evaluator for Apollo's accepted 78-to-23 walk ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jp
from jax import lax
import numpy as np
import onnx
from onnx import numpy_helper


@dataclass(frozen=True)
class ApolloWalkJax:
    kernels: tuple[jp.ndarray, ...]
    biases: tuple[jp.ndarray, ...]
    observation_mean: jp.ndarray
    observation_std: jp.ndarray
    layer_norm_scale: jp.ndarray | None
    layer_norm_bias: jp.ndarray | None

    def __call__(self, observation: jp.ndarray) -> jp.ndarray:
        normalized = (observation - self.observation_mean) / self.observation_std
        hidden = (
            jp.matmul(normalized, self.kernels[0], precision=lax.Precision.HIGHEST)
            + self.biases[0]
        )
        if self.layer_norm_scale is not None and self.layer_norm_bias is not None:
            mean = jp.mean(hidden, axis=-1, keepdims=True)
            variance = jp.mean(jp.square(hidden - mean), axis=-1, keepdims=True)
            hidden = (hidden - mean) * jp.rsqrt(variance + 1.0e-6)
            hidden = hidden * self.layer_norm_scale + self.layer_norm_bias
        hidden = jp.where(hidden > 0.0, hidden, jp.expm1(hidden))
        for kernel, bias in zip(self.kernels[1:-1], self.biases[1:-1], strict=True):
            hidden = jp.matmul(hidden, kernel, precision=lax.Precision.HIGHEST) + bias
            hidden = jp.where(hidden > 0.0, hidden, jp.expm1(hidden))
        output = (
            jp.matmul(hidden, self.kernels[-1], precision=lax.Precision.HIGHEST)
            + self.biases[-1]
        )
        return jp.clip(output, -5.0, 5.0)


def load_apollo_walk_jax(model_path: Path) -> ApolloWalkJax:
    model = onnx.load(str(model_path))
    arrays = {
        initializer.name: np.asarray(
            numpy_helper.to_array(initializer), dtype=np.float32
        ).copy()
        for initializer in model.graph.initializer
    }
    if "actor.0.weight" in arrays:
        layer_names = ("actor.0", "actor.2", "actor.4", "actor.6")
        required = {
            *(
                f"{name}.{field}"
                for name in layer_names
                for field in ("weight", "bias")
            ),
            "normalizer._mean",
            "add",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"Apollo walk ONNX is missing tensors: {sorted(missing)}")
        observation_mean = jp.asarray(arrays["normalizer._mean"].reshape(78))
        observation_std = jp.asarray(arrays["add"].reshape(78))
        layer_norm_scale = None
        layer_norm_bias = None
    else:
        layer_names = tuple(f"fc{index}" for index in range(1, 5))
        required = {
            *(
                f"{name}.{field}"
                for name in layer_names
                for field in ("weight", "bias")
            ),
            "layer_norm.weight",
            "layer_norm.bias",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"Apollo walk ONNX is missing tensors: {sorted(missing)}")
        observation_mean = jp.zeros(78, dtype=jp.float32)
        observation_std = jp.ones(78, dtype=jp.float32)
        layer_norm_scale = jp.asarray(arrays["layer_norm.weight"])
        layer_norm_bias = jp.asarray(arrays["layer_norm.bias"])
    kernels = tuple(jp.asarray(arrays[f"{name}.weight"].T) for name in layer_names)
    biases = tuple(jp.asarray(arrays[f"{name}.bias"]) for name in layer_names)
    if kernels[0].shape != (78, 512) or kernels[-1].shape != (128, 23):
        raise ValueError("Apollo walk ONNX has an unsupported architecture")
    return ApolloWalkJax(
        kernels=kernels,
        biases=biases,
        observation_mean=observation_mean,
        observation_std=observation_std,
        layer_norm_scale=layer_norm_scale,
        layer_norm_bias=layer_norm_bias,
    )
