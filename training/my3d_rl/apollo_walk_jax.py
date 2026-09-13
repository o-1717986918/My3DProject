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
    adapter_kernels: tuple[jp.ndarray, ...]
    adapter_biases: tuple[jp.ndarray, ...]

    @property
    def observation_size(self) -> int:
        return int(self.observation_mean.shape[0])

    def __call__(self, observation: jp.ndarray) -> jp.ndarray:
        normalized = (observation - self.observation_mean) / self.observation_std
        # Goalkeeper exports keep Apollo's 78-value actor frozen and feed the
        # complete 84-value observation only to the residual adapter.
        hidden = (
            jp.matmul(
                normalized[..., : self.kernels[0].shape[0]],
                self.kernels[0],
                precision=lax.Precision.HIGHEST,
            )
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
        if self.adapter_kernels:
            adapter = normalized
            for kernel, bias in zip(
                self.adapter_kernels[:-1], self.adapter_biases[:-1], strict=True
            ):
                adapter = (
                    jp.matmul(adapter, kernel, precision=lax.Precision.HIGHEST)
                    + bias
                )
                adapter = jp.where(adapter > 0.0, adapter, jp.expm1(adapter))
            adapter = (
                jp.matmul(
                    adapter,
                    self.adapter_kernels[-1],
                    precision=lax.Precision.HIGHEST,
                )
                + self.adapter_biases[-1]
            )
            output = output + jp.tanh(adapter)
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
    elif "fc1.weight" in arrays:
        layer_names = tuple(f"fc{index}" for index in range(1, 5))
        required = {
            *(
                f"{name}.{field}"
                for name in layer_names
                for field in ("weight", "bias")
            ),
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"Apollo walk ONNX is missing tensors: {sorted(missing)}")
        actor_size = int(
            arrays.get("adapter_fc1.weight", arrays["fc1.weight"]).shape[1]
        )
        observation_mean = jp.zeros(actor_size, dtype=jp.float32)
        observation_std = jp.ones(actor_size, dtype=jp.float32)
        layer_norm_names = {"layer_norm.weight", "layer_norm.bias"}
        present_layer_norm_names = layer_norm_names.intersection(arrays)
        if present_layer_norm_names and present_layer_norm_names != layer_norm_names:
            raise ValueError("Apollo walk ONNX has an incomplete layer norm")
        has_layer_norm = present_layer_norm_names == layer_norm_names
        layer_norm_scale = (
            jp.asarray(arrays["layer_norm.weight"]) if has_layer_norm else None
        )
        layer_norm_bias = (
            jp.asarray(arrays["layer_norm.bias"]) if has_layer_norm else None
        )
    else:
        raise ValueError(
            "Apollo walk ONNX does not contain actor.0 or fc1 actor tensors"
        )
    kernels = tuple(jp.asarray(arrays[f"{name}.weight"].T) for name in layer_names)
    biases = tuple(jp.asarray(arrays[f"{name}.bias"]) for name in layer_names)
    if (
        kernels[0].shape not in ((78, 512), (84, 512))
        or kernels[-1].shape != (128, 23)
    ):
        raise ValueError("Apollo walk ONNX has an unsupported architecture")
    adapter_layer_names = ("adapter_fc1", "adapter_fc2", "adapter_fc3")
    adapter_required = {
        f"{name}.{field}"
        for name in adapter_layer_names
        for field in ("weight", "bias")
    }
    adapter_present = adapter_required.intersection(arrays)
    if adapter_present and adapter_present != adapter_required:
        raise ValueError("Apollo goalkeeper ONNX has an incomplete adapter")
    adapter_kernels = (
        tuple(
            jp.asarray(arrays[f"{name}.weight"].T) for name in adapter_layer_names
        )
        if adapter_present
        else ()
    )
    adapter_biases = (
        tuple(jp.asarray(arrays[f"{name}.bias"]) for name in adapter_layer_names)
        if adapter_present
        else ()
    )
    if adapter_kernels and (
        adapter_kernels[0].shape != (84, 128)
        or adapter_kernels[1].shape != (128, 64)
        or adapter_kernels[2].shape != (64, 23)
    ):
        raise ValueError("Apollo goalkeeper ONNX has an unsupported adapter")
    return ApolloWalkJax(
        kernels=kernels,
        biases=biases,
        observation_mean=observation_mean,
        observation_std=observation_std,
        layer_norm_scale=layer_norm_scale,
        layer_norm_bias=layer_norm_bias,
        adapter_kernels=adapter_kernels,
        adapter_biases=adapter_biases,
    )
