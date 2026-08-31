"""Supervised initialization and ONNX export for kick_policy_v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import optax


class KickBehaviorClone(nn.Module):
    """Small deployable MLP with bounded joint-action output."""

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        hidden = jnp.tanh(nn.Dense(256)(observations))
        hidden = jnp.tanh(nn.Dense(128)(hidden))
        return jnp.tanh(nn.Dense(23)(hidden))


@dataclass(frozen=True)
class KickBCResult:
    params: Any
    observation_mean: np.ndarray
    observation_std: np.ndarray
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    train_loss: float
    validation_loss: float
    history: tuple[dict[str, float], ...]


def load_teacher_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as dataset:
        required = {"observations", "actions", "episode_ids"}
        missing = required - set(dataset.files)
        if missing:
            raise ValueError(f"teacher dataset is missing {sorted(missing)}")
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.float32)
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int32)
    if observations.ndim != 2 or observations.shape[1] != 96:
        raise ValueError("teacher observations must have shape [N, 96]")
    if actions.shape != (observations.shape[0], 23):
        raise ValueError("teacher actions must have shape [N, 23]")
    if episode_ids.shape != (observations.shape[0],):
        raise ValueError("episode_ids must have shape [N]")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("teacher dataset must be finite")
    if np.max(np.abs(actions)) > 1.0 + 1.0e-6:
        raise ValueError("teacher actions exceed the deployment contract")
    return {
        "observations": observations,
        "actions": actions,
        "episode_ids": episode_ids,
    }


def train_behavior_clone(
    dataset: dict[str, np.ndarray],
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    validation_episode_ids: tuple[int, ...] | None = None,
) -> KickBCResult:
    observations = dataset["observations"]
    actions = dataset["actions"]
    episode_ids = dataset["episode_ids"]
    unique_episodes = tuple(int(value) for value in np.unique(episode_ids))
    if len(unique_episodes) < 2:
        raise ValueError("behavior cloning requires at least two condition episodes")
    if validation_episode_ids is None:
        validation_episode_ids = (unique_episodes[-1],)
    validation_episode_ids = tuple(int(value) for value in validation_episode_ids)
    if not set(validation_episode_ids) <= set(unique_episodes):
        raise ValueError("validation episode is not present in the dataset")
    train_episode_ids = tuple(
        episode for episode in unique_episodes if episode not in validation_episode_ids
    )
    if not train_episode_ids:
        raise ValueError("validation split leaves no training episodes")
    if steps < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size and learning_rate must be positive")

    train_mask = np.isin(episode_ids, train_episode_ids)
    validation_mask = np.isin(episode_ids, validation_episode_ids)
    train_observations = observations[train_mask]
    train_actions = actions[train_mask]
    validation_observations = observations[validation_mask]
    validation_actions = actions[validation_mask]
    observation_mean = train_observations.mean(axis=0)
    observation_std = np.maximum(train_observations.std(axis=0), 1.0e-3)
    normalized_train = (train_observations - observation_mean) / observation_std
    normalized_validation = (
        validation_observations - observation_mean
    ) / observation_std

    model = KickBehaviorClone()
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, 96), dtype=jnp.float32))["params"]
    optimizer = optax.adamw(learning_rate, weight_decay=1.0e-6)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        batch_observations: jax.Array,
        batch_actions: jax.Array,
    ) -> tuple[Any, Any, jax.Array]:
        def loss_fn(candidate_params: Any) -> jax.Array:
            prediction = model.apply({"params": candidate_params}, batch_observations)
            return jnp.mean(jnp.square(prediction - batch_actions))

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, new_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_optimizer_state, loss

    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    report_interval = max(1, steps // 20)
    for step in range(steps):
        indices = rng.integers(0, normalized_train.shape[0], size=batch_size)
        params, optimizer_state, loss = train_step(
            params,
            optimizer_state,
            jnp.asarray(normalized_train[indices]),
            jnp.asarray(train_actions[indices]),
        )
        if step % report_interval == 0 or step + 1 == steps:
            history.append({"step": float(step + 1), "batch_loss": float(loss)})

    train_prediction = np.asarray(
        model.apply({"params": params}, jnp.asarray(normalized_train))
    )
    validation_prediction = np.asarray(
        model.apply({"params": params}, jnp.asarray(normalized_validation))
    )
    train_loss = float(np.mean(np.square(train_prediction - train_actions)))
    validation_loss = float(
        np.mean(np.square(validation_prediction - validation_actions))
    )
    return KickBCResult(
        params=params,
        observation_mean=observation_mean.astype(np.float32),
        observation_std=observation_std.astype(np.float32),
        train_episode_ids=train_episode_ids,
        validation_episode_ids=validation_episode_ids,
        train_loss=train_loss,
        validation_loss=validation_loss,
        history=tuple(history),
    )


def apply_behavior_clone(result: KickBCResult, observations: np.ndarray) -> np.ndarray:
    normalized = (
        np.asarray(observations, dtype=np.float32) - result.observation_mean
    ) / result.observation_std
    model = KickBehaviorClone()
    return np.asarray(model.apply({"params": result.params}, jnp.asarray(normalized)))


def apply_behavior_clone_numpy(
    result: KickBCResult, observations: np.ndarray
) -> np.ndarray:
    """Evaluate the exported graph semantics using NumPy float32 operations."""
    hidden = (
        np.asarray(observations, dtype=np.float32) - result.observation_mean
    ) / result.observation_std
    for kernel, bias in _dense_arrays(result.params):
        hidden = np.tanh(hidden @ kernel + bias)
    return hidden


def _dense_arrays(params: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.asarray(params[f"Dense_{index}"]["kernel"], dtype=np.float32),
            np.asarray(params[f"Dense_{index}"]["bias"], dtype=np.float32),
        )
        for index in range(3)
    ]


def export_behavior_clone_onnx(result: KickBCResult, output_path: Path) -> None:
    """Export the normalized Flax MLP as a dependency-free ONNX graph."""
    initializers = [
        numpy_helper.from_array(result.observation_mean, name="observation_mean"),
        numpy_helper.from_array(result.observation_std, name="observation_std"),
    ]
    nodes = [
        helper.make_node("Sub", ["observations", "observation_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "observation_std"], ["normalized"]),
    ]
    previous = "normalized"
    dense_arrays = _dense_arrays(result.params)
    for index, (kernel, bias) in enumerate(dense_arrays):
        kernel_name = f"dense_{index}_kernel"
        bias_name = f"dense_{index}_bias"
        matmul_name = f"dense_{index}_matmul"
        add_name = f"dense_{index}_add"
        output_name = "actions" if index == len(dense_arrays) - 1 else f"hidden_{index}"
        initializers.extend(
            [
                numpy_helper.from_array(kernel, name=kernel_name),
                numpy_helper.from_array(bias, name=bias_name),
            ]
        )
        nodes.append(helper.make_node("MatMul", [previous, kernel_name], [matmul_name]))
        nodes.append(helper.make_node("Add", [matmul_name, bias_name], [add_name]))
        nodes.append(helper.make_node("Tanh", [add_name], [output_name]))
        previous = output_name

    graph = helper.make_graph(
        nodes,
        "kick_policy_v2_behavior_clone",
        [helper.make_tensor_value_info("observations", TensorProto.FLOAT, [1, 96])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 23])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="My3DProject",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def verify_onnx_parity(
    result: KickBCResult,
    model_path: Path,
    observations: np.ndarray,
) -> dict[str, float]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    selected = np.asarray(observations[: min(256, len(observations))], dtype=np.float32)
    expected = apply_behavior_clone_numpy(result, selected)
    actual = np.concatenate(
        [session.run(None, {"observations": row[None, :]})[0] for row in selected],
        axis=0,
    )
    error = np.abs(expected - actual)
    return {
        "samples": float(selected.shape[0]),
        "maximum_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error)),
    }
