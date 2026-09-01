"""Continuous physical-outcome prediction for striker action banks."""

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


class StrikerOutcomeRegressor(nn.Module):
    """Predict the log final goal distance for every action prototype."""

    output_size: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        hidden = jnp.tanh(nn.Dense(256)(observations))
        hidden = jnp.tanh(nn.Dense(256)(hidden))
        hidden = jnp.tanh(nn.Dense(128)(hidden))
        return nn.Dense(self.output_size)(hidden)


@dataclass(frozen=True)
class StrikerOutcomeRegressorResult:
    params: Any
    observation_size: int
    action_prior_indices: tuple[int, ...]
    observation_mean: np.ndarray
    observation_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    fit_rollout_ids: tuple[int, ...]
    calibration_rollout_ids: tuple[int, ...]
    fit_loss: float
    calibration_loss: float
    history: tuple[dict[str, float], ...]


def _dense_arrays(params: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.asarray(params[f"Dense_{index}"]["kernel"], dtype=np.float32),
            np.asarray(params[f"Dense_{index}"]["bias"], dtype=np.float32),
        )
        for index in range(4)
    ]


def train_outcome_regressor(
    observations: np.ndarray,
    final_goal_distance_m: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    *,
    action_prior_indices: tuple[int, ...],
    fit_rollout_ids: tuple[int, ...],
    calibration_rollout_ids: tuple[int, ...],
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    unsafe_penalty_m: float = 20.0,
) -> StrikerOutcomeRegressorResult:
    """Fit an action-conditioned smooth physical outcome model."""
    states = np.asarray(observations, dtype=np.float32)
    distances = np.asarray(final_goal_distance_m, dtype=np.float32)
    unsafe = np.asarray(fall, dtype=np.float32)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    action_indices = tuple(int(value) for value in action_prior_indices)
    fit_ids = tuple(int(value) for value in fit_rollout_ids)
    calibration_ids = tuple(int(value) for value in calibration_rollout_ids)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("regressor observations must be a non-empty matrix")
    if (
        distances.ndim != 2
        or unsafe.shape != distances.shape
        or distances.shape[1] != states.shape[0]
        or ids.shape != (states.shape[0],)
        or len(action_indices) != distances.shape[0]
    ):
        raise ValueError("regressor outcomes, observations and IDs are misaligned")
    if (
        len(set(action_indices)) != len(action_indices)
        or not fit_ids
        or not calibration_ids
        or set(fit_ids) & set(calibration_ids)
    ):
        raise ValueError("regressor actions or rollout partitions are invalid")
    available_ids = set(int(value) for value in np.unique(ids))
    if not set(fit_ids) | set(calibration_ids) <= available_ids:
        raise ValueError("regressor split references unavailable rollouts")
    if (
        steps < 1
        or batch_size < 1
        or learning_rate <= 0.0
        or unsafe_penalty_m <= 0.0
        or not np.isfinite(states).all()
        or not np.isfinite(distances).all()
        or np.any(distances < 0.0)
        or not set(np.unique(unsafe).tolist()) <= {0.0, 1.0}
    ):
        raise ValueError("regressor inputs or optimization settings are invalid")

    fit_mask = np.isin(ids, fit_ids)
    calibration_mask = np.isin(ids, calibration_ids)
    observation_mean = states[fit_mask].mean(axis=0)
    observation_std = np.maximum(states[fit_mask].std(axis=0), 1.0e-3)
    normalized_states = (states - observation_mean) / observation_std
    penalized_distance = distances + unsafe_penalty_m * unsafe
    log_targets = np.log1p(penalized_distance).T
    target_mean = log_targets[fit_mask].mean(axis=0)
    target_std = np.maximum(log_targets[fit_mask].std(axis=0), 1.0e-3)
    normalized_targets = (log_targets - target_mean) / target_std

    model = StrikerOutcomeRegressor(output_size=len(action_indices))
    params = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, states.shape[1]), dtype=jnp.float32),
    )["params"]
    optimizer = optax.adamw(learning_rate, weight_decay=1.0e-5)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        batch_observations: jax.Array,
        batch_targets: jax.Array,
    ) -> tuple[Any, Any, jax.Array]:
        def loss_fn(candidate_params: Any) -> jax.Array:
            predictions = model.apply(
                {"params": candidate_params}, batch_observations
            )
            return jnp.mean(optax.huber_loss(predictions, batch_targets, delta=1.0))

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        return (
            optax.apply_updates(current_params, updates),
            next_optimizer_state,
            loss,
        )

    fit_rows = np.flatnonzero(fit_mask)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    report_interval = max(1, steps // 20)
    for step in range(steps):
        rows = rng.choice(fit_rows, size=batch_size, replace=True)
        params, optimizer_state, loss = train_step(
            params,
            optimizer_state,
            jnp.asarray(normalized_states[rows]),
            jnp.asarray(normalized_targets[rows]),
        )
        if step % report_interval == 0 or step + 1 == steps:
            history.append({"step": float(step + 1), "batch_loss": float(loss)})

    def loss_for(mask: np.ndarray) -> float:
        predictions = model.apply(
            {"params": params}, jnp.asarray(normalized_states[mask])
        )
        targets = jnp.asarray(normalized_targets[mask])
        return float(jnp.mean(optax.huber_loss(predictions, targets, delta=1.0)))

    return StrikerOutcomeRegressorResult(
        params=params,
        observation_size=int(states.shape[1]),
        action_prior_indices=action_indices,
        observation_mean=observation_mean.astype(np.float32),
        observation_std=observation_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        fit_rollout_ids=fit_ids,
        calibration_rollout_ids=calibration_ids,
        fit_loss=loss_for(fit_mask),
        calibration_loss=loss_for(calibration_mask),
        history=tuple(history),
    )


def apply_outcome_regressor_numpy(
    result: StrikerOutcomeRegressorResult, observations: np.ndarray
) -> np.ndarray:
    """Return per-action predicted log1p penalized goal distances."""
    hidden = (
        np.asarray(observations, dtype=np.float32) - result.observation_mean
    ) / result.observation_std
    layers = _dense_arrays(result.params)
    for kernel, bias in layers[:-1]:
        hidden = np.tanh(hidden @ kernel + bias)
    kernel, bias = layers[-1]
    normalized = hidden @ kernel + bias
    return normalized * result.target_std + result.target_mean


def export_outcome_regressor_onnx(
    result: StrikerOutcomeRegressorResult, output_path: Path
) -> None:
    """Export portable per-action log-distance predictions."""
    initializers = [
        numpy_helper.from_array(result.observation_mean, name="observation_mean"),
        numpy_helper.from_array(result.observation_std, name="observation_std"),
        numpy_helper.from_array(result.target_mean, name="target_mean"),
        numpy_helper.from_array(result.target_std, name="target_std"),
    ]
    nodes = [
        helper.make_node("Sub", ["observations", "observation_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "observation_std"], ["normalized"]),
    ]
    previous = "normalized"
    layers = _dense_arrays(result.params)
    for index, (kernel, bias) in enumerate(layers):
        kernel_name = f"dense_{index}_kernel"
        bias_name = f"dense_{index}_bias"
        matmul_name = f"dense_{index}_matmul"
        add_name = f"dense_{index}_add"
        initializers.extend(
            [
                numpy_helper.from_array(kernel, name=kernel_name),
                numpy_helper.from_array(bias, name=bias_name),
            ]
        )
        nodes.append(helper.make_node("MatMul", [previous, kernel_name], [matmul_name]))
        nodes.append(helper.make_node("Add", [matmul_name, bias_name], [add_name]))
        if index < len(layers) - 1:
            output_name = f"hidden_{index}"
            nodes.append(helper.make_node("Tanh", [add_name], [output_name]))
            previous = output_name
        else:
            nodes.append(
                helper.make_node(
                    "Mul", [add_name, "target_std"], ["scaled_log_distance"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Add",
                    ["scaled_log_distance", "target_mean"],
                    ["predicted_log1p_goal_distance"],
                )
            )
    graph = helper.make_graph(
        nodes,
        "striker_outcome_regressor",
        [
            helper.make_tensor_value_info(
                "observations", TensorProto.FLOAT, [1, result.observation_size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "predicted_log1p_goal_distance",
                TensorProto.FLOAT,
                [1, len(result.action_prior_indices)],
            )
        ],
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


def verify_outcome_regressor_onnx(
    result: StrikerOutcomeRegressorResult,
    model_path: Path,
    observations: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(
        observations[: min(256, len(observations))], dtype=np.float32
    )
    expected = apply_outcome_regressor_numpy(result, selected)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = np.concatenate(
        [session.run(None, {"observations": row[None, :]})[0] for row in selected],
        axis=0,
    )
    error = np.abs(actual - expected)
    return {
        "samples": float(selected.shape[0]),
        "maximum_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error)),
    }
