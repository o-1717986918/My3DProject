"""Causal switch-time and prototype selection for exact-CPU kick banks."""

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


class KickSwitchSelector(nn.Module):
    """Predict per-prototype physical success from the current causal state."""

    output_size: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        hidden = jnp.tanh(nn.Dense(256)(observations))
        hidden = jnp.tanh(nn.Dense(128)(hidden))
        return nn.Dense(self.output_size)(hidden)


@dataclass(frozen=True)
class KickSwitchSelectorResult:
    params: Any
    observation_size: int
    prototype_indices: tuple[int, ...]
    observation_mean: np.ndarray
    observation_std: np.ndarray
    fit_rollout_ids: tuple[int, ...]
    calibration_rollout_ids: tuple[int, ...]
    positive_weights: np.ndarray
    fit_loss: float
    calibration_loss: float
    history: tuple[dict[str, float], ...]


def build_causal_sequence_features(
    observations: np.ndarray,
    rollout_ids: np.ndarray,
    confirmation_cycles: np.ndarray,
    *,
    cycle_normalizer: float,
) -> np.ndarray:
    """Add the first aligned state and elapsed alignment count to each frame."""
    states = np.asarray(observations, dtype=np.float32)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    cycles = np.asarray(confirmation_cycles, dtype=np.float32)
    if (
        states.ndim != 2
        or states.shape[0] < 1
        or ids.shape != (states.shape[0],)
        or cycles.shape != ids.shape
    ):
        raise ValueError("causal sequence observations, IDs and cycles are misaligned")
    if (
        not np.isfinite(states).all()
        or not np.isfinite(cycles).all()
        or np.any(cycles < 1.0)
        or not np.isfinite(cycle_normalizer)
        or cycle_normalizer <= 0.0
    ):
        raise ValueError("causal sequence features require finite positive values")
    anchors = np.empty_like(states)
    for rollout_id in np.unique(ids):
        rows = np.flatnonzero(ids == rollout_id)
        order = rows[np.argsort(cycles[rows], kind="stable")]
        if np.unique(cycles[order]).size != order.size:
            raise ValueError("a causal sequence contains duplicate confirmation cycles")
        anchors[rows] = states[order[0]]
    elapsed = (cycles / float(cycle_normalizer)).reshape(-1, 1)
    return np.concatenate([states, anchors, elapsed], axis=1).astype(np.float32)


def grouped_fit_calibration_split(
    rollout_ids: np.ndarray,
    train_rows: np.ndarray,
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split only training rollouts, keeping every adjacent frame together."""
    ids = np.asarray(rollout_ids, dtype=np.int64)
    eligible = np.asarray(train_rows, dtype=bool)
    if ids.ndim != 1 or eligible.shape != ids.shape:
        raise ValueError("rollout IDs and training rows must be aligned vectors")
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration fraction must be in (0, 0.5)")
    train_ids = np.unique(ids[eligible])
    if train_ids.size < 3:
        raise ValueError("selector fitting requires at least three training rollouts")
    leaked = [
        int(rollout_id)
        for rollout_id in train_ids
        if np.any(~eligible[ids == rollout_id])
    ]
    if leaked:
        raise ValueError("training rows leak complete rollouts across partitions")
    order = np.random.default_rng(seed).permutation(train_ids)
    calibration_count = int(round(train_ids.size * calibration_fraction))
    calibration_count = min(max(calibration_count, 1), train_ids.size - 1)
    calibration = tuple(sorted(int(value) for value in order[:calibration_count]))
    fit = tuple(sorted(int(value) for value in order[calibration_count:]))
    return fit, calibration


def _balanced_group_rows(
    rollout_ids: np.ndarray, selected_ids: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    rows = [np.flatnonzero(rollout_ids == rollout_id) for rollout_id in selected_ids]
    if not rows or any(values.size < 1 for values in rows):
        raise ValueError("every selector rollout must contain at least one row")
    maximum = max(values.size for values in rows)
    padded = np.zeros((len(rows), maximum), dtype=np.int64)
    lengths = np.asarray([values.size for values in rows], dtype=np.int64)
    for index, values in enumerate(rows):
        padded[index, : values.size] = values
    return padded, lengths


def _binary_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(
        jnp.exp(-jnp.abs(logits))
    )


def train_switch_selector(
    observations: np.ndarray,
    success: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    *,
    prototype_indices: tuple[int, ...],
    fit_rollout_ids: tuple[int, ...],
    calibration_rollout_ids: tuple[int, ...],
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    fall_weight: float = 8.0,
) -> KickSwitchSelectorResult:
    """Fit a group-balanced multi-label physical-success predictor."""
    states = np.asarray(observations, dtype=np.float32)
    labels = np.asarray(success, dtype=np.float32)
    unsafe = np.asarray(fall, dtype=np.float32)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    selected = tuple(int(value) for value in prototype_indices)
    fit_ids = tuple(int(value) for value in fit_rollout_ids)
    calibration_ids = tuple(int(value) for value in calibration_rollout_ids)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("selector observations must be a non-empty matrix")
    if (
        labels.ndim != 2
        or unsafe.shape != labels.shape
        or labels.shape[1] != states.shape[0]
        or ids.shape != (states.shape[0],)
    ):
        raise ValueError("selector labels, falls, observations and IDs are misaligned")
    if not selected or min(selected) < 0 or max(selected) >= labels.shape[0]:
        raise ValueError("prototype indices are empty or out of range")
    if len(set(selected)) != len(selected):
        raise ValueError("prototype indices must be unique")
    if not fit_ids or not calibration_ids or set(fit_ids) & set(calibration_ids):
        raise ValueError("fit and calibration rollout groups must be disjoint")
    available_ids = set(int(value) for value in np.unique(ids))
    if not set(fit_ids) | set(calibration_ids) <= available_ids:
        raise ValueError("selector split references unavailable rollouts")
    if steps < 1 or batch_size < 1 or learning_rate <= 0.0 or fall_weight < 0.0:
        raise ValueError("selector optimization settings are invalid")
    if (
        not np.isfinite(states).all()
        or not set(np.unique(labels).tolist()) <= {0.0, 1.0}
        or not set(np.unique(unsafe).tolist()) <= {0.0, 1.0}
    ):
        raise ValueError("selector inputs must be finite and labels must be binary")

    fit_mask = np.isin(ids, fit_ids)
    calibration_mask = np.isin(ids, calibration_ids)
    fit_labels = labels[np.asarray(selected), :][:, fit_mask].T
    fit_falls = unsafe[np.asarray(selected), :][:, fit_mask].T
    observation_mean = states[fit_mask].mean(axis=0)
    observation_std = np.maximum(states[fit_mask].std(axis=0), 1.0e-3)
    normalized = (states - observation_mean) / observation_std
    positives = fit_labels.sum(axis=0)
    negative = fit_labels.shape[0] - positives
    positive_weights = np.clip(negative / np.maximum(positives, 1.0), 1.0, 20.0)
    group_rows, group_lengths = _balanced_group_rows(ids, fit_ids)

    model = KickSwitchSelector(output_size=len(selected))
    params = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, states.shape[1]), dtype=jnp.float32),
    )["params"]
    optimizer = optax.adamw(learning_rate, weight_decay=1.0e-6)
    optimizer_state = optimizer.init(params)
    jax_positive_weights = jnp.asarray(positive_weights, dtype=jnp.float32)

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        batch_observations: jax.Array,
        batch_labels: jax.Array,
        batch_falls: jax.Array,
    ) -> tuple[Any, Any, jax.Array]:
        def loss_fn(candidate_params: Any) -> jax.Array:
            logits = model.apply({"params": candidate_params}, batch_observations)
            element_weights = jnp.where(
                batch_labels > 0.5, jax_positive_weights, 1.0
            )
            element_weights = element_weights * (
                1.0 + fall_weight * batch_falls
            )
            losses = _binary_cross_entropy(logits, batch_labels)
            return jnp.sum(losses * element_weights) / jnp.sum(element_weights)

        loss, gradients = jax.value_and_grad(loss_fn)(current_params)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        return (
            optax.apply_updates(current_params, updates),
            next_optimizer_state,
            loss,
        )

    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    report_interval = max(1, steps // 20)
    selected_array = np.asarray(selected)
    for step in range(steps):
        group_indices = rng.integers(0, len(fit_ids), size=batch_size)
        offsets = (
            rng.random(batch_size) * group_lengths[group_indices]
        ).astype(np.int64)
        row_indices = group_rows[group_indices, offsets]
        params, optimizer_state, loss = train_step(
            params,
            optimizer_state,
            jnp.asarray(normalized[row_indices]),
            jnp.asarray(labels[selected_array, :][:, row_indices].T),
            jnp.asarray(unsafe[selected_array, :][:, row_indices].T),
        )
        if step % report_interval == 0 or step + 1 == steps:
            history.append({"step": float(step + 1), "batch_loss": float(loss)})

    def loss_for(mask: np.ndarray) -> float:
        logits = model.apply({"params": params}, jnp.asarray(normalized[mask]))
        target = jnp.asarray(labels[selected_array, :][:, mask].T)
        return float(jnp.mean(_binary_cross_entropy(logits, target)))

    return KickSwitchSelectorResult(
        params=params,
        observation_size=int(states.shape[1]),
        prototype_indices=selected,
        observation_mean=observation_mean.astype(np.float32),
        observation_std=observation_std.astype(np.float32),
        fit_rollout_ids=fit_ids,
        calibration_rollout_ids=calibration_ids,
        positive_weights=positive_weights.astype(np.float32),
        fit_loss=loss_for(fit_mask),
        calibration_loss=loss_for(calibration_mask),
        history=tuple(history),
    )


def apply_switch_selector(
    result: KickSwitchSelectorResult, observations: np.ndarray
) -> np.ndarray:
    normalized = (
        np.asarray(observations, dtype=np.float32) - result.observation_mean
    ) / result.observation_std
    model = KickSwitchSelector(output_size=len(result.prototype_indices))
    logits = model.apply({"params": result.params}, jnp.asarray(normalized))
    return np.asarray(jax.nn.sigmoid(logits))


def _dense_arrays(params: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.asarray(params[f"Dense_{index}"]["kernel"], dtype=np.float32),
            np.asarray(params[f"Dense_{index}"]["bias"], dtype=np.float32),
        )
        for index in range(3)
    ]


def apply_switch_selector_numpy(
    result: KickSwitchSelectorResult, observations: np.ndarray
) -> np.ndarray:
    hidden = (
        np.asarray(observations, dtype=np.float32) - result.observation_mean
    ) / result.observation_std
    layers = _dense_arrays(result.params)
    for kernel, bias in layers[:-1]:
        hidden = np.tanh(hidden @ kernel + bias)
    kernel, bias = layers[-1]
    logits = hidden @ kernel + bias
    return 1.0 / (1.0 + np.exp(-logits))


def sequential_policy_metrics(
    success: np.ndarray,
    fall: np.ndarray,
    rollout_ids: np.ndarray,
    confirmation_cycles: np.ndarray,
    probabilities: np.ndarray,
    rows: np.ndarray,
    *,
    prototype_indices: tuple[int, ...],
    threshold: float,
    consecutive_frames: int,
    fallback_prototype_index: int | None = None,
    fallback_confirmation_cycles: int | None = None,
) -> dict[str, Any]:
    """Replay a causal wait-or-kick policy on complete approach sequences."""
    labels = np.asarray(success, dtype=bool)
    unsafe = np.asarray(fall, dtype=bool)
    ids = np.asarray(rollout_ids, dtype=np.int64)
    cycles = np.asarray(confirmation_cycles, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    selected_rows = np.asarray(rows, dtype=bool)
    selected = tuple(int(value) for value in prototype_indices)
    if (
        labels.ndim != 2
        or unsafe.shape != labels.shape
        or ids.shape != cycles.shape
        or ids.shape != selected_rows.shape
        or labels.shape[1] != ids.size
        or scores.shape != (ids.size, len(selected))
    ):
        raise ValueError("sequential selector inputs are misaligned")
    if not selected or min(selected) < 0 or max(selected) >= labels.shape[0]:
        raise ValueError("sequential selector prototypes are out of range")
    if not 0.0 < threshold < 1.0 or consecutive_frames < 1:
        raise ValueError("release threshold and consecutive frames are invalid")
    if (fallback_prototype_index is None) != (
        fallback_confirmation_cycles is None
    ):
        raise ValueError("fallback prototype and cycle must be provided together")
    if fallback_prototype_index is not None and (
        fallback_prototype_index < 0
        or fallback_prototype_index >= labels.shape[0]
        or fallback_confirmation_cycles is None
        or fallback_confirmation_cycles < 1
    ):
        raise ValueError("fallback prototype or confirmation cycle is invalid")

    releases = successes = falls = 0
    decisions: list[dict[str, int | float | bool]] = []
    rollout_values = np.unique(ids[selected_rows])
    for rollout_id in rollout_values:
        sequence = np.flatnonzero(selected_rows & (ids == rollout_id))
        sequence = sequence[np.argsort(cycles[sequence], kind="stable")]
        streak = np.zeros(len(selected), dtype=np.int64)
        decision: dict[str, int | float | bool] | None = None
        for row in sequence:
            eligible = scores[row] >= threshold
            streak = np.where(eligible, streak + 1, 0)
            ready = np.flatnonzero(streak >= consecutive_frames)
            use_fallback = bool(
                ready.size == 0
                and fallback_confirmation_cycles is not None
                and cycles[row] >= fallback_confirmation_cycles
            )
            if ready.size == 0 and not use_fallback:
                continue
            if use_fallback:
                prototype = int(fallback_prototype_index)
                confidence = 0.0
                decision_kind = "fallback"
            else:
                local_prototype = int(ready[np.argmax(scores[row, ready])])
                prototype = selected[local_prototype]
                confidence = float(scores[row, local_prototype])
                decision_kind = "learned"
            did_succeed = bool(labels[prototype, row])
            did_fall = bool(unsafe[prototype, row])
            releases += 1
            successes += int(did_succeed)
            falls += int(did_fall)
            decision = {
                "rollout_id": int(rollout_id),
                "row": int(row),
                "confirmation_cycles": int(cycles[row]),
                "prototype_index": prototype,
                "confidence": confidence,
                "decision_kind": decision_kind,
                "success": did_succeed,
                "fall": did_fall,
            }
            break
        if decision is not None:
            decisions.append(decision)
    total = int(rollout_values.size)
    return {
        "rollouts": total,
        "releases": releases,
        "successes": successes,
        "falls": falls,
        "release_rate": releases / total if total else 0.0,
        "success_rate": successes / total if total else 0.0,
        "release_precision": successes / releases if releases else 0.0,
        "decisions": decisions,
    }


def export_switch_selector_onnx(
    result: KickSwitchSelectorResult, output_path: Path
) -> None:
    """Export normalized current-state probabilities as a portable ONNX graph."""
    initializers = [
        numpy_helper.from_array(result.observation_mean, name="observation_mean"),
        numpy_helper.from_array(result.observation_std, name="observation_std"),
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
        add_name = "logits" if index == len(layers) - 1 else f"dense_{index}_add"
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
    nodes.append(helper.make_node("Sigmoid", ["logits"], ["probabilities"]))
    graph = helper.make_graph(
        nodes,
        "kick_switch_selector",
        [
            helper.make_tensor_value_info(
                "observations", TensorProto.FLOAT, [1, result.observation_size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities",
                TensorProto.FLOAT,
                [1, len(result.prototype_indices)],
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


def verify_switch_selector_onnx(
    result: KickSwitchSelectorResult,
    model_path: Path,
    observations: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(
        observations[: min(256, len(observations))], dtype=np.float32
    )
    expected = apply_switch_selector_numpy(result, selected)
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
