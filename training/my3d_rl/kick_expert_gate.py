"""Low-rank release-time gate for parameterized kick experts.

The gate deliberately makes exactly one expert choice at action release.  The
selected closed-loop kick prototype then owns the complete motion; this avoids
the destructive frame-to-frame averaging seen in single-head behavior cloning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


FEATURE_PROFILES = {
    # Proprioception, previous Walk action, and current ball kinematics.  The
    # remaining command fields are constant in a single-condition corpus.
    "physical_v1": np.arange(81, dtype=np.int32),
    "proprioception_v1": np.arange(75, dtype=np.int32),
}


def kick_strict_success_margin(
    contact: np.ndarray,
    fall: np.ndarray,
    range_error_m: np.ndarray,
    lateral_error_m: np.ndarray,
    speed_error_mps: np.ndarray,
) -> np.ndarray:
    """Return a continuous version of the exact 3.5 m success contract."""
    touched = np.asarray(contact, dtype=bool)
    fallen = np.asarray(fall, dtype=bool)
    range_error = np.asarray(range_error_m, dtype=np.float32)
    lateral_error = np.asarray(lateral_error_m, dtype=np.float32)
    speed_error = np.asarray(speed_error_mps, dtype=np.float32)
    if not (
        touched.shape
        == fallen.shape
        == range_error.shape
        == lateral_error.shape
        == speed_error.shape
    ):
        raise ValueError("strict kick margin arrays are misaligned")
    if not (
        np.isfinite(range_error).all()
        and np.isfinite(lateral_error).all()
        and np.isfinite(speed_error).all()
    ):
        raise ValueError("strict kick margin arrays must be finite")
    margin = np.minimum.reduce(
        [
            (0.5 - range_error) / 0.5,
            (0.5 - lateral_error) / 0.5,
            1.0 - speed_error,
        ]
    )
    margin = np.where(touched, margin, -2.0)
    return np.where(fallen, margin - 4.0, margin).astype(np.float32)


def kick_forward_drive_success(
    contact: np.ndarray,
    maximum_progress_m: np.ndarray,
    lateral_error_m: np.ndarray,
    maximum_directional_speed_mps: np.ndarray,
) -> np.ndarray:
    """Classify a useful strong forward action independently of falling.

    Falling is reported and costed separately instead of erasing a useful
    touch.  This is a forward-drive/clearance screen, not a precise pass.
    """
    touched = np.asarray(contact, dtype=bool)
    progress = np.asarray(maximum_progress_m, dtype=np.float32)
    lateral = np.asarray(lateral_error_m, dtype=np.float32)
    speed = np.asarray(maximum_directional_speed_mps, dtype=np.float32)
    if not (touched.shape == progress.shape == lateral.shape == speed.shape):
        raise ValueError("forward drive arrays are misaligned")
    if not (
        np.isfinite(progress).all()
        and np.isfinite(lateral).all()
        and np.isfinite(speed).all()
    ):
        raise ValueError("forward drive arrays must be finite")
    return touched & (progress >= 2.5) & (np.abs(lateral) <= 1.0) & (speed >= 1.5)


@dataclass(frozen=True)
class KickExpertGateResult:
    """Portable parameters for a state-to-expert physical-utility predictor."""

    feature_indices: np.ndarray
    prototype_indices: tuple[int, ...]
    observation_mean: np.ndarray
    observation_std: np.ndarray
    target_mean: np.ndarray
    latent_basis: np.ndarray
    latent_weights: np.ndarray
    latent_rank: int
    ridge: float


def feature_indices(profile: str) -> np.ndarray:
    """Return a defensive copy of one documented observation profile."""
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown kick expert feature profile: {profile}")
    return FEATURE_PROFILES[profile].copy()


def fit_kick_expert_gate(
    observations: np.ndarray,
    physical_score: np.ndarray,
    fit_rows: np.ndarray,
    *,
    prototype_indices: tuple[int, ...],
    feature_indices: np.ndarray,
    latent_rank: int,
    ridge: float,
) -> KickExpertGateResult:
    """Fit a ridge-regularized low-rank expert-utility model.

    The outcome matrix is factorized only on fitting rows.  A linear map from
    normalized release state to its low-dimensional outcome coordinates then
    predicts all expert utilities.  This shares evidence across experts without
    blending their trajectories.
    """
    states = np.asarray(observations, dtype=np.float64)
    scores = np.asarray(physical_score, dtype=np.float64)
    rows = np.asarray(fit_rows, dtype=bool)
    selected = tuple(int(value) for value in prototype_indices)
    features = np.asarray(feature_indices, dtype=np.int32)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("kick expert observations must be a non-empty matrix")
    if scores.ndim != 2 or scores.shape[1] != states.shape[0]:
        raise ValueError("kick expert score matrix is misaligned")
    if rows.shape != (states.shape[0],) or np.count_nonzero(rows) < 2:
        raise ValueError("kick expert fitting rows are invalid")
    if (
        not selected
        or len(set(selected)) != len(selected)
        or min(selected) < 0
        or max(selected) >= scores.shape[0]
    ):
        raise ValueError("kick expert prototype indices are invalid")
    if (
        features.ndim != 1
        or features.size < 1
        or len(set(features.tolist())) != features.size
        or np.any(features < 0)
        or np.any(features >= states.shape[1])
    ):
        raise ValueError("kick expert feature indices are invalid")
    maximum_rank = min(np.count_nonzero(rows) - 1, len(selected))
    if latent_rank < 1 or latent_rank > maximum_rank:
        raise ValueError("kick expert latent rank is invalid")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("kick expert ridge must be finite and positive")
    if not np.isfinite(states).all() or not np.isfinite(scores).all():
        raise ValueError("kick expert inputs must be finite")

    selected_states = states[:, features]
    observation_mean = selected_states[rows].mean(axis=0)
    observation_std = np.maximum(selected_states[rows].std(axis=0), 1.0e-3)
    normalized = (selected_states - observation_mean) / observation_std
    targets = scores[np.asarray(selected), :][:, rows].T
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    left, singular, right = np.linalg.svd(centered_targets, full_matrices=False)
    latent_targets = left[:, :latent_rank] * singular[:latent_rank]
    latent_basis = right[:latent_rank]

    design = np.concatenate(
        [normalized[rows], np.ones((np.count_nonzero(rows), 1))], axis=1
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    latent_weights = np.linalg.solve(
        design.T @ design + ridge * regularizer,
        design.T @ latent_targets,
    )
    return KickExpertGateResult(
        feature_indices=features,
        prototype_indices=selected,
        observation_mean=observation_mean.astype(np.float32),
        observation_std=observation_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        latent_basis=latent_basis.astype(np.float32),
        latent_weights=latent_weights.astype(np.float32),
        latent_rank=latent_rank,
        ridge=float(ridge),
    )


def apply_kick_expert_gate(
    result: KickExpertGateResult, observations: np.ndarray
) -> np.ndarray:
    """Predict one physical-utility value per retained kick expert."""
    states = np.asarray(observations, dtype=np.float32)
    if states.ndim != 2 or (
        states.shape[1] <= int(np.max(result.feature_indices))
    ):
        raise ValueError("kick expert inference observations are invalid")
    selected = states[:, result.feature_indices]
    normalized = (selected - result.observation_mean) / result.observation_std
    design = np.concatenate(
        [normalized, np.ones((normalized.shape[0], 1), dtype=np.float32)], axis=1
    )
    latent = design @ result.latent_weights
    return latent @ result.latent_basis + result.target_mean


def expert_choice_metrics(
    result: KickExpertGateResult,
    predicted_score: np.ndarray,
    success: np.ndarray,
    fall: np.ndarray,
    physical_score: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    """Evaluate release-time expert choices without outcome peeking."""
    predictions = np.asarray(predicted_score, dtype=np.float64)
    succeeded = np.asarray(success, dtype=bool)
    fallen = np.asarray(fall, dtype=bool)
    utility = np.asarray(physical_score, dtype=np.float64)
    selected_rows = np.asarray(rows, dtype=bool)
    if (
        predictions.shape
        != (selected_rows.size, len(result.prototype_indices))
        or succeeded.shape != fallen.shape
        or succeeded.shape != utility.shape
        or succeeded.shape[1] != selected_rows.size
    ):
        raise ValueError("kick expert metric arrays are misaligned")
    row_indices = np.flatnonzero(selected_rows)
    local_choices = np.argmax(predictions[selected_rows], axis=1)
    choices = np.asarray(result.prototype_indices, dtype=np.int32)[local_choices]
    chosen_success = succeeded[choices, row_indices]
    chosen_fall = fallen[choices, row_indices]
    chosen_score = utility[choices, row_indices]
    return {
        "entries": int(row_indices.size),
        "successes": int(chosen_success.sum()),
        "success_rate": float(chosen_success.mean()) if row_indices.size else 0.0,
        "falls": int(chosen_fall.sum()),
        "fall_rate": float(chosen_fall.mean()) if row_indices.size else 0.0,
        "mean_physical_score": (
            float(chosen_score.mean()) if row_indices.size else float("nan")
        ),
        "used_experts": int(np.unique(choices).size),
        "prototype_indices": choices.astype(int).tolist(),
    }


def export_kick_expert_gate_onnx(
    result: KickExpertGateResult, output_path: Path, *, observation_size: int
) -> None:
    """Export the low-rank gate; the runtime performs one ArgMax at release."""
    if observation_size <= int(np.max(result.feature_indices)):
        raise ValueError("ONNX observation size does not cover gate features")
    gather_indices = result.feature_indices.astype(np.int64)
    latent_kernel = result.latent_weights[:-1]
    latent_bias = result.latent_weights[-1]
    initializers = [
        numpy_helper.from_array(gather_indices, name="feature_indices"),
        numpy_helper.from_array(result.observation_mean, name="observation_mean"),
        numpy_helper.from_array(result.observation_std, name="observation_std"),
        numpy_helper.from_array(latent_kernel, name="latent_kernel"),
        numpy_helper.from_array(latent_bias, name="latent_bias"),
        numpy_helper.from_array(result.latent_basis, name="latent_basis"),
        numpy_helper.from_array(result.target_mean, name="target_mean"),
    ]
    nodes = [
        helper.make_node(
            "Gather", ["observations", "feature_indices"], ["selected"], axis=1
        ),
        helper.make_node("Sub", ["selected", "observation_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "observation_std"], ["normalized"]),
        helper.make_node("MatMul", ["normalized", "latent_kernel"], ["latent_mm"]),
        helper.make_node("Add", ["latent_mm", "latent_bias"], ["latent"]),
        helper.make_node("MatMul", ["latent", "latent_basis"], ["score_mm"]),
        helper.make_node("Add", ["score_mm", "target_mean"], ["expert_scores"]),
    ]
    graph = helper.make_graph(
        nodes,
        "kick_expert_gate",
        [
            helper.make_tensor_value_info(
                "observations", TensorProto.FLOAT, [1, observation_size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "expert_scores",
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


def verify_kick_expert_gate_onnx(
    result: KickExpertGateResult,
    model_path: Path,
    observations: np.ndarray,
) -> dict[str, float]:
    """Compare portable ONNX outputs with the NumPy reference."""
    selected = np.asarray(
        observations[: min(256, len(observations))], dtype=np.float32
    )
    expected = apply_kick_expert_gate(result, selected)
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
