from pathlib import Path

import numpy as np

from my3d_rl.kick_expert_gate import (
    apply_kick_expert_gate,
    expert_choice_metrics,
    export_kick_expert_gate_onnx,
    feature_indices,
    fit_kick_expert_gate,
    kick_forward_drive_success,
    kick_strict_success_margin,
    verify_kick_expert_gate_onnx,
)


def _synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
    observations = np.zeros((24, 98), dtype=np.float32)
    observations[:, 0] = state
    observations[:, 1] = np.square(state)
    # Three experts are optimal on left, center, and right release states.
    centers = np.asarray([-0.75, 0.0, 0.75], dtype=np.float32)
    score = -np.square(state[None, :] - centers[:, None])
    fit_rows = np.ones(24, dtype=bool)
    return observations, score, fit_rows


def test_low_rank_gate_selects_distinct_fixed_experts() -> None:
    observations, score, fit_rows = _synthetic_problem()
    result = fit_kick_expert_gate(
        observations,
        score,
        fit_rows,
        prototype_indices=(0, 1, 2),
        feature_indices=np.asarray([0, 1], dtype=np.int32),
        latent_rank=2,
        ridge=0.01,
    )
    prediction = apply_kick_expert_gate(result, observations)
    choices = np.argmax(prediction, axis=1)
    assert choices[0] == 0
    assert choices[len(choices) // 2] == 1
    assert choices[-1] == 2


def test_gate_metrics_keep_falls_as_a_finite_reported_cost() -> None:
    observations, score, fit_rows = _synthetic_problem()
    result = fit_kick_expert_gate(
        observations,
        score,
        fit_rows,
        prototype_indices=(0, 1, 2),
        feature_indices=np.asarray([0, 1], dtype=np.int32),
        latent_rank=2,
        ridge=0.01,
    )
    prediction = apply_kick_expert_gate(result, observations)
    choices = np.argmax(prediction, axis=1)
    success = np.zeros_like(score, dtype=np.uint8)
    success[choices, np.arange(choices.size)] = 1
    fall = np.zeros_like(success)
    fall[choices[0], 0] = 1
    metrics = expert_choice_metrics(
        result, prediction, success, fall, score, fit_rows
    )
    assert metrics["successes"] == observations.shape[0]
    assert metrics["falls"] == 1
    assert metrics["fall_rate"] == 1.0 / observations.shape[0]


def test_gate_onnx_matches_numpy(tmp_path: Path) -> None:
    observations, score, fit_rows = _synthetic_problem()
    result = fit_kick_expert_gate(
        observations,
        score,
        fit_rows,
        prototype_indices=(0, 1, 2),
        feature_indices=np.asarray([0, 1], dtype=np.int32),
        latent_rank=2,
        ridge=0.01,
    )
    model = tmp_path / "gate.onnx"
    export_kick_expert_gate_onnx(result, model, observation_size=98)
    parity = verify_kick_expert_gate_onnx(result, model, observations)
    assert parity["maximum_absolute_error"] <= 1.0e-6


def test_feature_profiles_are_defensive_and_bounded() -> None:
    physical = feature_indices("physical_v1")
    physical[0] = 80
    assert feature_indices("physical_v1")[0] == 0
    assert feature_indices("physical_v1")[-1] == 80


def test_strict_margin_has_finite_contact_and_fall_costs() -> None:
    margin = kick_strict_success_margin(
        np.asarray([1, 0, 1]),
        np.asarray([0, 0, 1]),
        np.asarray([0.25, 0.25, 0.25]),
        np.asarray([0.25, 0.25, 0.25]),
        np.asarray([0.25, 0.25, 0.25]),
    )
    np.testing.assert_allclose(margin, [0.5, -2.0, -3.5])


def test_forward_drive_does_not_require_zero_falls() -> None:
    success = kick_forward_drive_success(
        np.asarray([1, 1, 0, 1]),
        np.asarray([2.5, 2.49, 4.0, 4.0]),
        np.asarray([1.0, 0.2, 0.2, 1.01]),
        np.asarray([1.5, 3.0, 3.0, 3.0]),
    )
    assert success.tolist() == [True, False, False, False]
