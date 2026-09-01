from pathlib import Path

import numpy as np
import pytest

from my3d_rl.kick_switch_selector import (
    apply_switch_selector_numpy,
    build_causal_sequence_features,
    export_switch_selector_onnx,
    grouped_fit_calibration_split,
    sequential_policy_metrics,
    train_switch_selector,
    verify_switch_selector_onnx,
)


def test_causal_sequence_features_reuse_only_the_first_aligned_state() -> None:
    observations = np.arange(24, dtype=np.float32).reshape(6, 4)
    rollout_ids = np.array([2, 1, 2, 1, 2, 1])
    cycles = np.array([3, 5, 1, 1, 5, 3])

    features = build_causal_sequence_features(
        observations, rollout_ids, cycles, cycle_normalizer=10.0
    )

    assert features.shape == (6, 9)
    np.testing.assert_array_equal(
        features[rollout_ids == 1, 4:8], np.tile(observations[3], (3, 1))
    )
    np.testing.assert_array_equal(
        features[rollout_ids == 2, 4:8], np.tile(observations[2], (3, 1))
    )
    np.testing.assert_allclose(features[:, -1], cycles / 10.0)


def test_grouped_fit_calibration_split_preserves_complete_rollouts() -> None:
    rollout_ids = np.repeat(np.arange(10), 3)
    train_rows = rollout_ids < 8

    fit, calibration = grouped_fit_calibration_split(
        rollout_ids,
        train_rows,
        seed=7,
        calibration_fraction=0.25,
    )

    assert set(fit).isdisjoint(calibration)
    assert set(fit) | set(calibration) == set(range(8))
    with pytest.raises(ValueError, match="leak"):
        grouped_fit_calibration_split(
            rollout_ids,
            train_rows & (np.arange(30) != 1),
            seed=7,
            calibration_fraction=0.25,
        )


def test_sequential_policy_waits_for_consecutive_confidence() -> None:
    success = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    fall = np.zeros_like(success)
    rollout_ids = np.array([10, 10, 10, 20, 20, 20])
    cycles = np.array([1, 3, 5, 1, 3, 5])
    probabilities = np.array(
        [
            [0.8, 0.1],
            [0.9, 0.2],
            [0.95, 0.2],
            [0.2, 0.8],
            [0.2, 0.9],
            [0.2, 0.95],
        ]
    )

    metrics = sequential_policy_metrics(
        success,
        fall,
        rollout_ids,
        cycles,
        probabilities,
        np.ones(6, dtype=bool),
        prototype_indices=(0, 1),
        threshold=0.85,
        consecutive_frames=2,
    )

    assert metrics["rollouts"] == 2
    assert metrics["releases"] == 2
    assert metrics["successes"] == 2
    assert [node["confirmation_cycles"] for node in metrics["decisions"]] == [5, 5]


def test_sequential_policy_uses_fallback_at_the_requested_cycle() -> None:
    metrics = sequential_policy_metrics(
        np.array([[0, 1, 0]], dtype=np.uint8),
        np.zeros((1, 3), dtype=np.uint8),
        np.array([4, 4, 4]),
        np.array([1, 3, 5]),
        np.zeros((3, 1)),
        np.ones(3, dtype=bool),
        prototype_indices=(0,),
        threshold=0.9,
        consecutive_frames=1,
        fallback_prototype_index=0,
        fallback_confirmation_cycles=3,
    )

    assert metrics["successes"] == 1
    assert metrics["decisions"][0]["decision_kind"] == "fallback"
    assert metrics["decisions"][0]["confirmation_cycles"] == 3


def test_selector_export_matches_numpy(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    rollout_ids = np.repeat(np.arange(6), 4)
    observations = rng.normal(size=(24, 9)).astype(np.float32)
    success = np.vstack(
        [
            observations[:, 0] > 0.0,
            observations[:, 1] > 0.0,
            observations[:, 2] > 0.0,
        ]
    ).astype(np.uint8)
    fall = np.zeros_like(success)
    result = train_switch_selector(
        observations,
        success,
        fall,
        rollout_ids,
        prototype_indices=(0, 2),
        fit_rollout_ids=(0, 1, 2, 3),
        calibration_rollout_ids=(4, 5),
        seed=13,
        steps=2,
        batch_size=8,
        learning_rate=1.0e-3,
    )
    model_path = tmp_path / "selector.onnx"
    export_switch_selector_onnx(result, model_path)

    probabilities = apply_switch_selector_numpy(result, observations)
    parity = verify_switch_selector_onnx(result, model_path, observations)

    assert probabilities.shape == (24, 2)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert parity["maximum_absolute_error"] < 2.0e-6
