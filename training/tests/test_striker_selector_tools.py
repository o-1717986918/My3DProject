import json
from pathlib import Path

import numpy as np

from training.tools.build_striker_action_bank_corpus import load_action_reports
from training.tools.analyze_striker_prior_selector import knn_probabilities
from training.tools.train_striker_prior_selector import _metrics
from my3d_rl.striker_outcome_regressor import (
    apply_outcome_regressor_numpy,
    export_outcome_regressor_onnx,
    train_outcome_regressor,
    verify_outcome_regressor_onnx,
)


def _write_report(path: Path, action_index: int, outcomes: list[bool]) -> None:
    path.write_text(
        json.dumps(
            {
                "purpose": "striker_closed_loop_exact_cpu_evaluation",
                "contract": "striker_policy_v1.yaml",
                "contract_sha256": "contract",
                "kick_prior": {"entries": ["bank"]},
                "success_definition": {
                    "goal_radius_m": 0.5,
                    "arrival_speed_tolerance_mps": 0.5,
                    "requires_contact": True,
                },
                "environment_config": {
                    "episode_length": 1000,
                    "fixed_kick_prior_index": action_index,
                },
                "rollouts": [
                    {
                        "seed": 40 + index,
                        "triggered": True,
                        "trigger_observation": [float(index)] * 102,
                        "trigger_walk_last_action": [float(index) / 10.0] * 23,
                        "trigger_privileged_observation": (
                            [float(index)] * 102
                            + [float(index) / 10.0] * 23
                            + [0.0] * 13
                        ),
                        "succeeded": succeeded,
                        "fallen": False,
                        "final_goal_distance_m": float(index + action_index),
                    }
                    for index, succeeded in enumerate(outcomes)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_action_reports_require_counterfactually_identical_trigger_states(
    tmp_path: Path,
):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first, 2, [True, False, True])
    _write_report(second, 5, [False, True, True])

    (
        observations,
        walk_last_action,
        privileged_observation,
        success,
        fall,
        final_goal_distance,
        rollout_ids,
        action_indices,
        evidence,
    ) = (
        load_action_reports([first, second])
    )

    assert observations.shape == (3, 102)
    assert walk_last_action.shape == (3, 23)
    assert privileged_observation.shape == (3, 138)
    assert success.tolist() == [[1, 0, 1], [0, 1, 1]]
    assert not fall.any()
    assert final_goal_distance.shape == (2, 3)
    assert final_goal_distance[0].tolist() == [2.0, 3.0, 4.0]
    assert rollout_ids.tolist() == [40, 41, 42]
    assert action_indices.tolist() == [2, 5]
    assert len(evidence["reports"][0]["sha256"]) == 64


def test_selector_metrics_use_fallback_below_calibrated_confidence():
    success = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    fall = np.zeros_like(success)
    probabilities = np.array(
        [[0.9, 0.1], [0.4, 0.45], [0.1, 0.8]], dtype=np.float32
    )

    metrics = _metrics(
        success,
        fall,
        probabilities,
        np.ones(3, dtype=bool),
        fallback_local_index=1,
        confidence_threshold=0.5,
    )

    assert metrics["successes"] == 3
    assert metrics["falls"] == 0
    assert metrics["learned_decisions"] == 2
    assert metrics["fallback_decisions"] == 1


def test_knn_selector_returns_per_action_success_probabilities():
    observations = np.array([[0.0], [0.1], [1.0], [0.05]], dtype=np.float32)
    success = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    probabilities = knn_probabilities(
        observations,
        success,
        np.array([True, True, True, False]),
        np.array([False, False, False, True]),
        np.array([0]),
        neighbors=2,
    )

    assert probabilities.shape == (1, 2)
    assert probabilities[0, 0] == 1.0
    assert probabilities[0, 1] == 0.0


def test_outcome_regressor_trains_and_exports_portable_action_scores(
    tmp_path: Path,
):
    observations = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(8, 3)
    distances = np.stack(
        [
            np.square(observations[:, 0] - 0.25) + 0.1,
            np.square(observations[:, 0] + 0.25) + 0.1,
        ]
    ).astype(np.float32)
    falls = np.zeros_like(distances, dtype=np.uint8)
    rollout_ids = np.arange(100, 108, dtype=np.int64)
    result = train_outcome_regressor(
        observations,
        distances,
        falls,
        rollout_ids,
        action_prior_indices=(3, 7),
        fit_rollout_ids=(100, 101, 102, 103, 104),
        calibration_rollout_ids=(105, 106),
        seed=19,
        steps=20,
        batch_size=4,
        learning_rate=1.0e-3,
    )
    predictions = apply_outcome_regressor_numpy(result, observations)
    model_path = tmp_path / "outcome.onnx"
    export_outcome_regressor_onnx(result, model_path)
    parity = verify_outcome_regressor_onnx(result, model_path, observations)

    assert predictions.shape == (8, 2)
    assert np.isfinite(predictions).all()
    assert model_path.is_file()
    assert parity["maximum_absolute_error"] < 1.0e-5
