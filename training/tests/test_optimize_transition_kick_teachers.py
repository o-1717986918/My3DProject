from pathlib import Path

import numpy as np

from my3d_rl import load_policy_contract
from my3d_rl.kick_env import TRANSITION_CONTRACT
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec
from training.tools.optimize_transition_kick_teachers import (
    _load_source,
    _sha256,
    _write_checkpoint,
)


def test_captured_observations_are_defensive_copies():
    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(duration_s=0.02, evaluation_duration_s=0.02),
        contract=load_policy_contract(TRANSITION_CONTRACT),
    )
    parameters = np.zeros(14)

    evaluator.rollout(parameters, capture_targets=True)
    first = evaluator.captured_observations
    first.fill(99.0)

    second = evaluator.captured_observations
    assert second.shape == (2, 98)
    assert not np.all(second == 99.0)


def test_captured_actions_and_targets_are_defensive_copies():
    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(duration_s=0.02, evaluation_duration_s=0.02),
        contract=load_policy_contract(TRANSITION_CONTRACT),
    )
    evaluator.rollout(np.zeros(14), capture_targets=True)
    actions = evaluator.captured_actions
    targets = evaluator.captured_targets
    actions.fill(99.0)
    targets.fill(99.0)

    assert not np.all(evaluator.captured_actions == 99.0)
    assert not np.all(evaluator.captured_targets == 99.0)


def test_checkpoint_replace_leaves_no_temporary_file(tmp_path: Path):
    path = tmp_path / "labels.json"
    _write_checkpoint(path, {"status": "running", "labels": []})

    assert path.is_file()
    assert not path.with_suffix(".json.tmp").exists()


def test_single_teacher_parameters_initialize_every_training_phase(tmp_path: Path):
    teacher = tmp_path / "teacher.json"
    teacher.write_text(
        """{
          "purpose": "r1_low_dimensional_kick_teacher",
          "spec": {
            "target_distance_m": 3.5,
            "target_angle_deg": 0.0,
            "requested_ball_speed_mps": 2.2,
            "desired_arrival_speed_mps": 0.8,
            "action_mode": "pass"
          },
          "ball_offset_m": {"x": 0.0, "y": 0.0},
          "parameters": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
          "metrics": {"contact": true}
        }""",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.npz"
    np.savez_compressed(
        corpus,
        qpos=np.zeros((4, 1), dtype=np.float32),
        qvel=np.zeros((4, 1), dtype=np.float32),
        walk_previous_action=np.zeros((4, 23), dtype=np.float32),
        split=np.array([0, 0, 1, 1], dtype=np.uint8),
        rollout_id=np.arange(4, dtype=np.int32),
        phase_bucket=np.array([2, 5, 2, 5], dtype=np.int32),
    )
    corpus.with_suffix(".json").write_text(
        "{\n"
        f'  "npz_sha256": "{_sha256(corpus)}",\n'
        '  "teacher_condition_index": 0,\n'
        '  "phase_bucket_count": 8\n'
        "}\n",
        encoding="utf-8",
    )

    record, arrays, initializers = _load_source(teacher, corpus, None, 60)

    assert record["distance_m"] == 3.5
    assert arrays["walk_previous_action"].shape == (4, 23)
    assert set(initializers) == {2, 5}
    assert all(np.allclose(value, 0.1) for value in initializers.values())
