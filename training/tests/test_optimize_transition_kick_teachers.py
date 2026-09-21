from pathlib import Path

import numpy as np

from my3d_rl import load_policy_contract
from my3d_rl.kick_env import TRANSITION_CONTRACT
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec
from training.tools.optimize_transition_kick_teachers import (
    _load_source,
    _map_repair_labels,
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


def test_repair_labels_map_only_identical_states_into_expanded_corpus(
    tmp_path: Path,
):
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source,
        qpos=np.array([[1.0], [2.0]], dtype=np.float32),
        qvel=np.array([[3.0], [4.0]], dtype=np.float32),
        walk_previous_action=np.zeros((2, 23), dtype=np.float32),
        rollout_id=np.array([10, 20], dtype=np.int32),
        split=np.zeros(2, dtype=np.uint8),
    )
    manifest = tmp_path / "labels.json"
    manifest.write_text(
        "{\n"
        '  "purpose": "exact_cpu_per_transition_kick_teacher_labels",\n'
        '  "complete": true,\n'
        '  "teacher_manifest_sha256": "teacher",\n'
        '  "contract_sha256": "contract",\n'
        f'  "transition_corpus": "{source}",\n'
        f'  "transition_corpus_sha256": "{_sha256(source)}",\n'
        '  "labels": [{"corpus_index": 1, "rollout_id": 20, '
        '"trained_success": true, "parameters": [0]}]\n'
        "}\n",
        encoding="utf-8",
    )
    arrays = {
        "qpos": np.array([[2.0], [5.0]], dtype=np.float32),
        "qvel": np.array([[4.0], [6.0]], dtype=np.float32),
        "walk_previous_action": np.zeros((2, 23), dtype=np.float32),
        "rollout_id": np.array([20, 30], dtype=np.int32),
        "split": np.zeros(2, dtype=np.uint8),
    }

    mapped = _map_repair_labels(
        manifest,
        arrays,
        teacher_manifest_sha256="teacher",
        contract_sha256="contract",
    )

    assert list(mapped) == [0]
    assert mapped[0]["rollout_id"] == 20
