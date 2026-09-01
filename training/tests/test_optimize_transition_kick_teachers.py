from pathlib import Path

import numpy as np

from my3d_rl import load_policy_contract
from my3d_rl.kick_env import TRANSITION_CONTRACT
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec
from training.tools.optimize_transition_kick_teachers import _write_checkpoint


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
