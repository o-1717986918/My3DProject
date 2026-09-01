from __future__ import annotations

import numpy as np
import pytest

from my3d_rl.soccer_motion_bc import (
    action_error_metrics,
    load_soccer_motion_teacher_dataset,
    motion_balanced_indices,
)
from my3d_rl.contract import load_policy_contract
from tools.train_soccer_motion_bc import DEFAULT_CONTRACT


def _dataset(path):
    np.savez_compressed(
        path,
        observation=np.zeros((8, 4), dtype=np.float32),
        base_action=np.zeros((8, 2), dtype=np.float32),
        teacher_action=np.full((8, 2), 0.25, dtype=np.float32),
        split=np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8),
        motion=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16),
        start_frame=np.arange(8),
        reference_frame=np.arange(8) + 1,
    )


def test_load_teacher_dataset_and_balanced_sampler(tmp_path):
    path = tmp_path / "teacher.npz"
    _dataset(path)
    data = load_soccer_motion_teacher_dataset(
        path, observation_size=4, action_size=2
    )
    indices = motion_balanced_indices(
        np.random.default_rng(7), data["motion"], data["split"], batch_size=1000
    )

    assert np.all(data["split"][indices] == 0)
    counts = np.bincount(data["motion"][indices], minlength=2)
    assert abs(int(counts[0]) - int(counts[1])) < 100


def test_teacher_dataset_rejects_action_contract_violation(tmp_path):
    path = tmp_path / "bad.npz"
    _dataset(path)
    with np.load(path) as archive:
        values = {name: archive[name] for name in archive.files}
    values["teacher_action"][0, 0] = 1.1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="contract"):
        load_soccer_motion_teacher_dataset(path, observation_size=4, action_size=2)


def test_action_error_metrics_are_reported_per_motion():
    prediction = np.array([[0.0, 0.0], [0.5, 0.5]])
    teacher = np.array([[0.0, 0.0], [1.0, 1.0]])
    base = np.zeros_like(prediction)
    metrics = action_error_metrics(
        prediction, teacher, base, np.array([0, 1], dtype=np.int16)
    )

    assert metrics["teacher_mse"] == pytest.approx(0.125)
    assert metrics["base_mse"] == pytest.approx(0.125)
    assert len(metrics["per_motion"]) == 2


def test_bc_default_contract_matches_residual_v3_profile():
    assert load_policy_contract(DEFAULT_CONTRACT).policy_name == "soccer_motion_policy_v2"
