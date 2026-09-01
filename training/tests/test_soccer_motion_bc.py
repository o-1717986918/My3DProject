from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from my3d_rl.soccer_motion_bc import (
    action_error_metrics,
    dagger_row_mask,
    load_soccer_motion_teacher_dataset,
    motion_balanced_indices,
    validate_bc_data_manifest,
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


def test_validate_bc_data_manifest_accepts_bound_dagger_aggregate(tmp_path):
    dataset = tmp_path / "dagger.npz"
    dataset.write_bytes(b"dataset")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "k1_b_exact_cpu_soccer_motion_dagger_collection",
                "output_dataset_sha256": hashlib.sha256(b"dataset").hexdigest(),
                "student_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    kind, path, payload = validate_bc_data_manifest(
        dataset,
        base_checkpoint=checkpoint,
        dagger_manifest=manifest,
    )

    assert kind == "dagger_aggregate"
    assert path == manifest
    assert payload["status"] == "complete"


def test_validate_bc_data_manifest_accepts_cross_fitted_dagger(tmp_path):
    dataset = tmp_path / "dagger.npz"
    dataset.write_bytes(b"dataset")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "k1_d_cross_fitted_soccer_motion_dagger_collection",
                "output_dataset_sha256": hashlib.sha256(b"dataset").hexdigest(),
                "student_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    kind, unused_path, unused_payload = validate_bc_data_manifest(
        dataset,
        base_checkpoint=checkpoint,
        dagger_manifest=manifest,
    )

    assert kind == "dagger_aggregate"


def test_validate_bc_data_manifest_rejects_wrong_dagger_checkpoint(tmp_path):
    dataset = tmp_path / "dagger.npz"
    dataset.write_bytes(b"dataset")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "k1_b_exact_cpu_soccer_motion_dagger_collection",
                "output_dataset_sha256": hashlib.sha256(b"dataset").hexdigest(),
                "student_checkpoint": str(tmp_path / "another-checkpoint"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mismatched"):
        validate_bc_data_manifest(
            dataset,
            base_checkpoint=checkpoint,
            dagger_manifest=manifest,
        )


def test_dagger_row_mask_binds_motion_and_start_and_frame_count():
    manifest = {
        "purpose": "k1_b_exact_cpu_soccer_motion_dagger_collection",
        "dagger_frames": 3,
        "per_motion": [
            {"motion": 0, "start_frames": [5]},
            {"motion": 1, "start_frames": [7]},
        ],
    }

    mask = dagger_row_mask(
        np.array([0, 0, 1, 1, 1]),
        np.array([1, 5, 7, 7, 9]),
        manifest,
    )

    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_cross_fitted_dagger_row_mask_uses_appended_frame_boundary():
    manifest = {
        "purpose": "k1_d_cross_fitted_soccer_motion_dagger_collection",
        "source_frames": 3,
        "dagger_frames": 2,
    }

    mask = dagger_row_mask(
        np.array([0, 0, 1, 0, 1]),
        np.array([5, 8, 7, 5, 7]),
        manifest,
    )

    np.testing.assert_array_equal(mask, [False, False, False, True, True])
