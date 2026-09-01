from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.optimize_soccer_motion_teacher_corpus import (
    aggregate_teacher_datasets,
    build_motion_command,
    require_external_path,
)


def test_motion_command_binds_outputs_and_seed(tmp_path):
    command = build_motion_command(
        corpus_root=Path("/data/corpus"),
        relative_path="soccer/a.t1.npz",
        checkpoint=Path("/runs/checkpoint"),
        profile="soccer_motion_residual_v3",
        phase_samples=8,
        population=64,
        generations=8,
        seed=91,
        motion_dir=tmp_path / "motion-00",
    )

    assert command[0]
    assert "soccer/a.t1.npz" in command
    assert command[command.index("--seed") + 1] == "91"
    assert command[command.index("--dataset-output") + 1].endswith(
        "motion-00/teacher-dataset.npz"
    )


def test_aggregate_teacher_datasets_verifies_and_combines(tmp_path):
    reports = []
    for motion in range(2):
        path = tmp_path / f"teacher-{motion}.npz"
        np.savez_compressed(
            path,
            observation=np.full((2, 3), motion, dtype=np.float32),
            base_action=np.zeros((2, 2), dtype=np.float32),
            teacher_action=np.ones((2, 2), dtype=np.float32),
            qpos=np.zeros((2, 4), dtype=np.float64),
            split=np.array([0, 1], dtype=np.int8),
            motion=np.full(2, motion, dtype=np.int16),
            start_frame=np.array([0, 1], dtype=np.int32),
            reference_frame=np.array([1, 2], dtype=np.int32),
        )
        import hashlib

        reports.append(
            {"dataset": str(path), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )

    summary = aggregate_teacher_datasets(reports, tmp_path / "combined.npz")

    assert summary["frames"] == 4
    assert summary["motions"] == 2
    assert summary["train_frames"] == 2
    assert summary["validation_frames"] == 2


def test_external_path_rejects_repository_path():
    with pytest.raises(ValueError, match="outside"):
        require_external_path(Path(__file__).resolve(), "run directory")
