from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tools.evaluate_soccer_motion_cpu import (
    _load_excluded_teacher_starts,
    _start_frames,
)


def test_start_frames_excludes_teacher_phases():
    starts = _start_frames(
        100,
        10,
        10,
        excluded={0, 20, 50, 90},
    )

    assert starts == [10, 30, 40, 60, 70, 80]


def test_load_excluded_teacher_starts_deduplicates_rows(tmp_path):
    path = tmp_path / "teacher.npz"
    np.savez_compressed(
        path,
        motion=np.array([0, 0, 1, 1], dtype=np.int16),
        start_frame=np.array([5, 5, 7, 9], dtype=np.int32),
    )

    starts, digest = _load_excluded_teacher_starts(path, motion_count=2)

    assert starts == {0: {5}, 1: {7, 9}}
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_excluded_teacher_starts_rejects_unknown_motion(tmp_path):
    path = tmp_path / "teacher.npz"
    np.savez_compressed(
        path,
        motion=np.array([2], dtype=np.int16),
        start_frame=np.array([5], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="out-of-range"):
        _load_excluded_teacher_starts(path, motion_count=2)
