from pathlib import Path

import numpy as np
import pytest

from my3d_rl.kick_bc import load_teacher_dataset, train_behavior_clone


def test_teacher_dataset_validation_rejects_bad_shapes(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        observations=np.zeros((4, 95), dtype=np.float32),
        actions=np.zeros((4, 23), dtype=np.float32),
        episode_ids=np.zeros(4, dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r"\[N, 96\]"):
        load_teacher_dataset(path)


def test_behavior_clone_requires_condition_level_validation_split():
    dataset = {
        "observations": np.zeros((8, 96), dtype=np.float32),
        "actions": np.zeros((8, 23), dtype=np.float32),
        "episode_ids": np.zeros(8, dtype=np.int32),
    }
    with pytest.raises(ValueError, match="at least two"):
        train_behavior_clone(
            dataset,
            seed=1,
            steps=1,
            batch_size=2,
            learning_rate=1.0e-3,
        )
