import hashlib
import json

import numpy as np
import pytest

from tools.train_striker_walk_clone import _load_clone_dataset


def test_clone_dataset_is_bound_to_manifest_and_direct_action_bounds(tmp_path):
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        teacher_observation=np.zeros((4, 138), dtype=np.float32),
        direct_joint_delta=np.zeros((4, 23), dtype=np.float32),
        split=np.array([0, 0, 1, 1], dtype=np.int8),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "apollo_walk_to_privileged_striker_actor_clone_corpus",
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    data, unused = _load_clone_dataset(
        dataset, manifest, observation_size=138, action_size=23
    )
    assert data["observation"].shape == (4, 138)

    with np.load(dataset) as archive:
        observations = archive["teacher_observation"]
    np.savez_compressed(
        dataset,
        teacher_observation=observations,
        direct_joint_delta=np.full((4, 23), 1.01, dtype=np.float32),
        split=np.array([0, 0, 1, 1], dtype=np.int8),
    )
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "apollo_walk_to_privileged_striker_actor_clone_corpus",
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bounds"):
        _load_clone_dataset(
            dataset, manifest, observation_size=138, action_size=23
        )
