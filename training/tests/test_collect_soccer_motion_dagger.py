import json

import numpy as np
import pytest

from my3d_rl.soccer_motion_dagger import sha256
from tools.collect_soccer_motion_dagger import (
    _episode_validation_split,
    _validate_source_dataset_lineage,
)


def test_episode_validation_split_is_stable_and_binary() -> None:
    first = _episode_validation_split(
        seed=20260982, motion=3, start_frame=17, folds=5, fold_index=0
    )
    second = _episode_validation_split(
        seed=20260982, motion=3, start_frame=17, folds=5, fold_index=0
    )

    assert first == second
    assert first in (0, 1)


def test_source_lineage_accepts_selected_teacher_corpus(tmp_path) -> None:
    dataset = tmp_path / "teacher.npz"
    np.savez_compressed(dataset, observation=np.zeros((1, 2)))

    lineage = _validate_source_dataset_lineage(
        source_dataset=dataset,
        selection={"combined_dataset": {"sha256": sha256(dataset)}},
        source_manifest=None,
    )

    assert lineage["kind"] == "selected_teacher_corpus"


def test_source_lineage_requires_and_binds_prior_dagger_manifest(tmp_path) -> None:
    dataset = tmp_path / "dagger.npz"
    np.savez_compressed(dataset, observation=np.zeros((1, 2)))
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "k1_b_exact_cpu_soccer_motion_dagger_collection",
                "output_dataset_sha256": sha256(dataset),
                "git_revision": "abc123",
            }
        ),
        encoding="utf-8",
    )

    lineage = _validate_source_dataset_lineage(
        source_dataset=dataset,
        selection={"combined_dataset": {"sha256": "different"}},
        source_manifest=manifest,
    )

    assert lineage["kind"] == "prior_dagger_aggregate"
    assert lineage["git_revision"] == "abc123"

    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "k1_b_exact_cpu_soccer_motion_dagger_collection",
                "output_dataset_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not bind"):
        _validate_source_dataset_lineage(
            source_dataset=dataset,
            selection={"combined_dataset": {"sha256": "different"}},
            source_manifest=manifest,
        )
