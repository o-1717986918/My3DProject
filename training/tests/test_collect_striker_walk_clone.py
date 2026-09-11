import numpy as np
import pytest

from tools.collect_striker_walk_clone import (
    _external_new_directory,
    _split_by_environment,
)


def test_split_keeps_each_environment_on_one_side():
    environment_ids = np.array([0, 1, 0, 4, 5, 4], dtype=np.int32)

    split = _split_by_environment(
        environment_ids, validation_folds=5, validation_fold_index=0
    )

    assert split.tolist() == [1, 0, 1, 0, 1, 0]


def test_collection_requires_a_new_external_directory(tmp_path):
    external = tmp_path / "new-run"
    assert _external_new_directory(external) == external.resolve()

    external.mkdir()
    with pytest.raises(FileExistsError):
        _external_new_directory(external)
