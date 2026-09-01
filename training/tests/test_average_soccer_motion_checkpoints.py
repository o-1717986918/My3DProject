import numpy as np
import pytest

from tools.average_soccer_motion_checkpoints import (
    _mean_tree,
    _select_normalizer,
    _trees_equal,
)


def test_mean_tree_averages_aligned_floating_parameters():
    first = {"params": {"kernel": np.array([1.0, 3.0], dtype=np.float32)}}
    second = {"params": {"kernel": np.array([3.0, 5.0], dtype=np.float32)}}

    result = _mean_tree([first, second])

    np.testing.assert_array_equal(
        result["params"]["kernel"], np.array([2.0, 4.0], dtype=np.float32)
    )


def test_mean_tree_rejects_different_structures():
    with pytest.raises(ValueError, match="structures differ"):
        _mean_tree([{"a": np.ones(1)}, {"b": np.ones(1)}])


def test_mean_tree_requires_identical_integer_leaves():
    with pytest.raises(ValueError, match="non-floating"):
        _mean_tree([{"count": np.array(1)}, {"count": np.array(2)}])


def test_trees_equal_is_exact():
    first = {"value": np.array([1.0, 2.0], dtype=np.float32)}
    assert _trees_equal(first, first)
    assert not _trees_equal(
        first, {"value": np.array([1.0, 2.001], dtype=np.float32)}
    )


def test_disabled_normalization_copies_retained_base():
    base = {"count": np.array(0)}
    selected, rule = _select_normalizer(
        base,
        [{"count": np.array(10)}, {"count": np.array(20)}],
        normalize_observations=False,
    )

    assert selected is base
    assert rule == "disabled_copied_from_retained_base"


def test_enabled_normalization_rejects_different_statistics():
    with pytest.raises(ValueError, match="normalizers differ"):
        _select_normalizer(
            {"count": np.array(0)},
            [{"count": np.array(1)}],
            normalize_observations=True,
        )
