import pytest

from tools.view_soccer_motion_policy import select_motion


def test_select_motion_supports_index_and_relative_path():
    paths = ("first.npz", "nested/second.npz")

    assert select_motion(paths, motion_index=1, motion_path=None) == 1
    assert select_motion(paths, motion_index=0, motion_path="nested/second.npz") == 1


def test_select_motion_rejects_unknown_selection():
    paths = ("first.npz",)
    with pytest.raises(ValueError, match="outside"):
        select_motion(paths, motion_index=1, motion_path=None)
    with pytest.raises(ValueError, match="unavailable"):
        select_motion(paths, motion_index=0, motion_path="missing.npz")
