from __future__ import annotations

import hashlib
import numpy as np
import pytest

from my3d_rl.soccer_motion_ball import (
    classify_ball_contacts,
    deterministic_ball_placement_perturbation,
    place_reference_ball_xy,
    select_reference_strike,
)
from tools.evaluate_soccer_motion_ball_cpu import (
    _checkpoint_tree_sha256,
    _validate_output_path,
)


def test_ball_evaluator_uses_canonical_checkpoint_tree_hash(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "z").write_bytes(b"last")
    (checkpoint / "a").write_bytes(b"first")
    expected = hashlib.sha256()
    for name, value in (("a", b"first"), ("z", b"last")):
        expected.update(name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(hashlib.sha256(value).hexdigest().encode("ascii"))
        expected.update(b"\n")

    assert _checkpoint_tree_sha256(checkpoint) == expected.hexdigest()


def test_ball_evaluator_refuses_existing_output(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError, match="already exists"):
        _validate_output_path(output)


def test_ball_placement_perturbation_is_deterministic_and_bounded():
    first = deterministic_ball_placement_perturbation(
        base_seed=99,
        motion=3,
        start_frame=7,
        radius_noise_m=0.02,
        arc_noise_rad=0.1,
    )
    second = deterministic_ball_placement_perturbation(
        base_seed=99,
        motion=3,
        start_frame=7,
        radius_noise_m=0.02,
        arc_noise_rad=0.1,
    )

    assert first == second
    assert 0 <= first.case_seed <= np.iinfo(np.int64).max
    assert abs(first.radius_offset_m) <= 0.02
    assert abs(first.arc_angle_rad) <= 0.1


def test_ball_placement_perturbation_changes_across_cases():
    first = deterministic_ball_placement_perturbation(
        base_seed=99,
        motion=3,
        start_frame=7,
        radius_noise_m=0.02,
        arc_noise_rad=0.1,
    )
    second = deterministic_ball_placement_perturbation(
        base_seed=99,
        motion=3,
        start_frame=8,
        radius_noise_m=0.02,
        arc_noise_rad=0.1,
    )

    assert first.case_seed != second.case_seed
    assert (first.radius_offset_m, first.arc_angle_rad) != (
        second.radius_offset_m,
        second.arc_angle_rad,
    )


def test_place_reference_ball_preserves_restart_endpoint_geometry():
    roots = np.array([[0.0, 0.0], [0.4, 0.0], [1.0, 0.0]])

    placed = place_reference_ball_xy(
        roots,
        start_frame=1,
        model_root_xy=np.array([-10.0, 2.0]),
        radius_offset_m=0.1,
        arc_angle_rad=np.pi / 2.0,
    )

    np.testing.assert_allclose(placed, [-10.0, 2.7], atol=1.0e-12)


def test_place_reference_ball_rotates_with_reset_yaw():
    placed = place_reference_ball_xy(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        start_frame=0,
        model_root_xy=np.zeros(2),
        yaw_rad=np.pi / 2.0,
    )

    np.testing.assert_allclose(placed, [0.0, 1.0], atol=1.0e-12)


def test_place_reference_ball_rejects_invalid_restart():
    with pytest.raises(ValueError, match="leave at least one"):
        place_reference_ball_xy(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            start_frame=1,
            model_root_xy=np.zeros(2),
        )


def test_select_reference_strike_uses_earliest_closest_frame():
    result = select_reference_strike(
        np.array(
            [
                [0.0, 0.0, 0.1],
                [0.5, 0.0, 0.1],
                [0.5, 0.0, 0.1],
                [0.8, 0.0, 0.1],
            ]
        ),
        np.array([0.5, 0.0, 0.1]),
        frequency_hz=50.0,
    )

    assert result.frame == 1
    assert result.center_distance_m == 0.0
    assert result.planar_speed_mps == pytest.approx(12.5)


def test_classify_ball_contacts_distinguishes_feet_and_other_robot_geoms():
    result = classify_ball_contacts(
        np.array([[3, 9], [4, 10], [12, 3]]),
        ball_geom=3,
        left_foot_geom=9,
        right_foot_geom=10,
        robot_geoms=frozenset({9, 10, 12}),
    )

    assert result.any_robot
    assert result.left_foot
    assert not result.right_foot


def test_classify_ball_contacts_handles_empty_pairs():
    result = classify_ball_contacts(
        np.empty((0, 2), dtype=np.int32),
        ball_geom=3,
        left_foot_geom=9,
        right_foot_geom=10,
        robot_geoms=frozenset({9, 10}),
    )

    assert not result.any_robot
    assert not result.left_foot
    assert not result.right_foot
