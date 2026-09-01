import numpy as np
import pytest

from my3d_rl.soccer_motion_reset import (
    derive_case_seed,
    deterministic_reset_perturbation,
    yaw_quaternion_rotate,
    yaw_vector_rotate,
)


def test_reset_perturbation_is_reproducible_and_bounded():
    first = deterministic_reset_perturbation(
        base_seed=20260980,
        motion=2,
        start_frame=17,
        action_size=23,
        joint_noise=0.002,
        root_velocity_noise=0.005,
        yaw_range=0.01,
    )
    second = deterministic_reset_perturbation(
        base_seed=20260980,
        motion=2,
        start_frame=17,
        action_size=23,
        joint_noise=0.002,
        root_velocity_noise=0.005,
        yaw_range=0.01,
    )

    assert first.case_seed == second.case_seed
    assert first.yaw == second.yaw
    np.testing.assert_array_equal(
        first.joint_position_noise, second.joint_position_noise
    )
    np.testing.assert_array_equal(
        first.root_velocity_noise, second.root_velocity_noise
    )
    assert np.max(np.abs(first.joint_position_noise)) <= 0.002
    assert np.max(np.abs(first.root_velocity_noise)) <= 0.005
    assert abs(first.yaw) <= 0.01


def test_reset_case_seed_changes_with_each_coordinate():
    base = derive_case_seed(7, 1, 2, 3)
    assert 0 <= base <= np.iinfo(np.int64).max
    assert base != derive_case_seed(8, 1, 2, 3)
    assert base != derive_case_seed(7, 2, 2, 3)
    assert base != derive_case_seed(7, 1, 3, 3)
    assert base != derive_case_seed(7, 1, 2, 4)


def test_reset_seed_rejects_negative_coordinate():
    with pytest.raises(ValueError, match="non-negative"):
        derive_case_seed(7, -1)


def test_shared_yaw_rotations_preserve_expected_geometry():
    np.testing.assert_allclose(
        yaw_quaternion_rotate(np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        yaw_vector_rotate(np.array([1.0, 0.0, 2.0]), np.pi / 2),
        np.array([0.0, 1.0, 2.0]),
        atol=1.0e-12,
    )
