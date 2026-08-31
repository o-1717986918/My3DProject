import numpy as np

from my3d_rl.periodic_reference import (
    circular_gradient,
    mirror_root_quaternion_xyzw,
    project_half_cycle,
)


def test_half_cycle_projection_is_exact_and_idempotent():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(12, 4))
    source = np.array([1, 0, 3, 2])
    factor = np.array([1.0, 1.0, -1.0, -1.0])

    projected = project_half_cycle(values, source, factor)
    projected_twice = project_half_cycle(projected, source, factor)

    np.testing.assert_allclose(projected[6:], projected[:6, source] * factor)
    np.testing.assert_allclose(projected_twice, projected)


def test_weighted_half_cycle_projection_remains_exact():
    values = np.arange(48, dtype=np.float64).reshape(12, 4)
    source = np.array([1, 0, 3, 2])
    factor = np.array([1.0, 1.0, -1.0, -1.0])

    projected = project_half_cycle(values, source, factor, source_half_weight=0.8)

    np.testing.assert_allclose(projected[6:], projected[:6, source] * factor)


def test_sagittal_root_reflection_is_an_involution():
    rotations = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.12, -0.18, 0.07, 0.973],
        ]
    )
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)

    twice = mirror_root_quaternion_xyzw(mirror_root_quaternion_xyzw(rotations))

    np.testing.assert_allclose(np.abs(np.sum(twice * rotations, axis=1)), 1.0)


def test_circular_gradient_preserves_constant_cycle_progress():
    frame_count = 20
    cycle_delta = np.array([2.0, 0.0, 0.0])
    position = np.zeros((frame_count, 3))
    position[:, 0] = np.arange(frame_count) * cycle_delta[0] / frame_count

    velocity = circular_gradient(position, 0.02, cycle_delta)

    np.testing.assert_allclose(velocity[:, 0], 5.0)
    np.testing.assert_allclose(velocity[:, 1:], 0.0)
