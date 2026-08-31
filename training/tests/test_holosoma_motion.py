import numpy as np

from my3d_rl.holosoma_motion import (
    canonicalize_forward,
    quaternion_slerp_wxyz,
    resample_qpos,
)


def test_quaternion_slerp_uses_short_arc_and_normalizes():
    start = np.array([1.0, 0.0, 0.0, 0.0])
    end = -np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    middle = quaternion_slerp_wxyz(start, end, 0.5)

    np.testing.assert_allclose(np.linalg.norm(middle), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.abs(middle),
        [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)],
        atol=1.0e-12,
    )


def test_resample_and_canonicalize_forward():
    qpos = np.zeros((4, 30), dtype=np.float64)
    qpos[:, 1] = np.arange(4)
    qpos[:, 2] = 0.65
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.arange(4) * 0.1

    resampled = resample_qpos(qpos, input_fps=30.0, output_fps=50.0)
    canonical, heading = canonicalize_forward(resampled)

    assert resampled.shape == (6, 30)
    np.testing.assert_allclose(canonical[0, :2], [0.0, 0.0], atol=1.0e-12)
    assert canonical[-1, 0] > 2.99
    assert abs(canonical[-1, 1]) < 1.0e-12
    np.testing.assert_allclose(heading, np.pi / 2, atol=1.0e-12)
    np.testing.assert_allclose(
        np.linalg.norm(canonical[:, 3:7], axis=1), 1.0, atol=1.0e-12
    )
