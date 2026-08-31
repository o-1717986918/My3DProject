import numpy as np
import pytest

from my3d_rl.gmr_motion import (
    clip_contract_joint_limits,
    contact_only_human_joints,
    map_named_qpos_to_contract,
)


def test_map_named_qpos_drops_wrists_and_zero_fills_head() -> None:
    source = np.zeros((2, 12), dtype=np.float64)
    source[:, 3] = 1.0
    source[:, 7] = [0.1, 0.2]
    source[:, 8] = [0.3, 0.4]
    source[:, 9:] = 99.0
    result = map_named_qpos_to_contract(
        source,
        {
            "Left_Shoulder_Pitch": 7,
            "Waist": 8,
            "Left_Wrist_Pitch": 9,
        },
        ["AAHead_yaw", "Head_pitch", "Left_Shoulder_Pitch", "Waist"],
    )
    np.testing.assert_allclose(result[:, 7:9], 0.0)
    np.testing.assert_allclose(result[:, 9], [0.1, 0.2])
    np.testing.assert_allclose(result[:, 10], [0.3, 0.4])


def test_map_named_qpos_rejects_missing_required_joint() -> None:
    source = np.zeros((2, 8), dtype=np.float64)
    source[:, 3] = 1.0
    with pytest.raises(ValueError, match="missing required joints"):
        map_named_qpos_to_contract(source, {}, ["Waist"])


def test_contact_only_human_joints_normalizes_shared_ground() -> None:
    left = np.array([[1.0, 2.0, 0.08], [1.1, 2.1, 0.18]])
    right = np.array([[3.0, 4.0, 0.12], [3.1, 4.1, 0.07]])
    human, ground = contact_only_human_joints(left, right)
    assert human.shape == (2, 22, 3)
    assert ground == pytest.approx(0.07)
    np.testing.assert_allclose(human[:, 8, 2], [0.01, 0.11])
    np.testing.assert_allclose(human[:, 4, 2], [0.05, 0.0])


def test_clip_contract_joint_limits_reports_corrections() -> None:
    qpos = np.zeros((2, 9), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = [[-2.0, 0.2], [0.5, 3.0]]
    clipped, report = clip_contract_joint_limits(
        qpos, np.array([-1.0, -0.5]), np.array([1.0, 2.0])
    )
    np.testing.assert_allclose(clipped[:, 7:], [[-1.0, 0.2], [0.5, 2.0]])
    assert report["clipped_value_count"] == 2
    assert report["maximum_abs_correction_rad"] == pytest.approx(1.0)
