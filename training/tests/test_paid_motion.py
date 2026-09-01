import numpy as np
import pytest

from my3d_rl.paid_motion import (
    GMR_HUMAN_TO_PAID_BODY,
    PAID_G1_BODY_ORDER,
    PAID_G1_JOINT_ORDER,
    load_paid_motion,
    paid_frame_for_gmr,
    semantic_projection_qpos,
    source_foot_contact,
)


T1_JOINT_ORDER = (
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)


def _write_paid(path, *, frames=40, kick_leg="right", fps=50):
    joint_position = np.zeros((frames, 29), dtype=np.float32)
    joint_position[:, 0] = np.linspace(0.0, 0.4, frames)
    joint_position[:, 11] = np.linspace(0.1, 0.5, frames)
    joint_velocity = np.gradient(joint_position, 1.0 / fps, axis=0)
    body_position = np.zeros((frames, 30, 3), dtype=np.float32)
    body_position[:, :, 2] = 0.6
    body_position[:, 0, 2] = 0.75
    left_foot = PAID_G1_BODY_ORDER.index("left_ankle_roll_link")
    right_foot = PAID_G1_BODY_ORDER.index("right_ankle_roll_link")
    body_position[:, [left_foot, right_foot], 2] = 0.05
    body_position[10:20, right_foot, 2] = 0.20
    body_quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
    body_quaternion[:, :, 0] = 1.0
    body_velocity = np.gradient(body_position, 1.0 / fps, axis=0)
    np.savez_compressed(
        path,
        fps=np.array([fps], dtype=np.int64),
        joint_pos=joint_position,
        joint_vel=joint_velocity,
        body_pos_w=body_position,
        body_quat_w=body_quaternion,
        body_lin_vel_w=body_velocity,
        body_ang_vel_w=np.zeros_like(body_velocity),
        kick_leg=np.array(kick_leg),
    )


def test_paid_loader_contact_and_semantic_projection(tmp_path):
    path = tmp_path / "synthetic_right.npz"
    _write_paid(path)

    clip = load_paid_motion(path)
    contact, diagnostics = source_foot_contact(
        clip, maximum_vertical_speed_m_s=20.0
    )
    qpos = semantic_projection_qpos(clip, T1_JOINT_ORDER)
    frame = paid_frame_for_gmr(clip, 0)

    assert clip.frame_count == 40
    assert clip.kick_leg == "right"
    assert contact[:, 0].all()
    assert not contact[10:20, 1].any()
    assert diagnostics["contact_frames"] == [40, 30]
    assert qpos.shape == (40, 30)
    np.testing.assert_allclose(qpos[:, :3], clip.body_position_world[:, 0])
    np.testing.assert_allclose(qpos[:, 7:9], 0.0)
    left_shoulder = PAID_G1_JOINT_ORDER.index("left_shoulder_pitch_joint")
    np.testing.assert_allclose(qpos[:, 9], clip.joint_position[:, left_shoulder])
    assert set(frame) == set(GMR_HUMAN_TO_PAID_BODY)
    np.testing.assert_allclose(frame["pelvis"][1], [1.0, 0.0, 0.0, 0.0])


def test_paid_loader_rejects_unpinned_frequency_and_label(tmp_path):
    wrong_fps = tmp_path / "wrong_right.npz"
    _write_paid(wrong_fps, fps=60)
    with pytest.raises(ValueError, match="60.0 Hz"):
        load_paid_motion(wrong_fps)

    wrong_label = tmp_path / "wrong_right.npz"
    _write_paid(wrong_label, kick_leg="left")
    with pytest.raises(ValueError, match="filename suffix"):
        load_paid_motion(wrong_label)
