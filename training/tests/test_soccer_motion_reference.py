import json

import numpy as np

from my3d_rl.soccer_motion_reference import validate_soccer_motion_reference


def _write_reference(path, *, support_contact=True, root_tilt=0.2):
    frames = 50
    root_position = np.zeros((frames, 3), dtype=np.float32)
    root_position[:, 0] = np.linspace(0.0, 0.4, frames)
    root_position[:, 2] = 0.65
    contact = np.zeros((frames, 2), dtype=bool)
    contact[5:45, 0] = support_contact
    contact[5:45, 1] = True
    metadata = {
        "source_url": "https://example.invalid/paid",
        "source_version": "test-revision",
        "source_license": "CC-BY-NC-4.0",
        "source_sha256": "0" * 64,
        "conversion_command": "synthetic-test",
        "retarget_method": "synthetic",
        "motion_type": "soccer_kick",
        "output_frequency_hz": 50.0,
        "kick_leg": "right",
        "competition_joint_limit_clipping": {
            "maximum_abs_correction_rad": 0.01
        },
        "rcss_replay": {
            "non_foot_pitch_contact_frames": 0,
            "minimum_contact_distance_m": -0.001,
            "ground_offset_max_step_m": 0.002,
        },
        "kick_geometry": {
            "peak_kick_foot_relative_speed_m_s": 2.0,
            "peak_other_foot_relative_speed_m_s": 1.0,
            "support_contact_near_peak": support_contact,
            "maximum_root_tilt_rad": root_tilt,
            "minimum_root_height_m": 0.65,
        },
    }
    np.savez_compressed(
        path,
        root_position=root_position,
        root_quaternion_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)
        ),
        root_linear_velocity=np.gradient(root_position, 0.02, axis=0).astype(
            np.float32
        ),
        root_angular_velocity=np.zeros((frames, 3), dtype=np.float32),
        joint_position=np.zeros((frames, 23), dtype=np.float32),
        joint_velocity=np.zeros((frames, 23), dtype=np.float32),
        foot_contact=contact,
        kick_leg=np.array("right"),
        metadata_json=np.array(json.dumps(metadata)),
    )


def test_soccer_reference_accepts_k0_candidate(tmp_path):
    path = tmp_path / "candidate.npz"
    _write_reference(path)

    result = validate_soccer_motion_reference(path)

    assert result["passed"]
    assert result["kick_leg"] == "right"
    assert result["frame_count"] == 50


def test_soccer_reference_rejects_missing_support_and_large_tilt(tmp_path):
    path = tmp_path / "failed.npz"
    _write_reference(path, support_contact=False, root_tilt=1.1)

    result = validate_soccer_motion_reference(path)

    assert not result["passed"]
    assert "support foot has no exact contact near peak kick speed" in result["errors"]
    assert "root tilt exceeds 1.0 rad" in result["errors"]
