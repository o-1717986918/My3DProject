import json

import numpy as np

from my3d_rl.motion_reference import validate_motion_reference


def _write_reference(path, *, persistent_flight=False, failed_projection=False):
    frames = 50
    time = np.arange(frames, dtype=np.float32) / 50.0
    metadata = {
        "source_url": "https://example.invalid/source",
        "source_version": "test-only",
        "source_license": "test-only",
        "conversion_command": "synthetic-test",
        "source_sha256": "0" * 64,
        "output_frequency_hz": 50.0,
        "rcss_replay": {
            "non_foot_pitch_contact_frames": 0,
            "minimum_contact_distance_m": -0.001,
        },
    }
    if failed_projection:
        metadata["periodic_projection"] = {"passed": False}
    contact = np.zeros((frames, 2), dtype=bool)
    if not persistent_flight:
        contact[0:10, 0] = True
        contact[14:24, 1] = True
        contact[28:38, 0] = True
        contact[42:50, 1] = True
    joint_position = np.zeros((frames, 23), dtype=np.float32)
    joint_position[:, 14] = 0.4 + 0.35 * np.sin(2.0 * np.pi * 1.5 * time)
    joint_position[:, 20] = 0.4 - 0.35 * np.sin(2.0 * np.pi * 1.5 * time)
    joint_velocity = np.gradient(joint_position, 0.02, axis=0).astype(np.float32)
    root_position = np.zeros((frames, 3), dtype=np.float32)
    root_position[:, 0] = 1.5 * time
    root_position[:, 2] = 0.65 + 0.02 * np.sin(2.0 * np.pi * 3.0 * time)
    root_velocity = np.gradient(root_position, 0.02, axis=0).astype(np.float32)
    np.savez(
        path,
        root_position=root_position,
        root_quaternion_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)
        ),
        root_linear_velocity=root_velocity,
        root_angular_velocity=np.zeros((frames, 3), dtype=np.float32),
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        foot_contact=contact,
        metadata_json=np.array(json.dumps(metadata)),
    )


def test_motion_reference_requires_and_accepts_running_morphology(tmp_path):
    path = tmp_path / "run-reference.npz"
    _write_reference(path)

    result = validate_motion_reference(path)

    assert result["passed"]
    assert result["frame_count"] == 50
    assert result["longest_airborne_frames"] == 4
    assert result["average_horizontal_speed_m_s"] > 1.4


def test_motion_reference_rejects_persistent_flight(tmp_path):
    path = tmp_path / "floating-reference.npz"
    _write_reference(path, persistent_flight=True)

    result = validate_motion_reference(path)

    assert not result["passed"]
    assert any("aerial interval" in error for error in result["errors"])
    assert any("left contact count" in error for error in result["errors"])


def test_motion_reference_rejects_failed_periodic_projection(tmp_path):
    path = tmp_path / "failed-periodic-reference.npz"
    _write_reference(path, failed_projection=True)

    result = validate_motion_reference(path)

    assert not result["passed"]
    assert "periodic projection metadata does not pass its gates" in result["errors"]
