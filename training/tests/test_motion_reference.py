import json

import numpy as np

from my3d_rl.motion_reference import validate_motion_reference


def test_motion_reference_requires_and_accepts_running_morphology(tmp_path):
    frames = 25
    metadata = {
        "source_url": "https://example.invalid/source",
        "source_version": "test-only",
        "source_license": "test-only",
        "conversion_command": "synthetic-test",
        "source_sha256": "0" * 64,
    }
    contact = np.ones((frames, 2), dtype=bool)
    contact[12] = False
    path = tmp_path / "run-reference.npz"
    np.savez(
        path,
        root_position=np.zeros((frames, 3), dtype=np.float32),
        root_quaternion_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)
        ),
        root_linear_velocity=np.zeros((frames, 3), dtype=np.float32),
        root_angular_velocity=np.zeros((frames, 3), dtype=np.float32),
        joint_position=np.zeros((frames, 23), dtype=np.float32),
        joint_velocity=np.zeros((frames, 23), dtype=np.float32),
        foot_contact=contact,
        metadata_json=np.array(json.dumps(metadata)),
    )

    result = validate_motion_reference(path)

    assert result["passed"]
    assert result["frame_count"] == frames
    assert result["longest_airborne_frames"] == 1
