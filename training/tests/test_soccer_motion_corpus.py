from __future__ import annotations

import hashlib
import json

import numpy as np

from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus


def _write_motion(path, frames: int, leg: str) -> None:
    np.savez_compressed(
        path,
        root_position=np.zeros((frames, 3), dtype=np.float32),
        root_quaternion_xyzw=np.tile(
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)
        ),
        root_linear_velocity=np.zeros((frames, 3), dtype=np.float32),
        root_angular_velocity=np.zeros((frames, 3), dtype=np.float32),
        joint_position=np.zeros((frames, 23), dtype=np.float32),
        joint_velocity=np.zeros((frames, 23), dtype=np.float32),
        foot_contact=np.ones((frames, 2), dtype=bool),
        kick_leg=np.array(leg),
    )


def test_corpus_pads_with_last_frame_and_excludes_terminal_reset(tmp_path):
    _write_motion(tmp_path / "a.npz", 3, "left")
    _write_motion(tmp_path / "b.npz", 5, "right")

    corpus = load_soccer_motion_corpus(tmp_path, validate=False)

    assert corpus.motion_count == 2
    assert corpus.maximum_frames == 5
    np.testing.assert_array_equal(corpus.lengths, [3, 5])
    np.testing.assert_array_equal(corpus.kick_leg_one_hot, [[1, 0], [0, 1]])
    np.testing.assert_allclose(corpus.reset_weights.sum(axis=1), 1.0)
    assert corpus.reset_weights[0, 2] == 0.0
    assert np.all(corpus.reset_weights[0, 3:] == 0.0)


def test_corpus_accepts_hash_bound_failure_weights(tmp_path):
    path = tmp_path / "motion.npz"
    _write_motion(path, 4, "right")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "records": [
                    {
                        "relative_path": "motion.npz",
                        "sha256": digest,
                        "failure_frame_sampling": {
                            "weights": [0.1, 0.2, 0.6, 0.1]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus = load_soccer_motion_corpus(
        tmp_path, failure_report=report, validate=False
    )
    np.testing.assert_allclose(corpus.reset_weights[0], [1 / 9, 2 / 9, 6 / 9, 0])
