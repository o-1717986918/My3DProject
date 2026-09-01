import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training.tools.train_kick import _load_parity_report, _load_transition_corpus


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.npz"
    np.savez_compressed(
        path,
        qpos=np.arange(6 * 37, dtype=np.float32).reshape(6, 37),
        qvel=np.arange(6 * 35, dtype=np.float32).reshape(6, 35),
        split=np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8),
        rollout_id=np.array([10, 11, 12, 20, 21, 22], dtype=np.int32),
        phase_bucket=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "purpose": "kick_policy_v3_walk_to_kick_transition_corpus",
                "npz_sha256": _sha256(path),
                "teacher_condition_index": 60,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_transition_corpus_loader_keeps_rollouts_out_of_validation(tmp_path: Path):
    path = _write_corpus(tmp_path)

    train_qpos, train_qvel, validation_qpos, validation_qvel, metadata = (
        _load_transition_corpus(path)
    )

    assert train_qpos.shape == (3, 37)
    assert train_qvel.shape == (3, 35)
    assert validation_qpos.shape == (3, 37)
    assert validation_qvel.shape == (3, 35)
    assert metadata["train_phase_buckets"] == [0, 1, 2]
    assert metadata["validation_phase_buckets"] == [0, 1, 2]
    assert metadata["teacher_condition_index"] == 60


def test_transition_corpus_loader_rejects_hash_mismatch(tmp_path: Path):
    path = _write_corpus(tmp_path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        _load_transition_corpus(path)


def _write_parity_report(tmp_path: Path, *, backend: str, passed: bool) -> Path:
    path = tmp_path / "parity.json"
    path.write_text(
        json.dumps(
            {
                "purpose": "kick_identical_control_cpu_mjx_parity",
                "accelerated_implementation": backend,
                "summary": {"parity_gate_passed": passed},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parity_report_accepts_matching_verified_backend(tmp_path: Path):
    path = _write_parity_report(tmp_path, backend="warp", passed=True)

    metadata = _load_parity_report(path, "warp")

    assert metadata["path"] == str(path.resolve())
    assert metadata["sha256"] == _sha256(path)
    assert metadata["summary"]["parity_gate_passed"] is True


@pytest.mark.parametrize(
    ("backend", "passed", "expected"),
    [("jax", True, "backend"), ("warp", False, "did not pass")],
)
def test_parity_report_rejects_unverified_run(
    tmp_path: Path, backend: str, passed: bool, expected: str
):
    path = _write_parity_report(tmp_path, backend=backend, passed=passed)

    with pytest.raises(ValueError, match=expected):
        _load_parity_report(path, "warp")
