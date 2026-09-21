from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.train_soccer_ball_motion import (
    _load_bootstrap_gate,
    _load_parent_run_gate,
    _target_angle_range_degrees,
)


def _write_bootstrap_report(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "params").write_bytes(b"k2-checkpoint")
    digest = hashlib.sha256()
    digest.update(b"params\0")
    digest.update(
        hashlib.sha256(b"k2-checkpoint").hexdigest().encode("ascii")
    )
    digest.update(b"\n")
    report = tmp_path / "transfer-report.json"
    report.write_text(
        json.dumps(
            {
                "purpose": "k2_zero_row_ball_target_checkpoint_transfer",
                "status": "complete",
                "target_checkpoint": str(checkpoint),
                "target_checkpoint_tree_sha256": digest.hexdigest(),
                "source_checkpoint_tree_sha256": "1" * 64,
                "parity": {
                    "passed": True,
                    "policy_max_abs": 0.0,
                    "value_max_abs": 0.0,
                    "required_max_abs": 5.0e-7,
                },
            }
        ),
        encoding="utf-8",
    )
    return report, checkpoint


def test_k2_trainer_binds_restore_checkpoint_to_transfer_report(tmp_path):
    report, checkpoint = _write_bootstrap_report(tmp_path)

    result = _load_bootstrap_gate(report, checkpoint)

    assert result["checkpoint_tree_sha256"]
    assert result["parity"]["passed"] is True
    assert result["report_sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()


def test_k2_trainer_rejects_failed_transfer_parity(tmp_path):
    report, checkpoint = _write_bootstrap_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["parity"]["passed"] = False
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="parity gate"):
        _load_bootstrap_gate(report, checkpoint)


def test_k2_trainer_rejects_another_checkpoint(tmp_path):
    report, unused = _write_bootstrap_report(tmp_path)
    another = tmp_path / "another"
    another.mkdir()
    (another / "params").write_bytes(b"other")

    with pytest.raises(ValueError, match="differs"):
        _load_bootstrap_gate(report, another)


def _write_parent_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "parent"
    checkpoint = run_dir / "checkpoints" / "000000196608"
    checkpoint.mkdir(parents=True)
    (checkpoint / "params").write_bytes(b"continued-k2")
    manifest = run_dir / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "k2_fixed_motion_ball_target_residual_training",
                "policy_contract": "soccer_ball_motion_policy_v1",
                "timestep_accounting_passed": True,
                "observed_final_timesteps": 393216,
                "git_revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    return manifest, checkpoint


def test_k2_trainer_accepts_checkpoint_from_completed_parent_run(tmp_path):
    manifest, checkpoint = _write_parent_run(tmp_path)

    result = _load_parent_run_gate(manifest, checkpoint)

    assert result["type"] == "k2_parent_checkpoint"
    assert result["checkpoint_step"] == 196608
    assert result["checkpoint_tree_sha256"]


def test_k2_trainer_rejects_checkpoint_outside_parent_run(tmp_path):
    manifest, unused = _write_parent_run(tmp_path)
    other = tmp_path / "other" / "000000196608"
    other.mkdir(parents=True)
    (other / "params").write_bytes(b"other")

    with pytest.raises(ValueError, match="outside"):
        _load_parent_run_gate(manifest, other)


def test_k2_trainer_rejects_incomplete_parent_run(tmp_path):
    manifest, checkpoint = _write_parent_run(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "running"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="did not complete"):
        _load_parent_run_gate(manifest, checkpoint)


def test_k2_trainer_resolves_fixed_and_overlapping_angle_curricula():
    assert _target_angle_range_degrees(15.0, None, None) == (15.0, 15.0)
    assert _target_angle_range_degrees(0.0, 15.0, 22.5) == (15.0, 22.5)


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (15.0, None, "both"),
        (22.5, 15.0, "exceeds"),
        (float("nan"), 15.0, "finite"),
    ],
)
def test_k2_trainer_rejects_invalid_angle_curriculum(
    minimum, maximum, message
):
    with pytest.raises(ValueError, match=message):
        _target_angle_range_degrees(0.0, minimum, maximum)
