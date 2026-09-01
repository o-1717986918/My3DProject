from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.train_soccer_ball_motion import _load_bootstrap_gate


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
