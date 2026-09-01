import hashlib
import json
from pathlib import Path

import pytest

from tools.train_soccer_motion import (
    _effective_timesteps,
    _load_curriculum_gate,
    _tree_sha256,
)


def _write_gate(tmp_path: Path, checkpoint: Path, *, passed: bool = True) -> Path:
    checkpoint.mkdir()
    (checkpoint / "params").write_bytes(b"checkpoint")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "purpose": "k1_exact_cpu_fixed_motion_phase_grid",
                "checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "purpose": "k1_paired_exact_cpu_policy_comparison",
                "candidate": str(candidate),
                "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "curriculum_advance_passed": passed,
                "paired_trials": 777,
                "completion_rate_delta": 0.01,
                "mean_survival_fraction_delta": 0.02,
            }
        ),
        encoding="utf-8",
    )
    return comparison


def test_curriculum_gate_binds_evaluated_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    comparison = _write_gate(tmp_path, checkpoint)

    result = _load_curriculum_gate(comparison, checkpoint)

    assert result["curriculum_advance_passed"] is True
    assert result["paired_trials"] == 777
    assert result["checkpoint_tree_sha256"] == _tree_sha256(checkpoint)


def test_curriculum_gate_rejects_failed_gate(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    comparison = _write_gate(tmp_path, checkpoint, passed=False)

    with pytest.raises(ValueError, match="did not authorize"):
        _load_curriculum_gate(comparison, checkpoint)


def test_curriculum_gate_rejects_another_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    comparison = _write_gate(tmp_path, checkpoint)
    another = tmp_path / "another"
    another.mkdir()
    (another / "params").write_bytes(b"other")

    with pytest.raises(ValueError, match="differs"):
        _load_curriculum_gate(comparison, another)


def test_curriculum_gate_rejects_modified_candidate_report(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    comparison = _write_gate(tmp_path, checkpoint)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        _load_curriculum_gate(comparison, checkpoint)


@pytest.mark.parametrize(
    ("requested", "num_evals", "expected", "intervals", "steps_per_interval"),
    (
        (262_144, 3, 393_216, 2, 1),
        (2_359_296, 5, 2_359_296, 4, 3),
        (1, 1, 196_608, 1, 1),
    ),
)
def test_effective_timesteps_mirrors_brax_rounding(
    requested, num_evals, expected, intervals, steps_per_interval
):
    result = _effective_timesteps(
        requested,
        optimizer_step_timesteps=196_608,
        num_evals=num_evals,
    )

    assert result == (expected, intervals, steps_per_interval)
