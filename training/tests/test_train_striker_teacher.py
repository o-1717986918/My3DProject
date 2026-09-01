import json
from pathlib import Path

import pytest

from my3d_rl.contract import load_policy_contract
from my3d_rl.striker_env import DEFAULT_CONTRACT
from training.tools.train_striker_teacher import (
    NORMALIZE_OBSERVATIONS,
    STAGES,
    _load_kick_prior,
    _load_parity_report,
)


def test_striker_curriculum_expands_without_changing_stage_order():
    near = STAGES["near_ball"]
    closed = STAGES["closed_loop"]
    robust = STAGES["robust"]

    assert near["robot_distance_range"][1] < closed["robot_distance_range"][1]
    assert closed["robot_distance_range"][1] < robust["robot_distance_range"][1]
    assert near["robot_lateral_range"][1] < closed["robot_lateral_range"][1]
    assert closed["robot_lateral_range"][1] < robust["robot_lateral_range"][1]
    assert near["target_distance_range"][0] < near["target_distance_range"][1]
    assert NORMALIZE_OBSERVATIONS is False


def test_striker_parity_gate_requires_matching_verified_backend(tmp_path: Path):
    path = tmp_path / "parity.json"
    path.write_text(
        json.dumps(
            {
                "purpose": "striker_identical_control_cpu_mjx_parity",
                "accelerated_implementation": "warp",
                "summary": {"parity_gate_passed": True},
            }
        ),
        encoding="utf-8",
    )

    accepted = _load_parity_report(path, "warp")
    assert accepted["summary"]["parity_gate_passed"] is True
    with pytest.raises(ValueError, match="backend"):
        _load_parity_report(path, "jax")


def test_striker_parity_gate_rejects_wrong_purpose(tmp_path: Path):
    path = tmp_path / "parity.json"
    path.write_text(
        json.dumps(
            {
                "purpose": "kick_identical_control_cpu_mjx_parity",
                "accelerated_implementation": "warp",
                "summary": {"parity_gate_passed": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="purpose"):
        _load_parity_report(path, "warp")


def test_kick_prior_loader_versions_selected_trajectory(tmp_path: Path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "purpose": "test_kick_prior",
                "spec": {"duration_s": 1.2},
                "parameters": [0.0] * 14,
            }
        ),
        encoding="utf-8",
    )

    trajectory, metadata = _load_kick_prior(
        path,
        load_policy_contract(DEFAULT_CONTRACT),
        condition_index=None,
    )

    assert trajectory.shape == (60, 23)
    assert metadata["steps"] == 60
    assert metadata["condition_index"] is None
    assert len(metadata["manifest_sha256"]) == 64
