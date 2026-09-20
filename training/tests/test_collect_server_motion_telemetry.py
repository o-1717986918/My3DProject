from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.collect_server_motion_telemetry import collect, parse_telemetry_line


def sample_line(*, time_s: float = 1.0, valid: int = 1) -> str:
    joints = ",".join("0" for _ in range(23))
    return (
        "APOLLO_REBUILD_MOTION_TELEMETRY "
        f"t={time_s} player=7 side=left motion=Walk "
        f"ball_valid={valid} ball_age={'0.02' if valid else 'inf'} "
        "ball_velocity_valid=0 "
        "self_xyz=0,0,0.8 self_quat_wxyz=1,0,0,0 "
        "self_velocity_body=0.4,0,0 gyro_deg_s=0,0,0 "
        "ball_xyz=0.5,0,0.11 ball_velocity=0,0,0 "
        f"joint_position_deg={joints} joint_velocity_deg_s={joints} "
        f"target_position_deg={joints} target_mask={'1' * 23}"
    )


def test_parse_complete_server_frame_and_reject_partial_joint_state() -> None:
    assert parse_telemetry_line("other log line") is None
    frame = parse_telemetry_line(sample_line())
    assert frame is not None
    assert frame["motion"] == "Walk"
    assert frame["joint_position_deg"].shape == (23,)
    assert np.all(frame["target_mask"] == 1)
    invalid_ball = parse_telemetry_line(sample_line(valid=0))
    assert invalid_ball is not None
    assert np.isinf(invalid_ball["ball_position_age_s"])
    with pytest.raises(ValueError, match="joint_position_deg"):
        parse_telemetry_line(sample_line().replace(
            f"joint_position_deg={','.join('0' for _ in range(23))}",
            "joint_position_deg=0,0",
        ))


def test_collect_holds_out_whole_matches_not_correlated_players(tmp_path: Path) -> None:
    match_dirs = [tmp_path / "first", tmp_path / "second"]
    for match_dir in match_dirs:
        match_dir.mkdir()
        for player in (1, 7):
            (match_dir / f"Apollo-Rebuild-{player}.log").write_text(
                sample_line(time_s=1.0) + "\n" + sample_line(time_s=1.02) + "\n",
                encoding="utf-8",
            )

    arrays, summary = collect(match_dirs)

    assert summary["samples"] == 8
    assert summary["fresh_near_ball_samples"] == 8
    assert summary["complete_target_samples"] == 8
    assert summary["train_samples"] == summary["validation_samples"] == 4
    assert np.all(arrays["split"][arrays["match_id"] == 0] == 0)
    assert np.all(arrays["split"][arrays["match_id"] == 1] == 1)


def test_collect_rejects_nonincreasing_player_time(tmp_path: Path) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / "Apollo-Rebuild-7.log").write_text(
        sample_line(time_s=1.0) + "\n" + sample_line(time_s=1.0) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-increasing"):
        collect([match_dir])
