from pathlib import Path

from scripts.analyze_contact_probe import analyze, find_release_time, load_statuses


def test_contact_probe_accepts_baseline_and_legacy_status_fields(
    tmp_path: Path,
) -> None:
    trigger = tmp_path / "player-7.log"
    trigger.write_text(
        "\n".join(
            [
                "APOLLO_REBUILD_STATUS t=1.00 ball_dist=0.33 ball_visible=1 "
                "ball_x=0 ball_y=0 x=-0.33 y=0 z=0.65 ball_speed=0 motion=Walk",
                "APOLLO_REBUILD_STATUS t=1.20 ball_dist=0.50 ball_visible=1 "
                "ball_x=0.50 ball_y=0.10 x=0 y=0 z=0.64 ball_speed=1.5 motion=Walk",
            ]
        ),
        encoding="utf-8",
    )
    observer = tmp_path / "player-6.log"
    observer.write_text(
        "MY3D_STATUS server_time=1.10 ball_position_valid=1 ball_visible=1 "
        "ball_x=0.20 ball_y=0.02 ball_velocity_valid=1 ball_vx=1 ball_vy=0",
        encoding="utf-8",
    )

    result = analyze(
        [trigger, observer], trigger, (0.0, 0.0), 0.0, 1.0, 0.15, 0.55, 0.10
    )

    assert result["contact"]
    assert abs(result["time_to_contact_s"] - 0.10) < 1.0e-9
    assert abs(result["maximum_forward_progress_m"] - 0.50) < 1.0e-9
    assert abs(result["lateral_error_at_max_progress_m"] - 0.10) < 1.0e-9
    assert result["peak_observed_ball_speed_mps"] == 1.5
    assert not result["fell"]


def test_release_time_uses_positions_when_legacy_log_has_no_ball_dist(
    tmp_path: Path,
) -> None:
    log = tmp_path / "legacy.log"
    log.write_text(
        "MY3D_STATUS server_time=2 ball_position_valid=1 ball_visible=1 "
        "ball_x=3 ball_y=0 x=2.7 y=0 z=0.65\n",
        encoding="utf-8",
    )

    release_time = find_release_time(
        load_statuses(log), (3.0, 0.0), 0.15, 0.55
    )

    assert release_time == 2.0
