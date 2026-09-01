from pathlib import Path

from scripts.analyze_apollo_pass import analyze_logs, parse_status_line


def test_parse_status_line_ignores_non_status_text() -> None:
    assert parse_status_line("server started") is None
    assert parse_status_line("MY3D_STATUS cycle=10 kick_mode=None") == {
        "cycle": "10",
        "kick_mode": "None",
    }


def test_targeted_pass_requires_forward_ball_progress(tmp_path: Path) -> None:
    contact_log = tmp_path / "contact.log"
    contact_log.write_text(
        "\n".join(
            [
                "MY3D_STATUS cycle=100 kick_mode=TargetedPass action_id=7 "
                "pass_seq=2 ball_x=0 ball_y=0 pass_target_x=4 pass_target_y=0",
                "MY3D_STATUS cycle=150 kick_mode=None action_id=0 pass_seq=0 "
                "ball_x=0.4 ball_y=0 pass_target_x=0 pass_target_y=0",
            ]
        ),
        encoding="utf-8",
    )
    miss_log = tmp_path / "miss.log"
    miss_log.write_text(
        "\n".join(
            [
                "MY3D_STATUS cycle=100 kick_mode=TargetedPass action_id=8 "
                "pass_seq=3 ball_x=0 ball_y=0 pass_target_x=4 pass_target_y=0",
                "MY3D_STATUS cycle=150 kick_mode=None action_id=0 pass_seq=0 "
                "ball_x=-0.2 ball_y=0 pass_target_x=0 pass_target_y=0",
            ]
        ),
        encoding="utf-8",
    )

    outcomes = analyze_logs([contact_log, miss_log])
    assert len(outcomes) == 2
    assert outcomes[0].contact
    assert outcomes[0].forward_progress_m == 0.4
    assert outcomes[0].lateral_error_m == 0.0
    assert outcomes[0].direction_error_deg == 0.0
    assert not outcomes[1].contact
    assert outcomes[1].forward_progress_m == 0.0


def test_pass_outcome_stops_before_a_new_targeted_action(tmp_path: Path) -> None:
    log = tmp_path / "two-actions.log"
    log.write_text(
        "\n".join(
            [
                "MY3D_STATUS cycle=100 kick_mode=TargetedPass action_id=7 "
                "pass_seq=2 ball_x=0 ball_y=0 pass_target_x=2 pass_target_y=0",
                "MY3D_STATUS cycle=120 kick_mode=TargetedPass action_id=7 "
                "pass_seq=2 ball_x=0.4 ball_y=0.3 pass_target_x=2 pass_target_y=0",
                "MY3D_STATUS cycle=130 kick_mode=TargetedPass action_id=8 "
                "pass_seq=3 ball_x=1.5 ball_y=1.5 pass_target_x=3 pass_target_y=0",
            ]
        ),
        encoding="utf-8",
    )

    outcomes = analyze_logs([log])
    first = outcomes[0]
    assert first.action_id == 7
    assert first.forward_progress_m == 0.4
    assert first.signed_lateral_error_m == 0.3
    assert first.lateral_error_m == 0.3
    assert 36.8 < first.signed_direction_error_deg < 36.9
    assert 36.8 < first.direction_error_deg < 36.9


def test_active_kick_identity_wins_over_new_strategy_plan(tmp_path: Path) -> None:
    log = tmp_path / "active-kick.log"
    log.write_text(
        "\n".join(
            [
                "MY3D_STATUS cycle=100 kick_mode=TargetedPass "
                "kick_action_id=7 kick_sequence_id=2 action_id=7 pass_seq=2 "
                "ball_x=0 ball_y=0 kick_target_x=2 kick_target_y=0",
                "MY3D_STATUS cycle=120 kick_mode=TargetedPass "
                "kick_action_id=7 kick_sequence_id=2 action_id=8 pass_seq=3 "
                "ball_x=0.4 ball_y=0.1 kick_target_x=2 kick_target_y=0",
                "MY3D_STATUS cycle=130 kick_mode=None kick_action_id=0 "
                "kick_sequence_id=0 action_id=8 pass_seq=3 ball_x=0.5 ball_y=0.1",
            ]
        ),
        encoding="utf-8",
    )

    outcomes = analyze_logs([log])
    assert len(outcomes) == 1
    assert outcomes[0].action_id == 7
    assert outcomes[0].sequence_id == 2
    assert outcomes[0].forward_progress_m == 0.5
