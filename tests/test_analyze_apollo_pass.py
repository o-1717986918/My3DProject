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
    assert not outcomes[1].contact
    assert outcomes[1].forward_progress_m == 0.0
