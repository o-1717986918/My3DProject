from pathlib import Path

from scripts.analyze_apollo_pass import analyze_logs


def _status(
    cycle: int,
    mode: str,
    ball_x: float,
    *,
    target_x: float = 0.55,
) -> str:
    return (
        "MY3D_STATUS "
        f"cycle={cycle} kick_mode={mode} ball_x={ball_x} ball_y=0 "
        f"kick_target_x={target_x} kick_target_y=0 "
        "kick_action_id=0 kick_sequence_id=0\n"
    )


def test_procedural_contact_does_not_credit_following_walk(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        _status(10, "DribbleTouch", 0.0)
        + _status(11, "DribbleTouch", 0.04)
        + _status(12, "None", 0.04)
        + _status(13, "None", 0.40),
        encoding="utf-8",
    )

    [outcome] = analyze_logs([log], kick_mode="DribbleTouch")

    assert outcome.last_cycle == 12
    assert outcome.forward_progress_m == 0.04
    assert not outcome.contact


def test_procedural_contact_is_measured_inside_action(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        _status(20, "DribbleTouch", 0.0)
        + _status(21, "DribbleTouch", 0.12)
        + _status(22, "None", 0.15),
        encoding="utf-8",
    )

    [outcome] = analyze_logs([log], kick_mode="DribbleTouch")

    assert outcome.last_cycle == 22
    assert outcome.forward_progress_m == 0.15
    assert outcome.contact


def test_targeted_pass_keeps_observing_post_command_coast(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        _status(30, "TargetedPass", 0.0, target_x=2.0)
        + _status(31, "None", 0.05, target_x=2.0)
        + _status(32, "None", 0.25, target_x=2.0),
        encoding="utf-8",
    )

    [outcome] = analyze_logs([log])

    assert outcome.last_cycle == 32
    assert outcome.forward_progress_m == 0.25
    assert outcome.contact
