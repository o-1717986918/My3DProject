from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_apollo_vs_base_match import analyze  # noqa: E402


def test_rebuild_status_contributes_visible_progress_and_near_ball(tmp_path: Path):
    (tmp_path / "server.log").write_text(
        "Final score: Apollo-Rebuild 1 - 0 Apollo-Base\n",
        encoding="utf-8",
    )
    (tmp_path / "Apollo-Rebuild-7.log").write_text(
        "\n".join(
            [
                "APOLLO_REBUILD_STATUS t=12.0 player=7 mode=4 role=6 "
                "ball_dist=0.50 ball_visible=1 ball_x=1.25 ball_y=0.1 "
                "x=0.75 y=0.1 z=0.8 self_speed=0.02 ball_speed=0.03 "
                "motion=Walk walk_target_norm=0.04 "
                "walk_target_absolute=0",
                "APOLLO_REBUILD_STATUS t=12.1 player=7 mode=4 role=6 "
                "ball_dist=0.60 ball_visible=1 ball_x=1.50 ball_y=0.1 "
                "x=0.90 y=0.1 z=0.8 self_speed=0.05 ball_speed=0.02 "
                "motion=Walk walk_target_norm=0.80 "
                "walk_target_absolute=0",
            ]
        ),
        encoding="utf-8",
    )

    report = analyze(tmp_path, "Apollo-Rebuild")

    developed = report["developed_team"]
    assert developed["visible_ball_progress"]["observation_buckets"] == 2
    assert developed["visible_ball_progress"]["maximum_x_m"] == 1.5
    assert developed["closest_visible_ball_distance"]["minimum_m"] == 0.5
    assert developed["closest_visible_ball_distance"]["within_1_1_m_buckets"] == 2
    assert developed["near_ball_response"]["samples"] == 2
    assert (
        developed["near_ball_response"]
        ["neutral_or_low_translation_command_samples"]
        == 1
    )
    assert developed["near_ball_response"]["role"] == {"6": 2}
    assert (
        developed["near_ball_response"]
        ["valid_translation_but_low_self_speed_samples"]
        == 1
    )
