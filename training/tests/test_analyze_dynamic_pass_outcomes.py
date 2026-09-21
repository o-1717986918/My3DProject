from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.analyze_dynamic_pass_outcomes import (
    load_outcomes,
    save_npz,
    summarize,
    summarize_ground_truth,
)


def test_pairs_release_observation_with_measured_result(tmp_path: Path) -> None:
    observation = ",".join(["0"] * 98)
    log = tmp_path / "Apollo-Rebuild-7.log"
    log.write_text(
        "\n".join(
            [
                "APOLLO_REBUILD_DYNAMIC_PASS_START "
                "t=10 player=7 motion=DynamicPass-r79 observation="
                + observation,
                "APOLLO_REBUILD_DYNAMIC_PASS_RESULT "
                "t=11.22 player=7 motion=DynamicPass-r79 "
                "termination=completed duration=1.22 selector_confidence=0.97 "
                "start_ball_speed=0 peak_ball_speed=2.1 final_ball_speed=1.2 "
                "ball_dx=1.9 ball_dy=0.2 ball_displacement=1.91 "
                "peak_ball_height=0.18 end_ball_valid=1 end_ball_age=0 upright=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outcomes = load_outcomes(tmp_path, "Apollo-Rebuild")

    assert len(outcomes) == 1
    assert outcomes[0].rollout_id == 79
    assert outcomes[0].measurement_valid
    assert outcomes[0].straight_2m_success
    assert not outcomes[0].forward_drive_success
    summary = summarize(outcomes)
    assert summary["straight_2m_success_count"] == 1
    assert summary["rollout_summaries"] == {
        "79": {
            "outcome_count": 1,
            "completed_count": 1,
            "measurement_valid_count": 1,
            "upright_count": 1,
            "straight_2m_success_count": 1,
            "forward_drive_success_count": 0,
            "median_ball_dx_m": 1.9,
            "median_abs_ball_dy_m": 0.2,
            "median_ball_displacement_m": 1.91,
            "median_peak_ball_speed_mps": 2.1,
        }
    }

    output = tmp_path / "outcomes.npz"
    save_npz(output, outcomes)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["observation"].shape == (1, 98)
        np.testing.assert_array_equal(archive["rollout_id"], [79])
        np.testing.assert_array_equal(archive["straight_2m_success"], [True])
        np.testing.assert_array_equal(archive["forward_drive_success"], [False])
        np.testing.assert_array_equal(archive["measurement_valid"], [True])


def test_forward_drive_keeps_ball_success_and_reports_fall_separately(
    tmp_path: Path,
) -> None:
    observation = ",".join(["0"] * 98)
    (tmp_path / "Apollo-Rebuild-7.log").write_text(
        "APOLLO_REBUILD_DYNAMIC_PASS_START "
        "t=10 player=7 motion=DynamicPass-r60467 observation="
        + observation
        + "\nAPOLLO_REBUILD_DYNAMIC_PASS_RESULT "
        "t=11.22 player=7 motion=DynamicPass-r60467 "
        "termination=completed duration=1.22 selector_confidence=1 "
        "start_ball_speed=0 peak_ball_speed=2.2 final_ball_speed=1.1 "
        "ball_dx=2.8 ball_dy=-0.4 ball_displacement=2.83 "
        "peak_ball_height=0.3 end_ball_valid=1 end_ball_age=0 upright=0\n",
        encoding="utf-8",
    )

    outcomes = load_outcomes(tmp_path, "Apollo-Rebuild")

    assert outcomes[0].forward_drive_success
    assert not outcomes[0].upright
    summary = summarize(outcomes)
    assert summary["forward_drive_success_count"] == 1
    assert summary["upright_count"] == 0


def test_stale_end_ball_cannot_count_as_success(tmp_path: Path) -> None:
    observation = ",".join(["0"] * 98)
    (tmp_path / "Apollo-Rebuild-7.log").write_text(
        "APOLLO_REBUILD_DYNAMIC_PASS_START "
        "t=10 player=7 motion=DynamicPass-r60467 observation="
        + observation
        + "\nAPOLLO_REBUILD_DYNAMIC_PASS_RESULT "
        "t=11.22 player=7 motion=DynamicPass-r60467 "
        "termination=completed duration=1.22 selector_confidence=1 "
        "start_ball_speed=-1 peak_ball_speed=2.2 final_ball_speed=-1 "
        "ball_dx=2.8 ball_dy=0 ball_displacement=2.8 "
        "peak_ball_height=0.3 end_ball_valid=1 end_ball_age=0.92 upright=1\n",
        encoding="utf-8",
    )

    outcome = load_outcomes(tmp_path, "Apollo-Rebuild")[0]

    assert not outcome.measurement_valid
    assert not outcome.forward_drive_success
    summary = summarize([outcome])
    assert summary["measurement_valid_count"] == 0
    assert summary["forward_drive_success_count"] == 0


def test_unmatched_result_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "Apollo-Rebuild-2.log").write_text(
        "APOLLO_REBUILD_DYNAMIC_PASS_RESULT "
        "t=5 player=2 motion=DynamicPass-r4 termination=interrupted\n",
        encoding="utf-8",
    )

    assert load_outcomes(tmp_path, "Apollo-Rebuild") == []
    assert summarize([]) == {"outcome_count": 0}


def test_ground_truth_uses_team_frame_and_does_not_require_upright(
    tmp_path: Path,
) -> None:
    observation = ",".join(["0"] * 98)
    (tmp_path / "Apollo-Rebuild-7.log").write_text(
        "APOLLO_REBUILD_DYNAMIC_PASS_START "
        "t=10 player=7 motion=DynamicPass-r60467 observation="
        + observation
        + "\nAPOLLO_REBUILD_DYNAMIC_PASS_RESULT "
        "t=11 player=7 motion=DynamicPass-r60467 "
        "termination=completed duration=1 selector_confidence=1 "
        "start_ball_speed=-1 peak_ball_speed=-1 final_ball_speed=-1 "
        "ball_dx=0 ball_dy=0 ball_displacement=0 peak_ball_height=0.11 "
        "end_ball_valid=1 end_ball_age=1 upright=0\n",
        encoding="utf-8",
    )
    outcome = load_outcomes(tmp_path, "Apollo-Rebuild")[0]
    times = np.asarray([9.96, 10.0, 10.5, 11.0])
    # A right-side team attacks toward global -x; canonical y is also flipped.
    positions = np.asarray(
        [[4.0, 1.0, 0.11], [4.0, 1.0, 0.11], [2.0, 0.8, 0.11], [1.2, 0.6, 0.11]]
    )

    report = summarize_ground_truth(
        [outcome], times, positions, team_side="right"
    )

    assert report["forward_drive_success_count"] == 1
    assert report["upright_count"] == 0
    assert report["outcomes"][0]["maximum_progress_m"] == pytest.approx(2.8)
    assert report["outcomes"][0]["final_ball_dy_m"] == pytest.approx(0.4)
