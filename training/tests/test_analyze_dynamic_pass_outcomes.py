from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.analyze_dynamic_pass_outcomes import load_outcomes, save_npz, summarize


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
    assert outcomes[0].straight_2m_success
    summary = summarize(outcomes)
    assert summary["straight_2m_success_count"] == 1
    assert summary["rollout_summaries"] == {
        "79": {
            "outcome_count": 1,
            "completed_count": 1,
            "upright_count": 1,
            "straight_2m_success_count": 1,
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


def test_unmatched_result_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "Apollo-Rebuild-2.log").write_text(
        "APOLLO_REBUILD_DYNAMIC_PASS_RESULT "
        "t=5 player=2 motion=DynamicPass-r4 termination=interrupted\n",
        encoding="utf-8",
    )

    assert load_outcomes(tmp_path, "Apollo-Rebuild") == []
    assert summarize([]) == {"outcome_count": 0}
