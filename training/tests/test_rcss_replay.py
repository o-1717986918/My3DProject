from __future__ import annotations

from pathlib import Path

import numpy as np

from my3d_rl.rcss_replay import load_ball_trajectory


def _frame(time_s: float, mode: str, ball_slt: str | None) -> str:
    identity = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    root_attribute = f" TRF(SLT {identity})" if mode == "full" else ""
    description = " DSC(ball)" if mode == "full" else ""
    transform = f" TRF(SLT {ball_slt})" if ball_slt is not None else ""
    return (
        f"((RSMP 1 0)((gt 1 0){time_s:.2f})"
        f"((sg 1 0){mode}(nd{root_attribute}(nd{description}(nd{transform})))))\n"
    )


def test_replay_ball_reader_applies_diff_pose_cache(tmp_path: Path) -> None:
    replay = tmp_path / "server-replay"
    replay.write_text(
        _frame(0.04, "full", "1 0 0 0 0 1 0 0 0 0 1 0 1 2 0.11 1")
        + _frame(0.08, "diff", None)
        + _frame(0.12, "diff", "1 0 0 0 0 1 0 0 0 0 1 0 1.5 2.2 0.2 1"),
        encoding="utf-8",
    )

    times, positions = load_ball_trajectory(replay)

    np.testing.assert_allclose(times, [0.04, 0.08, 0.12])
    np.testing.assert_allclose(
        positions,
        [[1.0, 2.0, 0.11], [1.0, 2.0, 0.11], [1.5, 2.2, 0.2]],
    )
