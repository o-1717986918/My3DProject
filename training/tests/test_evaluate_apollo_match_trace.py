from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.evaluate_apollo_match_trace import (
    TraceFrame,
    expand_trace,
    load_match_traces,
    velocity_command_from_status,
)


def test_velocity_command_matches_absolute_body_frame_and_clamps() -> None:
    command = velocity_command_from_status(
        {
            "x": "1.0",
            "y": "2.0",
            "yaw_deg": "90.0",
            "walk_target_x": "3.0",
            "walk_target_y": "2.0",
            "walk_target_absolute": "1",
            "walk_orientation_present": "0",
        }
    )

    np.testing.assert_allclose(command, [0.0, -0.5, -np.pi / 10.0], atol=1e-6)


def test_expand_trace_uses_status_timestamp_holds() -> None:
    frames = [
        TraceFrame(1.0, np.array([1.0, 0.0, 0.0]), np.zeros(2), False),
        TraceFrame(1.1, np.array([0.0, 0.5, 0.0]), np.ones(2), True),
    ]

    commands, balls, near = expand_trace(frames)

    assert commands.shape == (10, 3)
    np.testing.assert_array_equal(commands[:5], np.tile([1.0, 0.0, 0.0], (5, 1)))
    np.testing.assert_array_equal(balls[5:], np.ones((5, 2)))
    np.testing.assert_array_equal(near, [False] * 5 + [True] * 5)


def test_load_match_traces_splits_non_walk_states(tmp_path: Path) -> None:
    log = tmp_path / "Apollo-Rebuild-2.log"
    base = (
        "player=2 mode=4 role=6 ball_dist=0.5 ball_x=1 ball_y=0 "
        "x=0 y=0 yaw_deg=0 walk_target_x=2 walk_target_y=0 "
        "walk_target_absolute=1 walk_orientation_present=0"
    )
    log.write_text(
        "\n".join(
            [
                f"APOLLO_REBUILD_STATUS t=1.0 motion=Walk {base}",
                f"APOLLO_REBUILD_STATUS t=1.1 motion=Walk {base}",
                f"APOLLO_REBUILD_STATUS t=1.2 motion=GetUpRL {base}",
                f"APOLLO_REBUILD_STATUS t=1.3 motion=Walk {base}",
                f"APOLLO_REBUILD_STATUS t=1.4 motion=Walk {base}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    traces = load_match_traces(tmp_path, "Apollo-Rebuild")

    assert len(traces) == 2
    assert [len(frames) for _, frames in traces] == [2, 2]
