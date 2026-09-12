#!/usr/bin/env python3
"""Pair Apollo rebuild dynamic-pass release states with server outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


START_PREFIX = "APOLLO_REBUILD_DYNAMIC_PASS_START "
RESULT_PREFIX = "APOLLO_REBUILD_DYNAMIC_PASS_RESULT "
OBSERVATION_SIZE = 98


def _fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


@dataclass(frozen=True)
class DynamicPassOutcome:
    observation: np.ndarray
    player: int
    motion: str
    rollout_id: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    selector_confidence: float
    start_ball_speed_mps: float
    peak_ball_speed_mps: float
    final_ball_speed_mps: float
    ball_dx_m: float
    ball_dy_m: float
    ball_displacement_m: float
    peak_ball_height_m: float
    completed: bool
    upright: bool

    @property
    def straight_2m_success(self) -> bool:
        return (
            self.completed
            and self.upright
            and 1.5 <= self.ball_dx_m <= 2.5
            and abs(self.ball_dy_m) <= 0.30
        )


def _parse_rollout_id(motion: str) -> int:
    prefix = "DynamicPass-r"
    if not motion.startswith(prefix):
        raise ValueError(f"invalid dynamic-pass motion name: {motion}")
    return int(motion[len(prefix) :])


def load_outcomes(match_dir: Path, team_prefix: str) -> list[DynamicPassOutcome]:
    """Load completed START/RESULT pairs from per-player runtime logs."""

    paths = sorted(match_dir.glob(f"{team_prefix}-*.log"))
    if not paths:
        raise FileNotFoundError(
            f"no {team_prefix}-*.log files found in {match_dir}"
        )

    outcomes: list[DynamicPassOutcome] = []
    for path in paths:
        pending: dict[str, list[tuple[dict[str, str], np.ndarray]]] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(START_PREFIX):
                values = _fields(line)
                motion = values["motion"]
                observation = np.fromstring(
                    values["observation"], sep=",", dtype=np.float32
                )
                if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(
                    observation
                ).all():
                    raise ValueError(f"invalid selector observation in {path}")
                pending.setdefault(motion, []).append((values, observation))
                continue
            if not line.startswith(RESULT_PREFIX):
                continue
            result = _fields(line)
            motion = result["motion"]
            starts = pending.get(motion, [])
            if not starts:
                continue
            start, observation = starts.pop(0)
            outcomes.append(
                DynamicPassOutcome(
                    observation=observation,
                    player=int(result["player"]),
                    motion=motion,
                    rollout_id=_parse_rollout_id(motion),
                    start_time_s=float(start["t"]),
                    end_time_s=float(result["t"]),
                    duration_s=float(result["duration"]),
                    selector_confidence=float(result["selector_confidence"]),
                    start_ball_speed_mps=float(result["start_ball_speed"]),
                    peak_ball_speed_mps=float(result["peak_ball_speed"]),
                    final_ball_speed_mps=float(result["final_ball_speed"]),
                    ball_dx_m=float(result["ball_dx"]),
                    ball_dy_m=float(result["ball_dy"]),
                    ball_displacement_m=float(result["ball_displacement"]),
                    peak_ball_height_m=float(result["peak_ball_height"]),
                    completed=result["termination"] == "completed",
                    upright=result["upright"] == "1",
                )
            )
    return outcomes


def summarize(outcomes: list[DynamicPassOutcome]) -> dict[str, object]:
    if not outcomes:
        return {"outcome_count": 0}

    def median(name: str) -> float:
        return float(np.median([getattr(outcome, name) for outcome in outcomes]))

    return {
        "outcome_count": len(outcomes),
        "completed_count": sum(outcome.completed for outcome in outcomes),
        "upright_count": sum(outcome.upright for outcome in outcomes),
        "straight_2m_success_count": sum(
            outcome.straight_2m_success for outcome in outcomes
        ),
        "median_ball_dx_m": median("ball_dx_m"),
        "median_abs_ball_dy_m": float(
            np.median([abs(outcome.ball_dy_m) for outcome in outcomes])
        ),
        "median_ball_displacement_m": median("ball_displacement_m"),
        "median_peak_ball_speed_mps": median("peak_ball_speed_mps"),
        "rollout_counts": {
            str(rollout_id): sum(
                outcome.rollout_id == rollout_id for outcome in outcomes
            )
            for rollout_id in sorted({outcome.rollout_id for outcome in outcomes})
        },
    }


def save_npz(path: Path, outcomes: list[DynamicPassOutcome]) -> None:
    if not outcomes:
        raise ValueError("cannot save an empty outcome corpus")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observation=np.stack([outcome.observation for outcome in outcomes]),
        player=np.asarray([outcome.player for outcome in outcomes], dtype=np.int16),
        rollout_id=np.asarray(
            [outcome.rollout_id for outcome in outcomes], dtype=np.int32
        ),
        start_time_s=np.asarray(
            [outcome.start_time_s for outcome in outcomes], dtype=np.float64
        ),
        end_time_s=np.asarray(
            [outcome.end_time_s for outcome in outcomes], dtype=np.float64
        ),
        duration_s=np.asarray(
            [outcome.duration_s for outcome in outcomes], dtype=np.float32
        ),
        selector_confidence=np.asarray(
            [outcome.selector_confidence for outcome in outcomes], dtype=np.float32
        ),
        start_ball_speed_mps=np.asarray(
            [outcome.start_ball_speed_mps for outcome in outcomes],
            dtype=np.float32,
        ),
        peak_ball_speed_mps=np.asarray(
            [outcome.peak_ball_speed_mps for outcome in outcomes],
            dtype=np.float32,
        ),
        final_ball_speed_mps=np.asarray(
            [outcome.final_ball_speed_mps for outcome in outcomes],
            dtype=np.float32,
        ),
        ball_dx_m=np.asarray(
            [outcome.ball_dx_m for outcome in outcomes], dtype=np.float32
        ),
        ball_dy_m=np.asarray(
            [outcome.ball_dy_m for outcome in outcomes], dtype=np.float32
        ),
        ball_displacement_m=np.asarray(
            [outcome.ball_displacement_m for outcome in outcomes],
            dtype=np.float32,
        ),
        peak_ball_height_m=np.asarray(
            [outcome.peak_ball_height_m for outcome in outcomes],
            dtype=np.float32,
        ),
        completed=np.asarray([outcome.completed for outcome in outcomes]),
        upright=np.asarray([outcome.upright for outcome in outcomes]),
        straight_2m_success=np.asarray(
            [outcome.straight_2m_success for outcome in outcomes]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-dir",
        type=Path,
        nargs="+",
        required=True,
        help="one or more RCSSServerMJ match-log directories",
    )
    parser.add_argument("--team-prefix", default="Apollo-Rebuild")
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()

    outcomes = [
        outcome
        for match_dir in args.match_dir
        for outcome in load_outcomes(match_dir, args.team_prefix)
    ]
    if args.output_npz is not None:
        save_npz(args.output_npz, outcomes)
    print(json.dumps(summarize(outcomes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
