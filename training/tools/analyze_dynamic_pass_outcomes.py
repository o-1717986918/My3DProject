#!/usr/bin/env python3
"""Pair Apollo rebuild dynamic-pass release states with server outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from my3d_rl.rcss_replay import load_ball_trajectory


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
    end_ball_valid: bool
    end_ball_age_s: float
    completed: bool
    upright: bool

    @property
    def measurement_valid(self) -> bool:
        return (
            self.peak_ball_speed_mps >= 0.0
            and self.final_ball_speed_mps >= 0.0
            and self.end_ball_valid
            and self.end_ball_age_s <= 0.10
        )

    @property
    def straight_2m_success(self) -> bool:
        return (
            self.completed
            and self.measurement_valid
            and self.upright
            and 1.5 <= self.ball_dx_m <= 2.5
            and abs(self.ball_dy_m) <= 0.30
        )

    @property
    def forward_drive_success(self) -> bool:
        """Match the training clearance/drive contract without hiding falls.

        Falling is reported independently through ``upright``.  An occasional
        fall is a finite cost, not a reason to erase a useful ball outcome.
        """

        return (
            self.completed
            and self.measurement_valid
            and self.ball_dx_m >= 2.5
            and abs(self.ball_dy_m) <= 1.0
            and self.peak_ball_speed_mps >= 1.5
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
                    end_ball_valid=result["end_ball_valid"] == "1",
                    end_ball_age_s=float(result["end_ball_age"]),
                    completed=result["termination"] == "completed",
                    upright=result["upright"] == "1",
                )
            )
    return outcomes


def summarize(outcomes: list[DynamicPassOutcome]) -> dict[str, object]:
    if not outcomes:
        return {"outcome_count": 0}

    def subset_summary(
        subset: list[DynamicPassOutcome],
    ) -> dict[str, int | float]:
        def median(name: str) -> float:
            return float(np.median([getattr(outcome, name) for outcome in subset]))

        return {
            "outcome_count": len(subset),
            "completed_count": sum(outcome.completed for outcome in subset),
            "measurement_valid_count": sum(
                outcome.measurement_valid for outcome in subset
            ),
            "upright_count": sum(outcome.upright for outcome in subset),
            "straight_2m_success_count": sum(
                outcome.straight_2m_success for outcome in subset
            ),
            "forward_drive_success_count": sum(
                outcome.forward_drive_success for outcome in subset
            ),
            "median_ball_dx_m": median("ball_dx_m"),
            "median_abs_ball_dy_m": float(
                np.median([abs(outcome.ball_dy_m) for outcome in subset])
            ),
            "median_ball_displacement_m": median("ball_displacement_m"),
            "median_peak_ball_speed_mps": median("peak_ball_speed_mps"),
        }

    rollout_ids = sorted({outcome.rollout_id for outcome in outcomes})
    overall = subset_summary(outcomes)

    return {
        **overall,
        "rollout_counts": {
            str(rollout_id): sum(
                outcome.rollout_id == rollout_id for outcome in outcomes
            )
            for rollout_id in rollout_ids
        },
        "rollout_summaries": {
            str(rollout_id): subset_summary(
                [
                    outcome
                    for outcome in outcomes
                    if outcome.rollout_id == rollout_id
                ]
            )
            for rollout_id in rollout_ids
        },
    }


def summarize_ground_truth(
    outcomes: list[DynamicPassOutcome],
    replay_times_s: np.ndarray,
    replay_ball_position_m: np.ndarray,
    *,
    team_side: str,
) -> dict[str, object]:
    """Measure action windows against server truth, independent of vision age."""

    times = np.asarray(replay_times_s, dtype=np.float64)
    positions = np.asarray(replay_ball_position_m, dtype=np.float64)
    if (
        team_side not in {"left", "right"}
        or times.ndim != 1
        or positions.shape != (times.size, 3)
        or times.size < 2
        or not np.isfinite(times).all()
        or not np.isfinite(positions).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("invalid RCSS ground-truth trajectory")
    canonical_sign = 1.0 if team_side == "left" else -1.0
    rows: list[dict[str, float | int | bool]] = []
    for outcome in outcomes:
        start = int(np.argmin(np.abs(times - outcome.start_time_s)))
        end = int(np.argmin(np.abs(times - outcome.end_time_s)))
        if end <= start or abs(times[start] - outcome.start_time_s) > 0.05:
            continue
        segment = positions[start : end + 1]
        delta = canonical_sign * (segment[:, :2] - segment[0, :2])
        directional_speed = canonical_sign * np.diff(segment[:, 0]) / np.diff(
            times[start : end + 1]
        )
        maximum_progress = float(np.max(delta[:, 0]))
        final_dx = float(delta[-1, 0])
        final_dy = float(delta[-1, 1])
        peak_directional_speed = float(np.max(directional_speed))
        success = bool(
            outcome.completed
            and maximum_progress >= 2.5
            and abs(final_dy) <= 1.0
            and peak_directional_speed >= 1.5
        )
        rows.append(
            {
                "player": outcome.player,
                "rollout_id": outcome.rollout_id,
                "start_time_s": outcome.start_time_s,
                "end_time_s": outcome.end_time_s,
                "maximum_progress_m": maximum_progress,
                "final_ball_dx_m": final_dx,
                "final_ball_dy_m": final_dy,
                "peak_directional_speed_mps": peak_directional_speed,
                "forward_drive_success": success,
                "upright": outcome.upright,
            }
        )
    return {
        "source": "RCSSServerMJ RSMP replay scene graph",
        "team_side": team_side,
        "outcome_count": len(rows),
        "forward_drive_success_count": sum(
            bool(row["forward_drive_success"]) for row in rows
        ),
        "upright_count": sum(bool(row["upright"]) for row in rows),
        "outcomes": rows,
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
        end_ball_valid=np.asarray([outcome.end_ball_valid for outcome in outcomes]),
        end_ball_age_s=np.asarray(
            [outcome.end_ball_age_s for outcome in outcomes], dtype=np.float32
        ),
        measurement_valid=np.asarray(
            [outcome.measurement_valid for outcome in outcomes]
        ),
        completed=np.asarray([outcome.completed for outcome in outcomes]),
        upright=np.asarray([outcome.upright for outcome in outcomes]),
        straight_2m_success=np.asarray(
            [outcome.straight_2m_success for outcome in outcomes]
        ),
        forward_drive_success=np.asarray(
            [outcome.forward_drive_success for outcome in outcomes]
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
    parser.add_argument("--server-replay", type=Path)
    parser.add_argument("--team-side", choices=("left", "right"), default="left")
    args = parser.parse_args()

    outcomes = [
        outcome
        for match_dir in args.match_dir
        for outcome in load_outcomes(match_dir, args.team_prefix)
    ]
    if args.output_npz is not None:
        save_npz(args.output_npz, outcomes)
    report = summarize(outcomes)
    if args.server_replay is not None:
        replay_times, replay_positions = load_ball_trajectory(args.server_replay)
        report["server_ground_truth"] = summarize_ground_truth(
            outcomes,
            replay_times,
            replay_positions,
            team_side=args.team_side,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
