#!/usr/bin/env python3
"""Summarize a developed-vs-pristine Apollo web match from retained logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics


FINAL_SCORE = re.compile(
    r"Final score:\s+(?P<left>.+?)\s+(?P<left_score>\d+)\s+-\s+"
    r"(?P<right_score>\d+)\s+(?P<right>.+?)\s*$"
)
ILLEGAL_DEFENSE = re.compile(r"Illegal defense: penalizing (?P<side>[lr])-\d+-")


def _fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def analyze(run_dir: Path, current_team: str) -> dict[str, object]:
    server_path = run_dir / "server.log"
    if not server_path.is_file():
        raise FileNotFoundError(server_path)

    score: dict[str, object] | None = None
    illegal_defense = Counter[str]()
    for line in server_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := FINAL_SCORE.search(line):
            score = {
                "left_team": match.group("left"),
                "left_score": int(match.group("left_score")),
                "right_score": int(match.group("right_score")),
                "right_team": match.group("right"),
            }
        if match := ILLEGAL_DEFENSE.search(line):
            illegal_defense[match.group("side")] += 1

    status_fields = {
        "motion": Counter[str](),
        "strategy": Counter[str](),
        "duty": Counter[str](),
        "risk_mode": Counter[str](),
        "execution": Counter[str](),
        "phase": Counter[str](),
        "possession": Counter[str](),
    }
    setup_phases = Counter[str]()
    execution_event_motion = Counter[str]()
    execution_event_status = Counter[str]()
    execution_event_kick_mode = Counter[str]()
    visible_ball_x_by_cycle: dict[int, list[float]] = {}
    player_get_up_samples = Counter[str]()
    player_get_up_episodes = Counter[str]()
    get_up_entries_by_previous_motion = Counter[str]()
    status_samples = 0
    for log_path in sorted(run_dir.glob(f"{current_team}-*.log")):
        player = log_path.stem.rsplit("-", 1)[-1]
        previous_status_motion: str | None = None
        for line in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("MY3D_STATUS"):
                status_samples += 1
                values = _fields(line)
                for name in status_fields:
                    if value := values.get(name):
                        status_fields[name][value] += 1
                if values.get("motion") == "GetUpRL":
                    player_get_up_samples[player] += 1
                    if previous_status_motion != "GetUpRL":
                        player_get_up_episodes[player] += 1
                        get_up_entries_by_previous_motion[
                            previous_status_motion or "Unknown"
                        ] += 1
                if (
                    values.get("ball_visible") == "1"
                    and values.get("ball_position_valid") == "1"
                    and "cycle" in values
                    and "ball_x" in values
                ):
                    try:
                        cycle = int(values["cycle"])
                        ball_x = float(values["ball_x"])
                    except ValueError:
                        pass
                    else:
                        visible_ball_x_by_cycle.setdefault(cycle, []).append(ball_x)
                previous_status_motion = values.get("motion")
            elif line.startswith("MY3D_KICK_SETUP"):
                values = _fields(line)
                if phase := values.get("phase"):
                    setup_phases[phase] += 1
            elif line.startswith("MY3D_EXECUTION_EVENT"):
                values = _fields(line)
                if motion := values.get("motion"):
                    execution_event_motion[motion] += 1
                if status := values.get("status"):
                    execution_event_status[status] += 1
                if kick_mode := values.get("kick_mode"):
                    execution_event_kick_mode[kick_mode] += 1

    motions = status_fields["motion"]
    exact_kick_samples = sum(
        count
        for name, count in motions.items()
        if name.startswith(("LearnedKick", "ProceduralKick", "Parameterized"))
    )
    fallback_kick_samples = sum(
        count for name, count in motions.items() if name.startswith("FallbackKick")
    )
    cycle_ball_x = [
        statistics.median(samples)
        for _, samples in sorted(visible_ball_x_by_cycle.items())
        if samples
    ]
    ball_progress = {
        "visible_cycles": len(cycle_ball_x),
        "minimum_x_m": min(cycle_ball_x) if cycle_ball_x else None,
        "maximum_x_m": max(cycle_ball_x) if cycle_ball_x else None,
        "median_x_m": statistics.median(cycle_ball_x) if cycle_ball_x else None,
        "opponent_half_cycles": sum(x > 0.0 for x in cycle_ball_x),
        "opponent_half_fraction": (
            sum(x > 0.0 for x in cycle_ball_x) / len(cycle_ball_x)
            if cycle_ball_x
            else None
        ),
    }
    return {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "score": score,
        "server": {
            "illegal_defense_left": illegal_defense["l"],
            "illegal_defense_right": illegal_defense["r"],
        },
        "developed_team": {
            "name": current_team,
            "status_samples": status_samples,
            **{
                name: dict(counter.most_common())
                for name, counter in status_fields.items()
            },
            "kick_setup_phase": dict(setup_phases.most_common()),
            "execution_events": {
                "motion": dict(execution_event_motion.most_common()),
                "status": dict(execution_event_status.most_common()),
                "kick_mode": dict(execution_event_kick_mode.most_common()),
            },
            "visible_ball_progress": ball_progress,
            "exact_kick_samples": exact_kick_samples,
            "fallback_kick_samples": fallback_kick_samples,
            "players_with_get_up_samples": sorted(player_get_up_samples),
            "get_up_samples_by_player": dict(player_get_up_samples),
            "get_up_episodes_by_player": dict(player_get_up_episodes),
            "get_up_entries_by_previous_motion": dict(
                get_up_entries_by_previous_motion.most_common()
            ),
        },
        "interpretation_limits": [
            "pristine Apollo is silent, so its internal decisions are not inferred",
            "status counts are samples, not independent physical events",
            "visible-ball progress is the per-cycle median of developed-team observers, not server ground truth",
            "score and referee events come from the authoritative server log",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--current-team", default="My3D-Current")
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir, args.current_team), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
