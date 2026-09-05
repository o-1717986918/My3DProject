#!/usr/bin/env python3
"""Summarize a developed-vs-pristine Apollo web match from retained logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


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
    }
    setup_phases = Counter[str]()
    player_get_up_samples = Counter[str]()
    status_samples = 0
    for log_path in sorted(run_dir.glob(f"{current_team}-*.log")):
        player = log_path.stem.rsplit("-", 1)[-1]
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
            elif line.startswith("MY3D_KICK_SETUP"):
                values = _fields(line)
                if phase := values.get("phase"):
                    setup_phases[phase] += 1

    motions = status_fields["motion"]
    exact_kick_samples = sum(
        count
        for name, count in motions.items()
        if name.startswith(("LearnedKick", "ProceduralKick", "Parameterized"))
    )
    fallback_kick_samples = sum(
        count for name, count in motions.items() if name.startswith("FallbackKick")
    )
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
            "exact_kick_samples": exact_kick_samples,
            "fallback_kick_samples": fallback_kick_samples,
            "players_with_get_up_samples": sorted(player_get_up_samples),
            "get_up_samples_by_player": dict(player_get_up_samples),
        },
        "interpretation_limits": [
            "pristine Apollo is silent, so its internal decisions are not inferred",
            "status counts are samples, not independent physical events",
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
