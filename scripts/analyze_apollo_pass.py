"""Measure physical outcomes of Apollo targeted-pass commands from status logs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PassOutcome:
    log: str
    action_id: int
    sequence_id: int
    release_cycle: int
    last_cycle: int
    target_distance_m: float
    forward_progress_m: float
    contact: bool


def parse_status_line(line: str) -> dict[str, str] | None:
    marker = "MY3D_STATUS "
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    values: dict[str, str] = {}
    for token in line[marker_index + len(marker) :].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def load_statuses(path: Path) -> list[dict[str, str]]:
    statuses: list[dict[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parsed = parse_status_line(line)
            if parsed is not None:
                statuses.append(parsed)
    return statuses


def _number(status: dict[str, str], key: str) -> float:
    return float(status.get(key, "0"))


def analyze_logs(
    paths: Iterable[Path],
    *,
    minimum_progress_m: float = 0.1,
    outcome_window_cycles: int = 100,
) -> list[PassOutcome]:
    outcomes: list[PassOutcome] = []
    for path in paths:
        statuses = load_statuses(path)
        consumed: set[tuple[int, int]] = set()
        for index, status in enumerate(statuses):
            if status.get("kick_mode") != "TargetedPass":
                continue
            key = (int(status.get("action_id", "0")), int(status.get("pass_seq", "0")))
            if key in consumed:
                continue
            consumed.add(key)
            start_cycle = int(status.get("cycle", "0"))
            start = (_number(status, "ball_x"), _number(status, "ball_y"))
            target = (
                _number(status, "pass_target_x"),
                _number(status, "pass_target_y"),
            )
            direction = (target[0] - start[0], target[1] - start[1])
            target_distance = math.hypot(*direction)
            if target_distance <= 1.0e-9:
                continue
            unit = (direction[0] / target_distance, direction[1] / target_distance)
            best_progress = 0.0
            last_cycle = start_cycle
            for later in statuses[index:]:
                cycle = int(later.get("cycle", "0"))
                if cycle > start_cycle + outcome_window_cycles:
                    break
                position = (_number(later, "ball_x"), _number(later, "ball_y"))
                progress = (position[0] - start[0]) * unit[0] + (
                    position[1] - start[1]
                ) * unit[1]
                best_progress = max(best_progress, progress)
                last_cycle = cycle
                # Perception is captured before the command shown on the same
                # line is applied, so include this sample and then stop before
                # a later non-target kick can contaminate the outcome.
                if later.get("kick_mode") not in {"None", "TargetedPass", None}:
                    break
            outcomes.append(
                PassOutcome(
                    log=str(path),
                    action_id=key[0],
                    sequence_id=key[1],
                    release_cycle=start_cycle,
                    last_cycle=last_cycle,
                    target_distance_m=target_distance,
                    forward_progress_m=best_progress,
                    contact=best_progress >= minimum_progress_m,
                )
            )
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--minimum-progress", type=float, default=0.1)
    parser.add_argument("--window-cycles", type=int, default=100)
    parser.add_argument(
        "--metric",
        choices=("attempts", "contacts"),
        help="Print only one integer metric for shell acceptance scripts.",
    )
    args = parser.parse_args()
    outcomes = analyze_logs(
        args.logs,
        minimum_progress_m=args.minimum_progress,
        outcome_window_cycles=args.window_cycles,
    )
    contacts = sum(outcome.contact for outcome in outcomes)
    if args.metric == "attempts":
        print(len(outcomes))
    elif args.metric == "contacts":
        print(contacts)
    else:
        print(
            json.dumps(
                {
                    "attempts": len(outcomes),
                    "contacts": contacts,
                    "outcomes": [asdict(outcome) for outcome in outcomes],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
