"""Measure physical outcomes of Apollo target-aware kick commands."""

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
    kick_mode: str
    action_id: int
    sequence_id: int
    release_cycle: int
    last_cycle: int
    target_distance_m: float
    forward_progress_m: float
    signed_lateral_error_m: float
    lateral_error_m: float
    signed_direction_error_deg: float
    direction_error_deg: float
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
    kick_mode: str = "TargetedPass",
    minimum_progress_m: float = 0.1,
    outcome_window_cycles: int = 100,
) -> list[PassOutcome]:
    outcomes: list[PassOutcome] = []
    for path in paths:
        statuses = load_statuses(path)
        consumed: set[tuple[int, int]] = set()
        for index, status in enumerate(statuses):
            if status.get("kick_mode") != kick_mode:
                continue
            start_cycle = int(status.get("cycle", "0"))
            action_key = (
                int(status.get("kick_action_id", status.get("action_id", "0"))),
                int(status.get("kick_sequence_id", status.get("pass_seq", "0"))),
            )
            if kick_mode == "TargetedPass":
                if action_key in consumed:
                    continue
                consumed.add(action_key)
            elif index > 0 and statuses[index - 1].get("kick_mode") == kick_mode:
                continue
            start = (_number(status, "ball_x"), _number(status, "ball_y"))
            target = (
                _number(
                    status,
                    "kick_target_x" if "kick_target_x" in status else "pass_target_x",
                ),
                _number(
                    status,
                    "kick_target_y" if "kick_target_y" in status else "pass_target_y",
                ),
            )
            direction = (target[0] - start[0], target[1] - start[1])
            target_distance = math.hypot(*direction)
            if target_distance <= 1.0e-9:
                continue
            unit = (direction[0] / target_distance, direction[1] / target_distance)
            lateral_unit = (-unit[1], unit[0])
            best_progress = 0.0
            signed_lateral_at_best_progress = 0.0
            last_cycle = start_cycle
            for later in statuses[index:]:
                cycle = int(later.get("cycle", "0"))
                if cycle > start_cycle + outcome_window_cycles:
                    break
                later_key = (
                    int(later.get("kick_action_id", later.get("action_id", "0"))),
                    int(
                        later.get(
                            "kick_sequence_id", later.get("pass_seq", "0")
                        )
                    ),
                )
                if (
                    cycle > start_cycle
                    and kick_mode == "TargetedPass"
                    and later.get("kick_mode") == kick_mode
                    and later_key != action_key
                    and later_key != (0, 0)
                ):
                    break
                position = (_number(later, "ball_x"), _number(later, "ball_y"))
                displacement = (
                    position[0] - start[0],
                    position[1] - start[1],
                )
                progress = displacement[0] * unit[0] + displacement[1] * unit[1]
                if progress > best_progress:
                    best_progress = progress
                    signed_lateral_at_best_progress = (
                        displacement[0] * lateral_unit[0]
                        + displacement[1] * lateral_unit[1]
                    )
                last_cycle = cycle
                # Perception is captured before the command shown on the same
                # line is applied, so include this sample and then stop before
                # a later non-target kick can contaminate the outcome.
                if later.get("kick_mode") not in {"None", kick_mode, None}:
                    break
                # A procedural contact must move the ball during its own
                # command lifetime. Do not credit the subsequent chase gait
                # with a contact that the standalone trajectory did not make.
                if (
                    kick_mode != "TargetedPass"
                    and cycle > start_cycle
                    and later.get("kick_mode") != kick_mode
                ):
                    break
            outcomes.append(
                PassOutcome(
                    log=str(path),
                    kick_mode=kick_mode,
                    action_id=action_key[0],
                    sequence_id=action_key[1],
                    release_cycle=start_cycle,
                    last_cycle=last_cycle,
                    target_distance_m=target_distance,
                    forward_progress_m=best_progress,
                    signed_lateral_error_m=signed_lateral_at_best_progress,
                    lateral_error_m=abs(signed_lateral_at_best_progress),
                    signed_direction_error_deg=math.degrees(
                        math.atan2(
                            signed_lateral_at_best_progress, best_progress
                        )
                    )
                    if best_progress > 1.0e-9
                    else 0.0,
                    direction_error_deg=math.degrees(
                        math.atan2(
                            abs(signed_lateral_at_best_progress), best_progress
                        )
                    )
                    if best_progress > 1.0e-9
                    else 0.0,
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
        "--kick-mode",
        choices=("TargetedPass", "DribbleTouch"),
        default="TargetedPass",
    )
    parser.add_argument(
        "--metric",
        choices=("attempts", "contacts"),
        help="Print only one integer metric for shell acceptance scripts.",
    )
    args = parser.parse_args()
    outcomes = analyze_logs(
        args.logs,
        kick_mode=args.kick_mode,
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
