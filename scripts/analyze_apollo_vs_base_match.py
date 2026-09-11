#!/usr/bin/env python3
"""Summarize a developed-vs-pristine Apollo web match from retained logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics


FINAL_SCORE = re.compile(
    r"Final score:\s+(?P<left>.+?)\s+(?P<left_score>\d+)\s+-\s+"
    r"(?P<right_score>\d+)\s+(?P<right>.+?)\s*$"
)
ILLEGAL_DEFENSE = re.compile(
    r"Illegal defense: penalizing .* from (?P<side>left|right) goalie area"
)
REBUILD_PLAY_ON_MODE = "4"


def _fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def _ball_progress(
    ball_x_by_observation: dict[int, list[float]],
) -> dict[str, object]:
    observations = [
        statistics.median(samples)
        for _, samples in sorted(ball_x_by_observation.items())
        if samples
    ]
    return {
        "observation_buckets": len(observations),
        "minimum_x_m": min(observations) if observations else None,
        "maximum_x_m": max(observations) if observations else None,
        "median_x_m": statistics.median(observations) if observations else None,
        "opponent_half_buckets": sum(x > 0.0 for x in observations),
        "opponent_half_fraction": (
            sum(x > 0.0 for x in observations) / len(observations)
            if observations
            else None
        ),
    }


def _step_distribution(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    if not ordered:
        return {
            "samples": 0,
            "median_m": None,
            "p90_m": None,
            "maximum_m": None,
            "above_0_25_m": 0,
            "above_1_0_m": 0,
        }
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "samples": len(ordered),
        "median_m": statistics.median(ordered),
        "p90_m": ordered[p90_index],
        "maximum_m": ordered[-1],
        "above_0_25_m": sum(value > 0.25 for value in ordered),
        "above_1_0_m": sum(value > 1.0 for value in ordered),
    }


def _closest_ball_distance(
    distance_by_observation: dict[int, list[float]],
) -> dict[str, object]:
    distances = [
        min(samples)
        for _, samples in sorted(distance_by_observation.items())
        if samples
    ]
    return {
        "observation_buckets": len(distances),
        "minimum_m": min(distances) if distances else None,
        "median_m": statistics.median(distances) if distances else None,
        "within_1_1_m_buckets": sum(value <= 1.10 for value in distances),
        "within_1_5_m_buckets": sum(value <= 1.50 for value in distances),
        "within_2_0_m_buckets": sum(value <= 2.00 for value in distances),
    }


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
        "role": Counter[str](),
        "strategy": Counter[str](),
        "duty": Counter[str](),
        "risk_mode": Counter[str](),
        "execution": Counter[str](),
        "restart_phase": Counter[str](),
        "phase": Counter[str](),
        "possession": Counter[str](),
    }
    strategy_motion = Counter[str]()
    setup_phases = Counter[str]()
    setup_phases_by_mode: dict[str, Counter[str]] = {}
    execution_event_motion = Counter[str]()
    execution_event_status = Counter[str]()
    execution_event_kick_mode = Counter[str]()
    visible_ball_x_by_observation: dict[int, list[float]] = {}
    fresh_ball_x_by_observation: dict[int, list[float]] = {}
    bounded_ball_x_by_observation: dict[int, list[float]] = {}
    visible_ball_dist_by_observation: dict[int, list[float]] = {}
    player_get_up_samples = Counter[str]()
    player_get_up_episodes = Counter[str]()
    get_up_entries_by_previous_motion = Counter[str]()
    duty_switches = 0
    plan_revision_switches = 0
    tactical_target_steps_m: list[float] = []
    near_ball_samples = 0
    near_ball_formation_samples = 0
    near_ball_pressure_samples = 0
    near_ball_low_command_samples = 0
    near_ball_pure_turn_samples = 0
    near_ball_idle_samples = 0
    near_ball_valid_command_low_speed_samples = 0
    near_ball_stationary_ball_low_speed_samples = 0
    near_ball_role = Counter[str]()
    status_samples = 0
    for log_path in sorted(run_dir.glob(f"{current_team}-*.log")):
        player = log_path.stem.rsplit("-", 1)[-1]
        previous_status_motion: str | None = None
        previous_duty: str | None = None
        previous_plan_revision: str | None = None
        previous_tactical_target: tuple[float, float] | None = None
        for line in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith(("MY3D_STATUS", "APOLLO_REBUILD_STATUS")):
                status_samples += 1
                values = _fields(line)
                for name in status_fields:
                    if value := values.get(name):
                        status_fields[name][value] += 1
                if (strategy := values.get("strategy")) and (
                    motion := values.get("motion")
                ):
                    strategy_motion[f"{strategy}->{motion}"] += 1
                if values.get("motion") == "GetUpRL":
                    player_get_up_samples[player] += 1
                    if previous_status_motion != "GetUpRL":
                        player_get_up_episodes[player] += 1
                        get_up_entries_by_previous_motion[
                            previous_status_motion or "Unknown"
                        ] += 1
                duty = values.get("duty")
                if previous_duty is not None and duty != previous_duty:
                    duty_switches += 1
                if duty is not None:
                    previous_duty = duty
                plan_revision = values.get("plan_revision")
                if (
                    previous_plan_revision is not None
                    and plan_revision != previous_plan_revision
                ):
                    plan_revision_switches += 1
                if plan_revision is not None:
                    previous_plan_revision = plan_revision
                try:
                    tactical_target = (
                        float(values["tactical_target_x"]),
                        float(values["tactical_target_y"]),
                    )
                except (KeyError, ValueError):
                    tactical_target = None
                if (
                    previous_tactical_target is not None
                    and tactical_target is not None
                    and all(math.isfinite(value) for value in tactical_target)
                ):
                    tactical_target_steps_m.append(math.dist(
                        previous_tactical_target, tactical_target
                    ))
                if (
                    tactical_target is not None
                    and all(math.isfinite(value) for value in tactical_target)
                ):
                    previous_tactical_target = tactical_target
                ball_visible = values.get("ball_visible") == "1"
                ball_position_valid = (
                    values.get("ball_position_valid") == "1"
                    or ("ball_position_valid" not in values and ball_visible)
                )
                if ball_position_valid and (
                    "server_time" in values
                    or "t" in values
                    or "cycle" in values
                ) and "ball_x" in values:
                    try:
                        ball_x = float(values["ball_x"])
                        ball_age_s = float(values.get("ball_position_age", "inf"))
                        if "server_time" in values or "t" in values:
                            # Clients start on staggered cycles.  A 100 ms
                            # server-time bucket merges near-simultaneous team
                            # observations without pretending local cycle ids
                            # describe the same world instant.
                            observation = int(round(
                                float(values.get("server_time", values["t"]))
                                * 10.0))
                        else:
                            observation = int(values["cycle"])
                    except ValueError:
                        pass
                    else:
                        if math.isfinite(ball_x):
                            visible = ball_visible
                            near_contact = (
                                values.get("ball_near_contact_track") == "1"
                            )
                            if visible:
                                visible_ball_x_by_observation.setdefault(
                                    observation, []).append(ball_x)
                                try:
                                    ball_dist = float(values["ball_dist"])
                                except (KeyError, ValueError):
                                    pass
                                else:
                                    if math.isfinite(ball_dist):
                                        visible_ball_dist_by_observation.setdefault(
                                            observation, []).append(ball_dist)
                            if visible or ball_age_s <= 0.75:
                                fresh_ball_x_by_observation.setdefault(
                                    observation, []).append(ball_x)
                            if (
                                visible
                                or ball_age_s <= 0.75
                                or (near_contact and ball_age_s <= 3.5)
                            ):
                                bounded_ball_x_by_observation.setdefault(
                                    observation, []).append(ball_x)
                try:
                    play_on = (
                        values.get("play_on") == "1"
                        or values.get("mode") == REBUILD_PLAY_ON_MODE
                    )
                    fresh_near_ball = (
                        play_on
                        and ball_position_valid
                        and (
                            ball_visible
                            or float(values.get("ball_position_age", "inf"))
                            <= 0.75
                        )
                        and math.dist(
                            (float(values["x"]), float(values["y"])),
                            (float(values["ball_x"]), float(values["ball_y"])),
                        ) <= 1.10
                    )
                except (KeyError, ValueError):
                    fresh_near_ball = False
                if fresh_near_ball:
                    near_ball_samples += 1
                    near_ball_role[values.get("role", "Unknown")] += 1
                    if values.get("duty") == "Formation":
                        near_ball_formation_samples += 1
                    if values.get("duty") == "Pressure":
                        near_ball_pressure_samples += 1
                    try:
                        if "walk_target_norm" in values:
                            command_norm = float(values["walk_target_norm"])
                        elif values.get("walk_target_absolute") == "1":
                            command_norm = math.dist(
                                (float(values["x"]), float(values["y"])),
                                (
                                    float(values["walk_target_x"]),
                                    float(values["walk_target_y"]),
                                ),
                            )
                        else:
                            command_norm = math.hypot(
                                float(values["walk_target_x"]),
                                float(values["walk_target_y"]),
                            )
                    except (KeyError, ValueError):
                        command_norm = math.inf
                    try:
                        if values.get("walk_orientation_set") != "1":
                            yaw_error_deg = 0.0
                        elif values.get("walk_orientation_absolute") == "1":
                            yaw_error_deg = abs(
                                (
                                    float(values["walk_orientation"])
                                    - float(values["self_yaw"])
                                    + 180.0
                                ) % 360.0 - 180.0
                            )
                        else:
                            yaw_error_deg = abs(
                                float(values["walk_orientation"])
                            )
                    except (KeyError, ValueError):
                        yaw_error_deg = 0.0
                    if (
                        values.get("motion") in {"Neutral", "GetUpRL"}
                        or command_norm <= 0.10
                    ):
                        near_ball_low_command_samples += 1
                    if command_norm <= 0.10 and yaw_error_deg > 5.0:
                        near_ball_pure_turn_samples += 1
                    if (
                        values.get("motion") == "Neutral"
                        or (command_norm <= 0.10 and yaw_error_deg <= 5.0)
                    ):
                        near_ball_idle_samples += 1
                    try:
                        self_speed = float(values["self_speed"])
                        ball_speed = float(values.get("ball_speed", "inf"))
                    except (KeyError, ValueError):
                        pass
                    else:
                        if command_norm > 0.10 and self_speed <= 0.10:
                            near_ball_valid_command_low_speed_samples += 1
                            if 0.0 <= ball_speed <= 0.15:
                                near_ball_stationary_ball_low_speed_samples += 1
                previous_status_motion = values.get("motion")
            elif line.startswith("MY3D_KICK_SETUP"):
                values = _fields(line)
                if phase := values.get("phase"):
                    setup_phases[phase] += 1
                    mode = values.get("mode", "Unknown")
                    setup_phases_by_mode.setdefault(mode, Counter())[phase] += 1
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
    developed_illegal_defense: int | None = None
    opponent_illegal_defense: int | None = None
    if score is not None:
        if score["left_team"] == current_team:
            developed_illegal_defense = illegal_defense["left"]
            opponent_illegal_defense = illegal_defense["right"]
        elif score["right_team"] == current_team:
            developed_illegal_defense = illegal_defense["right"]
            opponent_illegal_defense = illegal_defense["left"]
    return {
        "schema_version": 6,
        "run_dir": str(run_dir.resolve()),
        "score": score,
        "server": {
            "illegal_defense_left": illegal_defense["left"],
            "illegal_defense_right": illegal_defense["right"],
        },
        "developed_team": {
            "name": current_team,
            "illegal_defense_events": developed_illegal_defense,
            "opponent_illegal_defense_events": opponent_illegal_defense,
            "status_samples": status_samples,
            **{
                name: dict(counter.most_common())
                for name, counter in status_fields.items()
            },
            "kick_setup_phase": dict(setup_phases.most_common()),
            "kick_setup_phase_by_mode": {
                mode: dict(counter.most_common())
                for mode, counter in sorted(setup_phases_by_mode.items())
            },
            "execution_events": {
                "motion": dict(execution_event_motion.most_common()),
                "status": dict(execution_event_status.most_common()),
                "kick_mode": dict(execution_event_kick_mode.most_common()),
            },
            "strategy_motion": dict(strategy_motion.most_common()),
            "visible_ball_progress": _ball_progress(
                visible_ball_x_by_observation),
            "closest_visible_ball_distance": _closest_ball_distance(
                visible_ball_dist_by_observation),
            "fresh_ball_progress": _ball_progress(
                fresh_ball_x_by_observation),
            "bounded_ball_track_progress": _ball_progress(
                bounded_ball_x_by_observation),
            "exact_kick_samples": exact_kick_samples,
            "fallback_kick_samples": fallback_kick_samples,
            "players_with_get_up_samples": sorted(player_get_up_samples),
            "get_up_samples_by_player": dict(player_get_up_samples),
            "get_up_episodes_by_player": dict(player_get_up_episodes),
            "get_up_entries_by_previous_motion": dict(
                get_up_entries_by_previous_motion.most_common()
            ),
            "tactical_churn": {
                "duty_switches": duty_switches,
                "plan_revision_switches": plan_revision_switches,
                "target_step": _step_distribution(tactical_target_steps_m),
            },
            "near_ball_response": {
                "samples": near_ball_samples,
                "formation_samples": near_ball_formation_samples,
                "pressure_samples": near_ball_pressure_samples,
                "neutral_or_low_translation_command_samples": (
                    near_ball_low_command_samples
                ),
                "pure_turn_samples": near_ball_pure_turn_samples,
                "neutral_or_idle_command_samples": near_ball_idle_samples,
                "valid_translation_but_low_self_speed_samples": (
                    near_ball_valid_command_low_speed_samples
                ),
                "stationary_ball_valid_translation_low_self_speed_samples": (
                    near_ball_stationary_ball_low_speed_samples
                ),
                "role": dict(near_ball_role.most_common()),
            },
        },
        "interpretation_limits": [
            "pristine Apollo is silent, so its internal decisions are not inferred",
            "status counts are samples, not independent physical events",
            "tactical churn is computed between adjacent periodic status samples per player and therefore omits switches between samples",
            "ball progress uses 100 ms server-time buckets when telemetry provides server_time, otherwise legacy local-cycle buckets",
            "fresh-ball progress accepts visible or at-most-0.75 s old estimates; bounded-track progress additionally accepts at-most-3.5 s near-contact tracks",
            "all ball progress is developed-team perception, not server ground truth",
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
