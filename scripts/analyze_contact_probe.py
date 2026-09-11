#!/usr/bin/env python3
"""Measure a fixed-scene ball contact from Apollo client telemetry."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


STATUS_MARKERS = ("APOLLO_REBUILD_STATUS ", "MY3D_STATUS ")


def parse_status(line: str) -> dict[str, str] | None:
    for marker in STATUS_MARKERS:
        marker_index = line.find(marker)
        if marker_index < 0:
            continue
        values: dict[str, str] = {}
        for token in line[marker_index + len(marker) :].split():
            if "=" in token:
                key, value = token.split("=", 1)
                values[key] = value
        return values
    return None


def load_statuses(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace") as stream:
        return [
            status
            for line in stream
            if (status := parse_status(line)) is not None
        ]


def number(status: dict[str, str], *keys: str) -> float:
    for key in keys:
        if key in status:
            return float(status[key])
    raise KeyError(keys[0])


def usable_ball(status: dict[str, str]) -> bool:
    if "ball_position_valid" in status and status["ball_position_valid"] != "1":
        return False
    if status.get("ball_visible") == "1":
        return True
    # The baseline rebuild intentionally keeps Apollo's smaller BallState and
    # only exposes fresh coordinates while the ball is visible.
    if "ball_position_valid" not in status:
        return False
    try:
        age_s = number(status, "ball_position_age")
    except (KeyError, ValueError):
        return False
    return age_s <= 0.75 or (
        status.get("ball_near_contact_track") == "1" and age_s <= 3.5
    )


def status_time(status: dict[str, str]) -> float:
    return number(status, "server_time", "t")


def find_release_time(
    trigger_statuses: list[dict[str, str]],
    initial: tuple[float, float],
    initial_radius_m: float,
    maximum_trigger_ball_distance_m: float,
) -> float:
    for status in trigger_statuses:
        try:
            ball = (number(status, "ball_x"), number(status, "ball_y"))
        except (KeyError, ValueError):
            continue
        try:
            ball_distance = number(status, "ball_dist")
        except (KeyError, ValueError):
            try:
                ball_distance = math.dist(
                    ball, (number(status, "x"), number(status, "y"))
                )
            except (KeyError, ValueError):
                continue
        if (
            usable_ball(status)
            and math.dist(ball, initial) <= initial_radius_m
            and ball_distance <= maximum_trigger_ball_distance_m
        ):
            return status_time(status)
    raise ValueError("trigger log never observed the release configuration")


def analyze(
    paths: list[Path],
    trigger_path: Path,
    initial: tuple[float, float],
    direction_deg: float,
    window_s: float,
    initial_radius_m: float,
    maximum_trigger_ball_distance_m: float,
    minimum_contact_progress_m: float,
) -> dict[str, object]:
    release_time_s = find_release_time(
        load_statuses(trigger_path),
        initial,
        initial_radius_m,
        maximum_trigger_ball_distance_m,
    )
    samples_by_bucket: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    fell = False
    minimum_height_m = math.inf
    getup_samples = 0
    for path in paths:
        for status in load_statuses(path):
            try:
                time_s = status_time(status)
            except (KeyError, ValueError):
                continue
            if time_s < release_time_s or time_s > release_time_s + window_s:
                continue
            if path == trigger_path:
                try:
                    height_m = number(status, "z")
                except (KeyError, ValueError):
                    pass
                else:
                    minimum_height_m = min(minimum_height_m, height_m)
                    fell = fell or height_m <= 0.35
                motion = status.get("motion", "")
                if motion.startswith("GetUp"):
                    getup_samples += 1
                    fell = True
            if not usable_ball(status):
                continue
            try:
                position = (number(status, "ball_x"), number(status, "ball_y"))
            except (KeyError, ValueError):
                continue
            try:
                speed_mps = number(status, "ball_speed")
            except (KeyError, ValueError):
                try:
                    speed_mps = math.hypot(
                        number(status, "ball_vx"), number(status, "ball_vy")
                    ) if status.get("ball_velocity_valid") == "1" else -1.0
                except (KeyError, ValueError):
                    speed_mps = -1.0
            bucket = int(round(time_s * 50.0))
            samples_by_bucket[bucket].append((*position, speed_mps))

    angle_rad = math.radians(direction_deg)
    forward = (math.cos(angle_rad), math.sin(angle_rad))
    lateral = (-forward[1], forward[0])
    trajectory: list[dict[str, float]] = []
    for bucket, bucket_samples in sorted(samples_by_bucket.items()):
        x_m = statistics.median(sample[0] for sample in bucket_samples)
        y_m = statistics.median(sample[1] for sample in bucket_samples)
        speed_samples = [sample[2] for sample in bucket_samples if sample[2] >= 0.0]
        speed_mps = statistics.median(speed_samples) if speed_samples else -1.0
        delta = (x_m - initial[0], y_m - initial[1])
        trajectory.append({
            "time_s": bucket / 50.0,
            "forward_progress_m": delta[0] * forward[0] + delta[1] * forward[1],
            "signed_lateral_m": delta[0] * lateral[0] + delta[1] * lateral[1],
            "speed_mps": speed_mps,
        })
    if not trajectory:
        raise ValueError("no usable ball trajectory samples were observed")

    peak = max(trajectory, key=lambda sample: sample["forward_progress_m"])
    contact_samples = [
        sample
        for sample in trajectory
        if sample["forward_progress_m"] >= minimum_contact_progress_m
    ]
    contact_time_s = (
        contact_samples[0]["time_s"] - release_time_s if contact_samples else None
    )
    return {
        "schema_version": 1,
        "release_time_s": release_time_s,
        "window_s": window_s,
        "observation_buckets": len(trajectory),
        "contact": bool(contact_samples),
        "time_to_contact_s": contact_time_s,
        "maximum_forward_progress_m": peak["forward_progress_m"],
        "lateral_error_at_max_progress_m": abs(peak["signed_lateral_m"]),
        "direction_error_at_max_progress_deg": (
            math.degrees(math.atan2(
                abs(peak["signed_lateral_m"]), peak["forward_progress_m"]
            ))
            if peak["forward_progress_m"] > 1.0e-9
            else None
        ),
        "peak_observed_ball_speed_mps": max(
            (sample["speed_mps"] for sample in trajectory), default=-1.0
        ),
        "fell": fell,
        "minimum_trigger_player_height_m": (
            minimum_height_m if math.isfinite(minimum_height_m) else None
        ),
        "trigger_player_getup_samples": getup_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--trigger-log", required=True, type=Path)
    parser.add_argument("--initial-x", required=True, type=float)
    parser.add_argument("--initial-y", required=True, type=float)
    parser.add_argument("--direction-deg", type=float, default=0.0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--initial-radius", type=float, default=0.15)
    parser.add_argument("--maximum-trigger-ball-distance", type=float, default=0.55)
    parser.add_argument("--minimum-contact-progress", type=float, default=0.10)
    args = parser.parse_args()
    result = analyze(
        args.logs,
        args.trigger_log,
        (args.initial_x, args.initial_y),
        args.direction_deg,
        args.window_seconds,
        args.initial_radius,
        args.maximum_trigger_ball_distance,
        args.minimum_contact_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
