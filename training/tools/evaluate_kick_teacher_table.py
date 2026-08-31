#!/usr/bin/env python3
"""Evaluate nearest-condition kick teachers on held-out ball placements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_teacher import (
    KickTeacherEvaluator,
    KickTeacherSpec,
    kick_trial_success,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--target-angle", type=float, default=0.0)
    parser.add_argument("--requested-speed", type=float, default=1.43)
    parser.add_argument("--arrival-speed", type=float, default=0.8)
    parser.add_argument("--mode", choices=("pass", "shot", "clear"), default="pass")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--ball-x-min", type=float, default=-0.01)
    parser.add_argument("--ball-x-max", type=float, default=0.08)
    parser.add_argument("--ball-y-min", type=float, default=-0.08)
    parser.add_argument("--ball-y-max", type=float, default=0.08)
    parser.add_argument("--phase-alignment-s-per-m", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in source["records"]
        if record["accepted"] and record["mode"] == args.mode
    ]
    if not records:
        raise ValueError("manifest contains no accepted records for the requested mode")

    spec = KickTeacherSpec(
        target_distance_m=args.target_distance,
        target_angle_deg=args.target_angle,
        requested_ball_speed_mps=args.requested_speed,
        desired_arrival_speed_mps=args.arrival_speed,
        action_mode=args.mode,
    )
    evaluator = KickTeacherEvaluator(spec)
    rng = np.random.default_rng(args.seed)
    trials: list[dict[str, object]] = []
    for trial_index in range(args.trials):
        ball_x = float(rng.uniform(args.ball_x_min, args.ball_x_max))
        ball_y = float(rng.uniform(args.ball_y_min, args.ball_y_max))

        def distance(record: dict[str, object]) -> float:
            return float(
                4.0 * ((float(record["distance_m"]) - args.target_distance) / 3.0) ** 2
                + ((float(record["angle_deg"]) - args.target_angle) / 15.0) ** 2
                + ((float(record["ball_x_offset_m"]) - ball_x) / 0.09) ** 2
                + ((float(record["ball_y_offset_m"]) - ball_y) / 0.08) ** 2
                + ((float(record["requested_speed_mps"]) - args.requested_speed) / 2.2)
                ** 2
            )

        selected = min(records, key=distance)
        metrics = evaluator.rollout(
            np.asarray(selected["parameters"], dtype=np.float64),
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
            phase_reference_ball_x_offset_m=float(selected["ball_x_offset_m"]),
            phase_alignment_s_per_m=args.phase_alignment_s_per_m,
        )
        trials.append(
            {
                "trial": trial_index,
                "ball_x_offset_m": ball_x,
                "ball_y_offset_m": ball_y,
                "selected_condition_index": selected["condition_index"],
                "success": kick_trial_success(metrics),
                "metrics": metrics,
            }
        )

    successful = sum(bool(trial["success"]) for trial in trials)
    required = int(np.ceil(0.9 * args.trials))
    report = {
        "purpose": "kick_teacher_table_held_out_evaluation",
        "source_manifest": str(args.manifest),
        "seed": args.seed,
        "trial_count": args.trials,
        "successful_trials": successful,
        "success_rate": successful / args.trials,
        "promotable": successful >= required,
        "phase_alignment_s_per_m": args.phase_alignment_s_per_m,
        "gate": {"required_successes": required, "passed": successful >= required},
        "trials": trials,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
