#!/usr/bin/env python3
"""Evaluate a serialized kick teacher under held-out ball placement noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_teacher import (
    KickTeacherEvaluator,
    KickTeacherSpec,
    clearance_trial_success,
    kick_trial_success,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--ball-x-min", type=float, default=-0.01)
    parser.add_argument("--ball-x-max", type=float, default=0.08)
    parser.add_argument("--ball-y-min", type=float, default=-0.08)
    parser.add_argument("--ball-y-max", type=float, default=0.08)
    parser.add_argument(
        "--success-profile",
        choices=("auto", "targeted", "clearance"),
        default="auto",
    )
    parser.add_argument("--clear-minimum-progress", type=float, default=4.5)
    parser.add_argument("--clear-corridor-half-width", type=float, default=1.5)
    parser.add_argument("--clear-minimum-launch-speed", type=float, default=2.5)
    parser.add_argument("--clear-minimum-torso-height", type=float, default=0.55)
    parser.add_argument("--clear-minimum-upright", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.ball_x_min > args.ball_x_max or args.ball_y_min > args.ball_y_max:
        raise ValueError("ball offset minimum must not exceed maximum")
    if min(
        args.clear_minimum_progress,
        args.clear_corridor_half_width,
        args.clear_minimum_launch_speed,
        args.clear_minimum_torso_height,
        args.clear_minimum_upright,
    ) < 0.0:
        raise ValueError("clearance thresholds must be non-negative")

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw_spec = source["spec"]
    spec = KickTeacherSpec(
        target_distance_m=float(raw_spec["target_distance_m"]),
        target_angle_deg=float(raw_spec["target_angle_deg"]),
        requested_ball_speed_mps=float(raw_spec["requested_ball_speed_mps"]),
        desired_arrival_speed_mps=float(raw_spec.get("desired_arrival_speed_mps", 1.0)),
        action_mode=str(raw_spec.get("action_mode", "pass")),
        duration_s=float(raw_spec["duration_s"]),
        evaluation_duration_s=float(raw_spec.get("evaluation_duration_s", 3.0)),
        control_dt_s=float(raw_spec["control_dt_s"]),
        simulation_dt_s=float(raw_spec["simulation_dt_s"]),
    )
    parameters = np.asarray(source["parameters"], dtype=np.float64)
    evaluator = KickTeacherEvaluator(
        spec,
        motion_base=str(source.get("motion_base", "walk")),
        stand_base_pose=str(source.get("stand_base_pose", "bent")),
        stand_support_crouch_rad=float(
            source.get("stand_support_crouch_rad", 0.0)
        ),
    )
    rng = np.random.default_rng(args.seed)
    success_profile = args.success_profile
    if success_profile == "auto":
        success_profile = "clearance" if spec.action_mode == "clear" else "targeted"

    if success_profile == "clearance":
        def classify(metrics: dict[str, float | bool]) -> bool:
            return clearance_trial_success(
                metrics,
                minimum_progress_m=args.clear_minimum_progress,
                corridor_half_width_m=args.clear_corridor_half_width,
                minimum_launch_speed_mps=args.clear_minimum_launch_speed,
                minimum_torso_height_m=args.clear_minimum_torso_height,
                minimum_upright=args.clear_minimum_upright,
            )

        gate_thresholds = {
            "minimum_progress_m": args.clear_minimum_progress,
            "corridor_half_width_m": args.clear_corridor_half_width,
            "minimum_launch_speed_mps": args.clear_minimum_launch_speed,
            "minimum_torso_height_m": args.clear_minimum_torso_height,
            "minimum_upright": args.clear_minimum_upright,
        }
    else:
        classify = kick_trial_success
        gate_thresholds = {
            "range_tolerance_m": 0.5,
            "corridor_half_width_m": 0.5,
            "launch_speed_tolerance_mps": 1.0,
        }

    trials: list[dict[str, object]] = []
    for trial_index in range(args.trials):
        ball_x = float(rng.uniform(args.ball_x_min, args.ball_x_max))
        ball_y = float(rng.uniform(args.ball_y_min, args.ball_y_max))
        metrics = evaluator.rollout(
            parameters,
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
        )
        trials.append(
            {
                "trial": trial_index,
                "ball_x_offset_m": ball_x,
                "ball_y_offset_m": ball_y,
                "success": classify(metrics),
                "metrics": metrics,
            }
        )

    successful = sum(bool(trial["success"]) for trial in trials)
    report = {
        "purpose": "r1_kick_teacher_held_out_evaluation",
        "source_manifest": str(args.manifest),
        "motion_base": evaluator.motion_base,
        "stand_base_pose": evaluator.stand_base_pose,
        "seed": args.seed,
        "trial_count": args.trials,
        "successful_trials": successful,
        "success_rate": successful / args.trials,
        "gate": {
            "required_successes": int(np.ceil(0.9 * args.trials)),
            "passed": successful >= int(np.ceil(0.9 * args.trials)),
            "success_profile": success_profile,
            **gate_thresholds,
        },
        "ball_offset_ranges_m": {
            "x": [args.ball_x_min, args.ball_x_max],
            "y": [args.ball_y_min, args.ball_y_max],
        },
        "trials": trials,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
