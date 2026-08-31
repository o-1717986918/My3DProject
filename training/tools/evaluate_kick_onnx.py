#!/usr/bin/env python3
"""Closed-loop exact-MuJoCo evaluation for a kick_policy_v2 ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.kick_teacher import (
    KickTeacherEvaluator,
    KickTeacherSpec,
    kick_trial_success,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--target-angle", type=float, default=0.0)
    parser.add_argument("--requested-speed", type=float, default=1.43)
    parser.add_argument("--arrival-speed", type=float, default=0.8)
    parser.add_argument("--mode", choices=("pass", "shot", "clear"), default="pass")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--ball-x-min", type=float, default=-0.01)
    parser.add_argument("--ball-x-max", type=float, default=0.08)
    parser.add_argument("--ball-y-min", type=float, default=-0.08)
    parser.add_argument("--ball-y-max", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    spec = KickTeacherSpec(
        target_distance_m=args.target_distance,
        target_angle_deg=args.target_angle,
        requested_ball_speed_mps=args.requested_speed,
        desired_arrival_speed_mps=args.arrival_speed,
        action_mode=args.mode,
    )
    evaluator = KickTeacherEvaluator(spec)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(args.seed)
    trials = []
    for trial_index in range(args.trials):
        ball_x = float(rng.uniform(args.ball_x_min, args.ball_x_max))
        ball_y = float(rng.uniform(args.ball_y_min, args.ball_y_max))
        metrics = evaluator.rollout_policy(
            session,
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
        )
        trials.append(
            {
                "trial": trial_index,
                "ball_x_offset_m": ball_x,
                "ball_y_offset_m": ball_y,
                "success": kick_trial_success(metrics),
                "metrics": metrics,
            }
        )

    successful = sum(bool(trial["success"]) for trial in trials)
    required = int(np.ceil(0.9 * args.trials))
    report = {
        "purpose": "kick_policy_v2_closed_loop_onnx_evaluation",
        "promotable": False,
        "model": str(args.model),
        "seed": args.seed,
        "trial_count": args.trials,
        "successful_trials": successful,
        "success_rate": successful / args.trials,
        "gate": {
            "required_successes": required,
            "passed": successful >= required,
            "range_tolerance_m": 0.5,
            "corridor_half_width_m": 0.5,
            "launch_speed_tolerance_mps": 1.0,
        },
        "spec": {
            "target_distance_m": spec.target_distance_m,
            "target_angle_deg": spec.target_angle_deg,
            "requested_ball_speed_mps": spec.requested_ball_speed_mps,
            "desired_arrival_speed_mps": spec.desired_arrival_speed_mps,
            "action_mode": spec.action_mode,
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
