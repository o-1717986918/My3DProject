#!/usr/bin/env python3
"""Evaluate table plus learned correction in exact CPU MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.kick_teacher import (
    KickTeacherEvaluator,
    KickTeacherSpec,
    kick_trial_success,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("correction_onnx", type=Path)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--ball-x-min", type=float, default=-0.01)
    parser.add_argument("--ball-x-max", type=float, default=0.08)
    parser.add_argument("--ball-y-min", type=float, default=-0.08)
    parser.add_argument("--ball-y-max", type=float, default=0.08)
    parser.add_argument("--correction-scale", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    source = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in source["records"]
        if bool(record["accepted"])
        and record["mode"] == "pass"
        and abs(float(record["distance_m"]) - 2.0) < 1.0e-9
        and abs(float(record["angle_deg"])) < 1.0e-9
    ]
    if not records:
        raise ValueError("teacher manifest has no accepted 2 m forward-pass records")
    session = ort.InferenceSession(
        str(args.correction_onnx.resolve()), providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].shape not in ([None, 96], [1, 96]):
        raise ValueError("correction ONNX input must have shape [N, 96]")
    if session.get_outputs()[0].shape not in ([None, 23], [1, 23]):
        raise ValueError("correction ONNX output must have shape [N, 23]")

    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(
            target_distance_m=2.0,
            target_angle_deg=0.0,
            requested_ball_speed_mps=1.43,
            desired_arrival_speed_mps=0.8,
            action_mode="pass",
        )
    )
    rng = np.random.default_rng(args.seed)
    trials: list[dict[str, object]] = []
    for trial_index in range(args.trials):
        ball_x = float(rng.uniform(args.ball_x_min, args.ball_x_max))
        ball_y = float(rng.uniform(args.ball_y_min, args.ball_y_max))

        def distance(record: dict[str, object]) -> float:
            return float(
                ((float(record["ball_x_offset_m"]) - ball_x) / 0.045) ** 2
                + ((float(record["ball_y_offset_m"]) - ball_y) / 0.04) ** 2
            )

        selected = min(records, key=distance)
        metrics = evaluator.rollout(
            np.asarray(selected["parameters"], dtype=np.float64),
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
            kick_correction_session=session,
            kick_correction_scale=args.correction_scale,
        )
        trials.append(
            {
                "trial": trial_index,
                "ball_x_offset_m": ball_x,
                "ball_y_offset_m": ball_y,
                "selected_condition_index": int(selected["condition_index"]),
                "success": kick_trial_success(metrics),
                "metrics": metrics,
            }
        )

    successful = sum(bool(trial["success"]) for trial in trials)
    required = int(np.ceil(0.9 * args.trials))
    report = {
        "purpose": "exact_cpu_kick_teacher_table_plus_correction_evaluation",
        "promotable": successful >= required,
        "promotion_blocker": (
            None
            if successful >= required
            else "exact CPU held-out success rate is below 90%"
        ),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "correction_onnx": str(args.correction_onnx.resolve()),
        "correction_onnx_sha256": _sha256(args.correction_onnx),
        "correction_scale": args.correction_scale,
        "seed": args.seed,
        "trial_count": args.trials,
        "successful_trials": successful,
        "success_rate": successful / args.trials,
        "contact_trials": sum(bool(trial["metrics"]["contact"]) for trial in trials),
        "fall_trials": sum(bool(trial["metrics"]["fell"]) for trial in trials),
        "gate": {"required_successes": required, "passed": successful >= required},
        "trials": trials,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
