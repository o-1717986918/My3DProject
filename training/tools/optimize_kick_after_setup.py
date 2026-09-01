#!/usr/bin/env python3
"""Re-optimize a kick residual from measured post-alignment robot states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, PARAMETER_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--condition-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--robust-samples", type=int, default=5)
    parser.add_argument("--ball-x-min", type=float, default=-0.01)
    parser.add_argument("--ball-x-max", type=float, default=0.08)
    parser.add_argument("--ball-y-min", type=float, default=-0.08)
    parser.add_argument("--ball-y-max", type=float, default=0.08)
    parser.add_argument("--setup-timeout", type=float, default=3.0)
    parser.add_argument("--setup-tolerance", type=float, default=0.02)
    parser.add_argument("--setup-confirmation-cycles", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    matches = [
        record
        for record in source["records"]
        if int(record["condition_index"]) == args.condition_index
        and bool(record["accepted"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "condition index must select exactly one accepted source record"
        )
    source_record = matches[0]
    spec = KickTeacherSpec(
        target_distance_m=float(source_record["distance_m"]),
        target_angle_deg=float(source_record["angle_deg"]),
        requested_ball_speed_mps=float(source_record["requested_speed_mps"]),
        desired_arrival_speed_mps=float(source_record["desired_arrival_speed_mps"]),
        action_mode=str(source_record["mode"]),
        evaluation_duration_s=args.setup_timeout + 3.0,
    )
    evaluator = KickTeacherEvaluator(spec)
    setup_x = float(source_record["ball_x_offset_m"])
    setup_y = float(source_record["ball_y_offset_m"])
    result = evaluator.optimize(
        seed=args.seed,
        population=args.population,
        generations=args.generations,
        robust_samples=args.robust_samples,
        ball_x_offset_m=setup_x,
        ball_y_offset_m=setup_y,
        initial_parameters=np.asarray(source_record["parameters"], dtype=np.float64),
        ball_x_range_m=(args.ball_x_min, args.ball_x_max),
        ball_y_range_m=(args.ball_y_min, args.ball_y_max),
        setup_ball_x_offset_m=setup_x,
        setup_ball_y_offset_m=setup_y,
        setup_timeout_s=args.setup_timeout,
        setup_tolerance_m=args.setup_tolerance,
        setup_confirmation_cycles=args.setup_confirmation_cycles,
    )
    metrics = evaluator.rollout(
        result.parameters,
        ball_x_offset_m=setup_x,
        ball_y_offset_m=setup_y,
        setup_ball_x_offset_m=setup_x,
        setup_ball_y_offset_m=setup_y,
        setup_timeout_s=args.setup_timeout,
        setup_tolerance_m=args.setup_tolerance,
        setup_confirmation_cycles=args.setup_confirmation_cycles,
    )
    accepted = bool(metrics["contact"] and not metrics["fell"])
    record = {
        "condition_index": args.condition_index,
        "mode": spec.action_mode,
        "distance_m": spec.target_distance_m,
        "angle_deg": spec.target_angle_deg,
        "requested_speed_mps": spec.requested_ball_speed_mps,
        "desired_arrival_speed_mps": spec.desired_arrival_speed_mps,
        "ball_x_offset_m": setup_x,
        "ball_y_offset_m": setup_y,
        "parameters": result.parameters.tolist(),
        "score": result.score,
        "metrics": metrics,
        "accepted": accepted,
    }
    manifest = {
        "purpose": "post_alignment_kick_residual_optimization",
        "promotable": False,
        "promotion_blocker": "requires independent 300-trial and server gates",
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "seed": args.seed,
        "population": args.population,
        "generations": args.generations,
        "robust_samples": args.robust_samples,
        "initial_ball_envelope_m": {
            "x": [args.ball_x_min, args.ball_x_max],
            "y": [args.ball_y_min, args.ball_y_max],
        },
        "setup": {
            "condition_index": args.condition_index,
            "ball_x_offset_m": setup_x,
            "ball_y_offset_m": setup_y,
            "timeout_s": args.setup_timeout,
            "tolerance_m": args.setup_tolerance,
            "confirmation_cycles": args.setup_confirmation_cycles,
        },
        "parameter_names": list(PARAMETER_NAMES),
        "history": list(result.history),
        "records": [record],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
