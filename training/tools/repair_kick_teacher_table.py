#!/usr/bin/env python3
"""Robustly re-optimize selected cells of an exact-physics kick table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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


def _cell_bounds(value: float, values: list[float]) -> tuple[float, float]:
    index = values.index(value)
    lower = values[index] if index == 0 else 0.5 * (values[index - 1] + value)
    upper = (
        values[index] if index + 1 == len(values) else 0.5 * (value + values[index + 1])
    )
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--condition-index", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=6501)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--robust-samples", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.restarts < 1:
        raise ValueError("--restarts must be positive")

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    records = source["records"]
    requested = set(args.condition_index)
    known = {int(record["condition_index"]) for record in records}
    if not requested <= known:
        raise ValueError(f"unknown condition indices: {sorted(requested - known)}")
    x_values = sorted({float(record["ball_x_offset_m"]) for record in records})
    y_values = sorted({float(record["ball_y_offset_m"]) for record in records})
    repaired: list[int] = []

    for record in records:
        condition_index = int(record["condition_index"])
        if condition_index not in requested:
            continue
        ball_x = float(record["ball_x_offset_m"])
        ball_y = float(record["ball_y_offset_m"])
        spec = KickTeacherSpec(
            target_distance_m=float(record["distance_m"]),
            target_angle_deg=float(record["angle_deg"]),
            requested_ball_speed_mps=float(record["requested_speed_mps"]),
            desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
            action_mode=str(record["mode"]),
        )
        evaluator = KickTeacherEvaluator(spec)
        candidates = [
            evaluator.optimize(
                seed=args.seed + condition_index * args.restarts + restart,
                population=args.population,
                generations=args.generations,
                robust_samples=args.robust_samples,
                ball_x_offset_m=ball_x,
                ball_y_offset_m=ball_y,
                initial_parameters=np.asarray(record["parameters"], dtype=np.float64),
                ball_x_range_m=_cell_bounds(ball_x, x_values),
                ball_y_range_m=_cell_bounds(ball_y, y_values),
            )
            for restart in range(args.restarts)
        ]
        best = max(candidates, key=lambda candidate: candidate.score)
        metrics = evaluator.rollout(
            best.parameters,
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
        )
        record.update(
            {
                "accepted": kick_trial_success(metrics),
                "parameters": best.parameters.tolist(),
                "score": best.score,
                "nominal_score": float(metrics["score"]),
                "metrics": metrics,
                "history": list(best.history),
                "restart_scores": [candidate.score for candidate in candidates],
                "repair": {
                    "seed": args.seed,
                    "population": args.population,
                    "generations": args.generations,
                    "robust_samples": args.robust_samples,
                    "restarts": args.restarts,
                    "ball_x_cell_m": list(_cell_bounds(ball_x, x_values)),
                    "ball_y_cell_m": list(_cell_bounds(ball_y, y_values)),
                },
            }
        )
        repaired.append(condition_index)

    accepted_count = sum(bool(record["accepted"]) for record in records)
    output = {
        **source,
        "purpose": "robust_repaired_kick_teacher_table",
        "promotable": False,
        "promotion_blocker": "requires independent 300-trial and server gates",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "accepted_condition_count": accepted_count,
        "complete": accepted_count == len(records),
        "repair_seed": args.seed,
        "repaired_condition_indices": repaired,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
