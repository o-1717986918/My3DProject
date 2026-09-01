#!/usr/bin/env python3
"""Export accepted teacher parameters as a provenance-carrying runtime table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--allow-evaluated-partial-grid", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [record for record in source["records"] if record["accepted"]]
    condition_count = int(source["condition_count"])
    complete_grid = len(records) == condition_count
    if not records:
        raise ValueError("runtime export requires accepted condition nodes")
    evaluation_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.evaluation
    ]
    evaluation_seeds = {int(report["seed"]) for report in evaluation_reports}
    cpu_gate_passed = bool(
        len(evaluation_reports) >= 3
        and len(evaluation_seeds) >= 3
        and all(
            bool(report.get("promotable"))
            and int(report.get("trial_count", 0)) >= 300
            and Path(report["source_manifest"]).resolve() == args.manifest.resolve()
            and not any(bool(trial["metrics"]["fell"]) for trial in report["trials"])
            for report in evaluation_reports
        )
    )
    if not complete_grid and not (
        args.allow_evaluated_partial_grid
        and len(records) / condition_count >= 0.95
        and cpu_gate_passed
    ):
        raise ValueError("runtime export requires a complete accepted condition grid")
    nodes = []
    for record in records:
        parameters = [float(value) for value in record["parameters"]]
        if len(parameters) != 14:
            raise ValueError("teacher parameter vector must contain 14 values")
        nodes.append(
            {
                "condition_index": int(record["condition_index"]),
                "distance_m": float(record["distance_m"]),
                "angle_deg": float(record["angle_deg"]),
                "requested_speed_mps": float(record["requested_speed_mps"]),
                "ball_x_offset_m": float(record["ball_x_offset_m"]),
                "ball_y_offset_m": float(record["ball_y_offset_m"]),
                "mode": str(record["mode"]),
                "parameters": parameters,
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "experimental_kick_residual_parameter_table",
        "promotable": False,
        "promotion_blocker": (
            "requires RCSSServerMJ transition and match validation"
            if cpu_gate_passed
            else "requires three-seed exact-physics validation"
        ),
        "cpu_gate_passed": cpu_gate_passed,
        "complete_grid": complete_grid,
        "accepted_condition_count": len(records),
        "declared_condition_count": condition_count,
        "missing_condition_indices": sorted(
            int(record["condition_index"])
            for record in source["records"]
            if not record["accepted"]
        ),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "evaluations": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(report["seed"]),
                "trial_count": int(report["trial_count"]),
                "successful_trials": int(report["successful_trials"]),
                "success_rate": float(report["success_rate"]),
            }
            for path, report in zip(args.evaluation, evaluation_reports, strict=True)
        ],
        "nodes": sorted(nodes, key=lambda node: node["condition_index"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "node_count": len(nodes)}))


if __name__ == "__main__":
    main()
