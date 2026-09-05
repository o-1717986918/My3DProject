#!/usr/bin/env python3
"""Run the frozen CPU locomotion evaluator over football command primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).parents[2]
EVALUATOR = REPOSITORY_ROOT / "training" / "tools" / "evaluate_onnx_run.py"


def soccer_command_suite() -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Return the fixed command surface required by the player foundation."""
    return (
        ("stand", (0.0, 0.0, 0.0)),
        ("precision_forward", (0.4, 0.0, 0.0)),
        ("fast_forward", (1.5, 0.0, 0.0)),
        ("reverse", (-0.2, 0.0, 0.0)),
        ("pure_left_strafe", (0.0, 0.30, 0.0)),
        ("pure_right_strafe", (0.0, -0.30, 0.0)),
        ("pure_left_turn", (0.0, 0.0, 0.75)),
        ("pure_right_turn", (0.0, 0.0, -0.75)),
        ("curve_left", (0.60, 0.0, 0.50)),
        ("curve_right", (0.60, 0.0, -0.50)),
    )


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("command suite requires at least one report")
    names = [str(report["suite_command_name"]) for report in reports]
    if len(names) != len(set(names)):
        raise ValueError("command suite repeats a command name")
    for report in reports:
        if not isinstance(report.get("soccer_command_gate_passed"), bool):
            raise ValueError("command report lacks the soccer command gate")
    passed = sum(bool(report["soccer_command_gate_passed"]) for report in reports)
    return {
        "command_count": len(reports),
        "passed_command_count": passed,
        "all_commands_passed": passed == len(reports),
        "minimum_upright_completion_rate": min(
            float(report["upright_completion_rate"]) for report in reports
        ),
        "maximum_median_planar_velocity_rmse_m_s": max(
            float(report["planar_velocity_tracking_rmse"]["median_m_s"])
            for report in reports
        ),
        "maximum_median_yaw_rate_rmse_rad_s": max(
            float(report["yaw_rate_tracking_rmse"]["median_rad_s"])
            for report in reports
        ),
        "failed_commands": [
            str(report["suite_command_name"])
            for report in reports
            if not bool(report["soccer_command_gate_passed"])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260941)
    parser.add_argument("--symmetry-ensemble", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not args.model.is_file() or not args.contract.is_file():
        raise FileNotFoundError("model and contract must exist")
    if not args.output_dir.is_absolute():
        raise ValueError("output-dir must be absolute")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    reports: list[dict[str, Any]] = []
    for index, (name, command) in enumerate(soccer_command_suite()):
        report_path = args.output_dir / f"{index:02d}-{name}.json"
        invocation = [
            sys.executable,
            str(EVALUATOR),
            "--model",
            str(args.model.resolve()),
            "--contract",
            str(args.contract.resolve()),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed + index),
            "--vx",
            str(command[0]),
            "--vy",
            str(command[1]),
            "--yaw-rate",
            str(command[2]),
            "--output",
            str(report_path),
        ]
        if args.symmetry_ensemble:
            invocation.append("--symmetry-ensemble")
        subprocess.run(invocation, check=True, stdout=subprocess.DEVNULL)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["suite_command_name"] = name
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(
            f"{name}: pass={report['soccer_command_gate_passed']} "
            f"completion={report['upright_completion_rate']:.3f} "
            "planar_rmse="
            f"{report['planar_velocity_tracking_rmse']['median_m_s']:.3f} "
            "yaw_rmse="
            f"{report['yaw_rate_tracking_rmse']['median_rad_s']:.3f}",
            flush=True,
        )

    summary = {
        "schema_version": 2,
        "purpose": "football_locomotion_command_suite_cpu_acceptance",
        "model": str(args.model.resolve()),
        "contract": str(args.contract.resolve()),
        "episodes_per_command": args.episodes,
        "seed": args.seed,
        "symmetry_ensemble": args.symmetry_ensemble,
        **summarize_reports(reports),
        "reports": [
            str((args.output_dir / f"{index:02d}-{name}.json").resolve())
            for index, (name, unused) in enumerate(soccer_command_suite())
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
