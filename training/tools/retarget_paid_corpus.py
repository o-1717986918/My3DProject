#!/usr/bin/env python3
"""Retarget and summarize every motion in a pinned local PAiD corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def _outside_repository(path: Path, repository: Path) -> None:
    try:
        path.relative_to(repository)
    except ValueError:
        return
    raise ValueError("derived PAiD output must stay outside the project repository")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paid_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--method", choices=("semantic", "gmr-body-ik"), required=True
    )
    parser.add_argument("--gmr-root", type=Path)
    parser.add_argument("--gmr-revision")
    parser.add_argument("--robot-scale", type=float, default=0.9)
    parser.add_argument("--expected-count", type=int, default=13)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    paid_root = args.paid_root.resolve()
    output_root = args.output_root.resolve()
    summary_path = args.summary.resolve()
    _outside_repository(output_root, repository)
    _outside_repository(summary_path, repository)
    motion_paths = sorted((paid_root / "motions").rglob("*.npz"))
    if len(motion_paths) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} PAiD motions, found {len(motion_paths)}"
        )
    tool = Path(__file__).with_name("retarget_paid_motion.py")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for motion_path in motion_paths:
        relative = motion_path.relative_to(paid_root / "motions")
        target = output_root / relative
        target = target.with_suffix(".t1.npz")
        report = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(tool),
            str(motion_path),
            str(target),
            "--paid-root",
            str(paid_root),
            "--method",
            args.method,
            "--robot-scale",
            str(args.robot_scale),
            "--report",
            str(report),
        ]
        if args.method == "gmr-body-ik":
            if args.gmr_root is None or args.gmr_revision is None:
                raise SystemExit(
                    "gmr-body-ik requires --gmr-root and --gmr-revision"
                )
            command.extend(
                [
                    "--gmr-root",
                    str(args.gmr_root.resolve()),
                    "--gmr-revision",
                    args.gmr_revision,
                ]
            )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository / "training")
        process = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
        )
        if report.is_file():
            result = json.loads(report.read_text(encoding="utf-8"))
            provenance = result.get("provenance", {})
            geometry = provenance.get("kick_geometry", {})
            replay = provenance.get("rcss_replay", {})
            clipping = provenance.get("competition_joint_limit_clipping", {})
            kick_speed = geometry.get("peak_kick_foot_relative_speed_m_s")
            other_speed = geometry.get("peak_other_foot_relative_speed_m_s")
            speed_ratio = (
                kick_speed / other_speed
                if kick_speed is not None
                and other_speed is not None
                and other_speed > 0.0
                else None
            )
            records.append(
                {
                    "relative_path": str(relative),
                    "return_code": process.returncode,
                    "passed": bool(result.get("passed", False)),
                    "errors": result.get("errors", []),
                    "source_sha256": provenance.get("source_sha256"),
                    "output_sha256": result.get("sha256"),
                    "frame_count": result.get("frame_count"),
                    "kick_leg": result.get("kick_leg"),
                    "maximum_joint_velocity_rad_s": result.get(
                        "maximum_joint_velocity_rad_s"
                    ),
                    "peak_kick_foot_relative_speed_m_s": kick_speed,
                    "peak_other_foot_relative_speed_m_s": other_speed,
                    "kick_to_other_foot_peak_speed_ratio": speed_ratio,
                    "maximum_root_tilt_rad": geometry.get(
                        "maximum_root_tilt_rad"
                    ),
                    "support_contact_near_peak": geometry.get(
                        "support_contact_near_peak"
                    ),
                    "non_foot_pitch_contact_frames": replay.get(
                        "non_foot_pitch_contact_frames"
                    ),
                    "ground_offset_max_step_m": replay.get(
                        "ground_offset_max_step_m"
                    ),
                    "clipped_value_count": clipping.get("clipped_value_count"),
                    "maximum_abs_joint_limit_correction_rad": clipping.get(
                        "maximum_abs_correction_rad"
                    ),
                }
            )
        else:
            records.append(
                {
                    "relative_path": str(relative),
                    "return_code": process.returncode,
                    "passed": False,
                    "errors": [
                        "retarget process did not produce a report",
                        process.stderr[-2000:],
                    ],
                }
            )

    passed = sum(bool(record["passed"]) for record in records)
    payload = {
        "schema_version": 1,
        "purpose": "paid_to_t1_k0_corpus_gate",
        "status": "complete",
        "method": args.method,
        "robot_scale": args.robot_scale,
        "paid_root": str(paid_root),
        "output_root": str(output_root),
        "motion_count": len(records),
        "passed_count": passed,
        "pass_rate": passed / len(records),
        "all_passed": passed == len(records),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
