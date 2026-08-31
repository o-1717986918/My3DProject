#!/usr/bin/env python3
"""Project a local T1 running clip onto a periodic symmetric contact cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.motion_reference import sha256, validate_motion_reference
from my3d_rl.periodic_reference import build_periodic_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end-inclusive", type=int)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).parents[1] / "contracts" / "run_policy_v2.yaml",
    )
    parser.add_argument("--source-half-weight", type=float, default=0.8)
    parser.add_argument("--smoothing-passes", type=int, default=4)
    parser.add_argument("--stance-correction-iterations", type=int, default=1)
    parser.add_argument("--stance-smoothing-passes", type=int, default=1)
    parser.add_argument("--root-yaw-scale", type=float, default=1.0)
    parser.add_argument("--root-xy-smoothing-passes", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    repository_root = Path(__file__).parents[2].resolve()
    if args.output.resolve().is_relative_to(repository_root):
        raise ValueError("output must live outside the repository")

    with np.load(args.input, allow_pickle=False) as archive:
        source = {
            name: np.asarray(archive[name]).copy()
            for name in (
                "root_position",
                "root_quaternion_xyzw",
                "root_linear_velocity",
                "root_angular_velocity",
                "joint_position",
                "joint_velocity",
                "foot_contact",
            )
        }
        metadata = json.loads(str(archive["metadata_json"].item()))

    end = (
        source["joint_position"].shape[0] - 1
        if args.frame_end_inclusive is None
        else args.frame_end_inclusive
    )
    if (
        args.frame_start < 0
        or end < args.frame_start
        or end >= source["joint_position"].shape[0]
    ):
        raise ValueError(
            f"invalid inclusive source frame range {args.frame_start}:{end}"
        )
    source = {
        name: values[args.frame_start : end + 1] for name, values in source.items()
    }

    arrays, projection = build_periodic_reference(
        source,
        load_policy_contract(args.contract),
        source_half_weight=args.source_half_weight,
        smoothing_passes=args.smoothing_passes,
        stance_correction_iterations=args.stance_correction_iterations,
        stance_smoothing_passes=args.stance_smoothing_passes,
        root_yaw_scale=args.root_yaw_scale,
        root_xy_smoothing_passes=args.root_xy_smoothing_passes,
    )
    command = " ".join(shlex.quote(value) for value in sys.argv)
    metadata.update(
        {
            "parent_reference": str(args.input.resolve()),
            "parent_reference_sha256": sha256(args.input),
            "parent_frame_range_inclusive": [args.frame_start, end],
            "conversion_command": command,
            "periodic_projection": projection,
            "rcss_replay": projection["rcss_replay"],
            "average_horizontal_speed_m_s": projection[
                "commanded_average_forward_speed_m_s"
            ],
        }
    )
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    schema_validation = validate_motion_reference(args.output)
    result = {
        "schema_version": 1,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "projection": projection,
        "motion_reference_validation": schema_validation,
        "passed": projection["passed"] and schema_validation["passed"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
