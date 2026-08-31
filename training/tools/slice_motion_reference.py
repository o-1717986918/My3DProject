#!/usr/bin/env python3
"""Extract and revalidate a phase window from a validated motion reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.holosoma_motion import replay_rcss_surface
from my3d_rl.motion_reference import sha256, validate_motion_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="inclusive frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).parents[1] / "contracts" / "run_policy_v2.yaml",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must live outside the repository")

    with np.load(args.input, allow_pickle=False) as archive:
        frame_count = int(archive["joint_position"].shape[0])
        if args.start < 0 or args.end < args.start or args.end >= frame_count:
            raise ValueError(
                f"invalid inclusive frame range {args.start}:{args.end} "
                f"for {frame_count} frames"
            )
        selected = {
            name: np.asarray(archive[name][args.start : args.end + 1]).copy()
            for name in (
                "root_position",
                "root_quaternion_xyzw",
                "root_linear_velocity",
                "root_angular_velocity",
                "joint_position",
                "joint_velocity",
            )
        }
        metadata = json.loads(str(archive["metadata_json"].item()))

    selected["root_position"][:, :2] -= selected["root_position"][0, :2]
    qpos = np.concatenate(
        [
            selected["root_position"],
            selected["root_quaternion_xyzw"][:, [3, 0, 1, 2]],
            selected["joint_position"],
        ],
        axis=1,
    )
    _, contacts, replay = replay_rcss_surface(qpos, load_policy_contract(args.contract))
    selected["foot_contact"] = contacts

    duration = (len(qpos) - 1) / 50.0
    horizontal_displacement = np.linalg.norm(
        selected["root_position"][-1, :2] - selected["root_position"][0, :2]
    )
    metadata.update(
        {
            "parent_reference": str(args.input.resolve()),
            "parent_reference_sha256": sha256(args.input),
            "slice_frame_range_inclusive": [args.start, args.end],
            "conversion_command": (
                "training/tools/slice_motion_reference.py "
                f"{args.input} --start {args.start} --end {args.end} "
                f"--output {args.output}"
            ),
            "average_horizontal_speed_m_s": float(horizontal_displacement / duration),
            "mean_root_yaw_rate_rad_s": float(
                np.mean(selected["root_angular_velocity"][:, 2])
            ),
            "rcss_replay": replay,
            "rcss_contact_count": contacts.sum(axis=0).astype(int).tolist(),
        }
    )
    selected["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **selected)
    result = validate_motion_reference(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise ValueError("sliced reference failed validation")


if __name__ == "__main__":
    main()
