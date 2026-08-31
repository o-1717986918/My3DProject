#!/usr/bin/env python3
"""Retarget a LAFAN1 BVH through pinned GMR and build a local T1 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.gmr_motion import (
    clip_contract_joint_limits,
    contact_only_human_joints,
    map_named_qpos_to_contract,
)
from my3d_rl.holosoma_motion import build_motion_reference
from my3d_rl.motion_reference import validate_motion_reference
from my3d_rl.rcss_scene import build_single_t1_soccer_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    result: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result[name] = int(model.jnt_qposadr[joint_id])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="LAFAN1 BVH input")
    parser.add_argument("output", type=Path, help="local-only motion_reference_v1 NPZ")
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=Path("training/contracts/run_policy_v3.yaml")
    )
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--gmr-revision", required=True)
    parser.add_argument("--retarget-start", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end-inclusive", type=int, required=True)
    parser.add_argument("--input-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--source-contact-height", type=float, default=0.04)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.output_fps != 50.0:
        raise SystemExit("motion_reference_v1 requires exactly 50 Hz")
    if not (0 <= args.retarget_start <= args.frame_start <= args.frame_end_inclusive):
        raise SystemExit("require retarget-start <= frame-start <= frame-end-inclusive")
    if not args.gmr_root.is_dir():
        raise SystemExit(f"GMR root does not exist: {args.gmr_root}")
    actual_gmr_revision = subprocess.run(
        ["git", "-C", str(args.gmr_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_gmr_revision != args.gmr_revision:
        raise SystemExit(
            f"GMR revision {actual_gmr_revision} != requested {args.gmr_revision}"
        )
    sys.path.insert(0, str(args.gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting  # noqa: PLC0415
    from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: PLC0415

    frames, actual_human_height = load_bvh_file(str(args.input))
    if args.frame_end_inclusive >= len(frames):
        raise SystemExit(
            f"frame-end-inclusive {args.frame_end_inclusive} exceeds {len(frames)} frames"
        )
    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="booster_t1_29dof",
        actual_human_height=actual_human_height,
        verbose=False,
    )
    selected_qpos: list[np.ndarray] = []
    left_foot: list[np.ndarray] = []
    right_foot: list[np.ndarray] = []
    for frame_index in range(args.retarget_start, args.frame_end_inclusive + 1):
        qpos = retargeter.retarget(frames[frame_index])
        if frame_index < args.frame_start:
            continue
        selected_qpos.append(qpos)
        left_foot.append(np.asarray(frames[frame_index]["LeftFootMod"][0]))
        right_foot.append(np.asarray(frames[frame_index]["RightFootMod"][0]))

    contract = load_policy_contract(args.contract)
    source_qpos = np.stack(selected_qpos)
    mapped_qpos = map_named_qpos_to_contract(
        source_qpos,
        _source_qpos_addresses(retargeter.model),
        contract.joint_order,
    )
    limit_model = build_single_t1_soccer_model(prefix="gmr_limit_", robot_x=0.0)
    joint_lower = np.array(
        [
            limit_model.joint("gmr_limit_" + name).range[0]
            for name in contract.joint_order
        ]
    )
    joint_upper = np.array(
        [
            limit_model.joint("gmr_limit_" + name).range[1]
            for name in contract.joint_order
        ]
    )
    mapped_qpos, limit_clipping = clip_contract_joint_limits(
        mapped_qpos, joint_lower, joint_upper
    )
    human_joints, source_ground = contact_only_human_joints(
        np.stack(left_foot), np.stack(right_foot)
    )
    args.intermediate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.intermediate,
        qpos=mapped_qpos.astype(np.float32),
        human_joints=human_joints.astype(np.float32),
        fps=np.array(args.input_fps, dtype=np.float32),
    )

    command = " ".join(shlex.quote(value) for value in sys.argv)
    provenance = {
        "source_url": args.source_url,
        "source_version": args.source_version,
        "source_license": args.source_license,
        "source_sha256": _sha256(args.input),
        "gmr_repository": "https://github.com/YanjieZe/GMR",
        "gmr_revision": actual_gmr_revision,
        "gmr_license": "MIT",
        "gmr_robot": "booster_t1_29dof",
        "gmr_retarget_range_inclusive": [args.retarget_start, args.frame_end_inclusive],
        "selected_source_frame_range_inclusive": [
            args.frame_start,
            args.frame_end_inclusive,
        ],
        "actual_human_height_m": float(actual_human_height),
        "source_contact_ground_offset_m": source_ground,
        "intermediate_sha256": _sha256(args.intermediate),
        "joint_mapping": "MuJoCo qpos address by exact joint name",
        "competition_joint_limit_clipping": limit_clipping,
        "zero_filled_joints": ["AAHead_yaw", "Head_pitch"],
        "dropped_source_joints": [
            "Left_Wrist_Pitch",
            "Left_Wrist_Yaw",
            "Left_Hand_Roll",
            "Right_Wrist_Pitch",
            "Right_Wrist_Yaw",
            "Right_Hand_Roll",
        ],
        "conversion_command": command,
    }
    arrays, metadata = build_motion_reference(
        args.intermediate,
        contract,
        output_fps=args.output_fps,
        frame_start=0,
        frame_end_inclusive=None,
        source_height_threshold=args.source_contact_height,
        provenance=provenance,
    )
    # Replace helper-specific field names with the actual importer identity.
    metadata["gmr_intermediate"] = metadata.pop("holosoma_input")
    metadata["gmr_intermediate_frame_range_inclusive"] = metadata.pop(
        "holosoma_frame_range_inclusive"
    )
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    result = validate_motion_reference(args.output)
    result["output_sha256"] = _sha256(args.output)
    result["intermediate_sha256"] = _sha256(args.intermediate)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
