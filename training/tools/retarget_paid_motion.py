#!/usr/bin/env python3
"""Build a local-only Booster T1 candidate from one pinned PAiD G1 motion."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import sys

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.gmr_motion import clip_contract_joint_limits, map_named_qpos_to_contract
from my3d_rl.paid_motion import (
    PAID_SCHEMA_REVISION,
    PAID_SOURCE_LICENSE,
    file_sha256,
    load_paid_motion,
    paid_frame_for_gmr,
    semantic_projection_qpos,
    source_foot_contact,
)
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.soccer_motion_reference import (
    build_soccer_motion_reference,
    validate_soccer_motion_reference,
)


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    result: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result[name] = int(model.jnt_qposadr[joint_id])
    return result


def _retarget_with_gmr(
    clip,
    contract,
    *,
    gmr_root: Path,
    gmr_revision: str,
    robot_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not 0.5 <= robot_scale <= 1.2:
        raise ValueError("robot-scale must lie in [0.5, 1.2]")
    actual_revision = _git_revision(gmr_root)
    if actual_revision != gmr_revision:
        raise ValueError(
            f"GMR revision {actual_revision} != requested {gmr_revision}"
        )
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="booster_t1_29dof",
        verbose=False,
        damping=0.5,
    )
    # GMR's SMPLX config assumes an adult-human source and scales it by 0.6/0.7.
    # PAiD already contains robot-world link poses, so use one explicit
    # robot-to-robot scale for all targets instead of applying human proportions.
    retargeter.human_scale_table = {
        name: robot_scale for name in retargeter.human_scale_table
    }
    semantic = semantic_projection_qpos(clip, contract.joint_order)
    semantic[:, :3] *= robot_scale
    initial_qpos = retargeter.model.qpos0.copy()
    initial_qpos[:7] = semantic[0, :7]
    target_addresses = _source_qpos_addresses(retargeter.model)
    for source_index, name in enumerate(contract.joint_order, start=7):
        if name in target_addresses:
            initial_qpos[target_addresses[name]] = semantic[0, source_index]
    initialization_clip_count = 0
    initialization_clip_max = 0.0
    for joint_id in range(retargeter.model.njnt):
        if not retargeter.model.jnt_limited[joint_id]:
            continue
        address = int(retargeter.model.jnt_qposadr[joint_id])
        before = float(initial_qpos[address])
        initial_qpos[address] = np.clip(
            before,
            retargeter.model.jnt_range[joint_id, 0],
            retargeter.model.jnt_range[joint_id, 1],
        )
        correction = abs(float(initial_qpos[address]) - before)
        initialization_clip_count += int(correction > 0.0)
        initialization_clip_max = max(initialization_clip_max, correction)
    retargeter.configuration.update(initial_qpos)

    # The upstream offsets convert SMPLX anatomical frames into T1 frames.
    # PAiD supplies G1 link frames instead.  Calibrate the constant body-frame
    # offsets from the semantically projected first pose so the IK starts from
    # an evidence-backed robot-to-robot correspondence, not an SMPLX guess.
    source_first = paid_frame_for_gmr(clip, 0)
    calibrated: dict[str, tuple[np.ndarray, Rotation]] = {}
    for table in (retargeter.ik_match_table1, retargeter.ik_match_table2):
        for target_body, entry in table.items():
            human_name, position_weight, rotation_weight, *_ = entry
            if position_weight == 0 and rotation_weight == 0:
                continue
            body_id = retargeter.model.body(target_body).id
            target_position = retargeter.configuration.data.xpos[body_id].copy()
            target_rotation = Rotation.from_quat(
                retargeter.configuration.data.xquat[body_id], scalar_first=True
            )
            source_position, source_quaternion = source_first[human_name]
            source_rotation = Rotation.from_quat(
                source_quaternion, scalar_first=True
            )
            rotation_offset = source_rotation.inv() * target_rotation
            local_position_offset = target_rotation.inv().apply(
                target_position - robot_scale * source_position
            )
            calibrated[human_name] = (local_position_offset, rotation_offset)
    for human_name, (position_offset, rotation_offset) in calibrated.items():
        if human_name in retargeter.pos_offsets1:
            retargeter.pos_offsets1[human_name] = position_offset
            retargeter.rot_offsets1[human_name] = rotation_offset
        if human_name in retargeter.pos_offsets2:
            retargeter.pos_offsets2[human_name] = position_offset
            retargeter.rot_offsets2[human_name] = rotation_offset

    # Re-solve the first target several times from the calibrated semantic pose
    # before recording it; otherwise solver start-up appears as a false kick
    # velocity spike at frame zero.
    first_frame = paid_frame_for_gmr(clip, 0)
    for _ in range(3):
        retargeter.retarget(first_frame)
    qpos = np.stack(
        [
            retargeter.retarget(paid_frame_for_gmr(clip, frame_index))
            for frame_index in range(clip.frame_count)
        ]
    )
    mapped = map_named_qpos_to_contract(
        qpos,
        _source_qpos_addresses(retargeter.model),
        contract.joint_order,
    )
    return mapped, {
        "gmr_repository": "https://github.com/YanjieZe/GMR",
        "gmr_revision": actual_revision,
        "gmr_license": "MIT",
        "gmr_source_type": "smplx logical-body interface",
        "gmr_target_robot": "booster_t1_29dof",
        "robot_to_robot_position_scale": robot_scale,
        "gmr_offset_calibration": "semantic_first_frame_body_pose",
        "gmr_initialization_limit_clipping": {
            "clipped_value_count": initialization_clip_count,
            "maximum_abs_correction_rad": initialization_clip_max,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="local-only T1 reference NPZ")
    parser.add_argument("--paid-root", type=Path, required=True)
    parser.add_argument("--paid-revision", default=PAID_SCHEMA_REVISION)
    parser.add_argument(
        "--method", choices=("semantic", "gmr-body-ik"), default="gmr-body-ik"
    )
    parser.add_argument("--gmr-root", type=Path)
    parser.add_argument("--gmr-revision")
    parser.add_argument("--robot-scale", type=float, default=0.9)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end-inclusive", type=int)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("training/contracts/run_policy_v3.yaml"),
    )
    parser.add_argument("--source-contact-height", type=float, default=0.04)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    paid_root = args.paid_root.resolve()
    input_path = args.input.resolve()
    try:
        input_path.relative_to(paid_root)
    except ValueError as exc:
        raise SystemExit("input must stay inside the audited PAiD clone") from exc
    actual_paid_revision = _git_revision(paid_root)
    if actual_paid_revision != args.paid_revision:
        raise SystemExit(
            f"PAiD revision {actual_paid_revision} != requested {args.paid_revision}"
        )
    clip = load_paid_motion(input_path)
    end = (
        clip.frame_count - 1
        if args.frame_end_inclusive is None
        else args.frame_end_inclusive
    )
    if not 0 <= args.frame_start <= end < clip.frame_count:
        raise SystemExit(
            f"invalid frame range {args.frame_start}:{end} for {clip.frame_count} frames"
        )
    full_contact, contact_diagnostics = source_foot_contact(
        clip, height_tolerance_m=args.source_contact_height
    )
    selection = slice(args.frame_start, end + 1)
    clip = replace(
        clip,
        joint_position=clip.joint_position[selection],
        joint_velocity=clip.joint_velocity[selection],
        body_position_world=clip.body_position_world[selection],
        body_quaternion_wxyz=clip.body_quaternion_wxyz[selection],
        body_linear_velocity_world=clip.body_linear_velocity_world[selection],
        body_angular_velocity_world=clip.body_angular_velocity_world[selection],
    )
    source_contact = full_contact[selection]
    contact_diagnostics = dict(contact_diagnostics)
    contact_diagnostics["selected_contact_frames"] = (
        source_contact.sum(axis=0).astype(int).tolist()
    )
    contract = load_policy_contract(args.contract)
    method_details: dict[str, object] = {}
    if args.method == "semantic":
        mapped_qpos = semantic_projection_qpos(clip, contract.joint_order)
        method_details["semantic_mapping"] = (
            "same-name leg/shoulder/elbow projection; G1 wrist-roll to T1 "
            "elbow-yaw; head zero; waist-yaw only"
        )
    else:
        if args.gmr_root is None or args.gmr_revision is None:
            raise SystemExit("gmr-body-ik requires --gmr-root and --gmr-revision")
        mapped_qpos, method_details = _retarget_with_gmr(
            clip,
            contract,
            gmr_root=args.gmr_root.resolve(),
            gmr_revision=args.gmr_revision,
            robot_scale=args.robot_scale,
        )

    limit_model = build_single_t1_soccer_model(prefix="paid_limit_", robot_x=0.0)
    joint_lower = np.array(
        [
            limit_model.joint("paid_limit_" + name).range[0]
            for name in contract.joint_order
        ]
    )
    joint_upper = np.array(
        [
            limit_model.joint("paid_limit_" + name).range[1]
            for name in contract.joint_order
        ]
    )
    mapped_qpos, clipping = clip_contract_joint_limits(
        mapped_qpos, joint_lower, joint_upper
    )
    command = " ".join(shlex.quote(value) for value in sys.argv)
    provenance = {
        "source_url": "https://github.com/TeleHuman/HumanoidSoccer",
        "source_version": actual_paid_revision,
        "source_license": PAID_SOURCE_LICENSE,
        "source_sha256": file_sha256(input_path),
        "source_relative_path": str(input_path.relative_to(paid_root)),
        "source_frame_range_inclusive": [args.frame_start, end],
        "source_contact": contact_diagnostics,
        "retarget_method": args.method,
        "competition_joint_limit_clipping": clipping,
        "conversion_command": command,
        "reuse_policy": (
            "local attributed non-commercial research only; output must not "
            "be committed or redistributed"
        ),
        **method_details,
    }
    arrays, _ = build_soccer_motion_reference(
        mapped_qpos,
        source_contact,
        contract,
        input_fps=clip.fps,
        output_fps=50.0,
        kick_leg=clip.kick_leg,
        provenance=provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    result = validate_soccer_motion_reference(args.output)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
