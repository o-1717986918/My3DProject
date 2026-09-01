#!/usr/bin/env python3
"""Evaluate a finite T1 soccer-motion corpus in exact CPU dynamics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.reference_dynamics import (
    nonperiodic_failure_frame_sampling_weights,
)
from my3d_rl.soccer_motion_dynamics import (
    load_soccer_motion_arrays,
    replay_soccer_motion_reference,
)
from my3d_rl.soccer_motion_reference import validate_soccer_motion_reference
from my3d_rl.t1_control import apollo_joint_gains


REPOSITORY_ROOT = Path(__file__).parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _start_frames(
    frame_count: int, phase_samples: int, minimum_remaining_frames: int
) -> list[int]:
    final_start = max(0, frame_count - minimum_remaining_frames)
    samples = np.linspace(0, final_start, phase_samples, dtype=np.int64)
    return sorted(set(int(value) for value in samples if value < frame_count - 1))


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    full_replays = [record["full_replay"] for record in records]
    phase_replays = [
        replay for record in records for replay in record["phase_replays"]
    ]
    failures = Counter(
        replay["termination_reason"]
        for replay in phase_replays
        if not replay["completed"]
    )
    return {
        "motion_count": len(records),
        "full_clip_completed_count": int(
            sum(replay["completed"] for replay in full_replays)
        ),
        "full_clip_screening_passed_count": int(
            sum(replay["screening_passed"] for replay in full_replays)
        ),
        "full_clip_completion_rate": float(
            np.mean([replay["completed"] for replay in full_replays])
        ),
        "phase_episode_count": len(phase_replays),
        "phase_completion_rate": float(
            np.mean([replay["completed"] for replay in phase_replays])
        ),
        "mean_phase_termination_phase": float(
            np.mean([replay["termination_phase"] for replay in phase_replays])
        ),
        "mean_full_clip_joint_tracking_rmse_rad": float(
            np.mean(
                [replay["joint_tracking_rmse_rad"] for replay in full_replays]
            )
        ),
        "maximum_full_clip_joint_tracking_rmse_rad": float(
            np.max(
                [replay["joint_tracking_rmse_rad"] for replay in full_replays]
            )
        ),
        "mean_full_clip_contact_agreement": float(
            np.mean([replay["foot_contact_agreement"] for replay in full_replays])
        ),
        "failure_reasons": dict(sorted(failures.items())),
    }


def _controller_gains(profile: str, joint_order: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if profile == "legacy_run_v3":
        return np.full(len(joint_order), 25.0), np.full(len(joint_order), 0.6)
    if profile == "selected_run_v4":
        return np.full(len(joint_order), 50.0), np.full(len(joint_order), 1.2)
    if profile == "apollo_runtime":
        pairs = np.asarray([apollo_joint_gains(name) for name in joint_order])
        return pairs[:, 0], pairs[:, 1]
    raise ValueError(f"unsupported gain profile {profile!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY_ROOT / "training/contracts/run_policy_v3.yaml",
    )
    parser.add_argument("--phase-samples", type=int, default=8)
    parser.add_argument("--minimum-remaining-frames", type=int, default=10)
    parser.add_argument("--target-lead-frames", type=int, default=0)
    parser.add_argument("--failure-kernel-size", type=int, default=5)
    parser.add_argument("--failure-lead-frames", type=int, default=0)
    parser.add_argument(
        "--gain-profile",
        choices=("legacy_run_v3", "selected_run_v4", "apollo_runtime"),
        default="legacy_run_v3",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase_samples < 1 or args.minimum_remaining_frames < 2:
        raise ValueError("phase samples and minimum remaining frames are too small")
    if (
        args.target_lead_frames < 0
        or args.failure_kernel_size < 1
        or args.failure_lead_frames < 0
    ):
        raise ValueError("invalid controller configuration")
    paths = sorted(args.corpus_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ motions found below {args.corpus_root}")

    contract = load_policy_contract(args.contract)
    kp, kd = _controller_gains(args.gain_profile, contract.joint_order)
    prefix = "soccer_track_"
    model = build_single_t1_soccer_model(prefix=prefix, robot_x=-10.0, robot_y=0.0)
    model.opt.timestep = 0.005
    records: list[dict[str, Any]] = []
    for path in paths:
        validation = validate_soccer_motion_reference(path)
        if not validation["passed"]:
            raise ValueError(f"K0 validation failed for {path}: {validation['errors']}")
        reference = load_soccer_motion_arrays(path)
        frame_count = int(reference["joint_position"].shape[0])
        starts = _start_frames(
            frame_count, args.phase_samples, args.minimum_remaining_frames
        )
        replays = [
            replay_soccer_motion_reference(
                model,
                contract,
                reference,
                prefix=prefix,
                start_frame=start,
                target_lead_frames=args.target_lead_frames,
                kp=kp,
                kd=kd,
            )
            for start in starts
        ]
        failures = np.asarray(
            [
                replay["termination_frame"]
                for replay in replays
                if not replay["completed"]
            ],
            dtype=np.int64,
        )
        weights = nonperiodic_failure_frame_sampling_weights(
            failures,
            frame_count=frame_count,
            kernel_size=args.failure_kernel_size,
            lead_frames=args.failure_lead_frames,
        )
        records.append(
            {
                "relative_path": str(path.relative_to(args.corpus_root)),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "frame_count": frame_count,
                "start_frames": starts,
                "full_replay": replays[starts.index(0)],
                "phase_replays": replays,
                "failure_frame_sampling": {
                    "algorithm": (
                        "finite_pre_failure_exponential_kernel_plus_uniform"
                    ),
                    "kernel_size": args.failure_kernel_size,
                    "kernel_decay": 0.8,
                    "uniform_ratio": 0.1,
                    "lead_frames": args.failure_lead_frames,
                    "failure_frames": failures.tolist(),
                    "weights": weights.tolist(),
                    "normalized_entropy": float(
                        -np.sum(weights * np.log(weights)) / np.log(frame_count)
                    ),
                    "maximum_probability_frame": int(np.argmax(weights)),
                },
            }
        )
        full = records[-1]["full_replay"]
        print(
            f"{records[-1]['relative_path']}: completed={full['completed']} "
            f"frame={full['termination_frame']} "
            f"rmse={full['joint_tracking_rmse_rad']:.4f}",
            file=sys.stderr,
        )

    payload = {
        "schema_version": 1,
        "purpose": "paid_t1_k1_zero_residual_dynamic_corpus_screening",
        "status": "complete",
        "method": args.method,
        "corpus_root": str(args.corpus_root.resolve()),
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "engine": f"MuJoCo {mujoco.__version__}",
        "protocol": {
            "control_frequency_hz": contract.frequency_hz,
            "physics_timestep_s": float(model.opt.timestep),
            "gain_profile": args.gain_profile,
            "kp": kp.tolist(),
            "kd": kd.tolist(),
            "target_lead_frames": args.target_lead_frames,
            "phase_samples": args.phase_samples,
            "minimum_remaining_frames": args.minimum_remaining_frames,
            "failure_kernel_size": args.failure_kernel_size,
            "failure_lead_frames": args.failure_lead_frames,
            "initialization": "exact_reference_state_at_start_frame",
            "learned_residual": "zero",
            "scene": "exact_RCSSServerMJ_soccer_world_plus_T1",
        },
        "aggregate": _aggregate(records),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
