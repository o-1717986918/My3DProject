#!/usr/bin/env python3
"""Optimize and serialize a reproducible exact-physics kick teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_env import DEFAULT_CONTRACT
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, PARAMETER_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--target-angle", type=float, default=0.0)
    parser.add_argument("--requested-speed", type=float, default=1.43)
    parser.add_argument("--desired-arrival-speed", type=float, default=1.0)
    parser.add_argument("--mode", choices=("pass", "shot", "clear"), default="pass")
    parser.add_argument(
        "--motion-base",
        choices=("walk", "stand"),
        default="walk",
        help="walk uses the Apollo ONNX base; stand is a model-free joint baseline",
    )
    parser.add_argument(
        "--stand-support-crouch-rad",
        type=float,
        default=0.0,
        help="model-free support-knee compliance used only by the stand base",
    )
    parser.add_argument(
        "--stand-base-pose",
        choices=("bent", "neutral"),
        default="bent",
        help="complete position-control base used by model-free trajectories",
    )
    parser.add_argument("--ball-x-offset", type=float, default=0.0)
    parser.add_argument("--ball-y-offset", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument(
        "--initial-manifest",
        type=Path,
        help="warm-start CEM from parameters in a previous teacher manifest",
    )
    parser.add_argument(
        "--initial-parameters",
        help="comma-separated 14-value warm start (exclusive with --initial-manifest)",
    )
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=1.0,
        help="bounded scale applied to --initial-manifest parameters",
    )
    parser.add_argument(
        "--robust-samples",
        type=int,
        default=1,
        help="fixed nominal/noisy ball placements scored per CEM candidate",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/win98/rl_runs/kick-teacher/kick-v2"),
    )
    args = parser.parse_args()
    if args.initial_scale <= 0.0 or args.initial_scale > 1.0:
        raise ValueError("--initial-scale must be in (0, 1]")

    if args.initial_manifest is not None and args.initial_parameters is not None:
        raise ValueError(
            "--initial-manifest and --initial-parameters are mutually exclusive"
        )
    initial_parameters = None
    if args.initial_manifest is not None:
        source = json.loads(args.initial_manifest.read_text(encoding="utf-8"))
        initial_parameters = (
            np.asarray(source["parameters"], dtype=np.float64)
            * args.initial_scale
        )
        if initial_parameters.shape != (len(PARAMETER_NAMES),):
            raise ValueError("initial manifest has an incompatible parameter vector")
    elif args.initial_parameters is not None:
        initial_parameters = np.asarray(
            [float(value) for value in args.initial_parameters.split(",")],
            dtype=np.float64,
        ) * args.initial_scale
        if initial_parameters.shape != (len(PARAMETER_NAMES),):
            raise ValueError("--initial-parameters must contain exactly 14 values")

    spec = KickTeacherSpec(
        target_distance_m=args.target_distance,
        target_angle_deg=args.target_angle,
        requested_ball_speed_mps=args.requested_speed,
        desired_arrival_speed_mps=args.desired_arrival_speed,
        action_mode=args.mode,
    )
    evaluator = KickTeacherEvaluator(
        spec,
        motion_base=args.motion_base,
        stand_base_pose=args.stand_base_pose,
        stand_support_crouch_rad=args.stand_support_crouch_rad,
    )
    result = evaluator.optimize(
        seed=args.seed,
        population=args.population,
        generations=args.generations,
        robust_samples=args.robust_samples,
        ball_x_offset_m=args.ball_x_offset,
        ball_y_offset_m=args.ball_y_offset,
        initial_parameters=initial_parameters,
    )
    times, observations, actions, joint_targets, metrics = evaluator.demonstration(
        result.parameters,
        ball_x_offset_m=args.ball_x_offset,
        ball_y_offset_m=args.ball_y_offset,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_prefix.with_suffix(".npz")
    manifest_path = args.output_prefix.with_suffix(".json")
    np.savez_compressed(
        trajectory_path,
        times_s=times,
        joint_targets_rad=joint_targets,
        observations=observations,
        actions=actions,
        parameters=result.parameters,
        parameter_names=np.asarray(PARAMETER_NAMES),
        joint_order=np.asarray(evaluator.contract.joint_order),
    )
    manifest = {
        "purpose": "r1_low_dimensional_kick_teacher",
        "promotable": False,
        "promotion_blocker": "requires multi-seed envelope and RCSSServerMJ validation",
        "contract": str(DEFAULT_CONTRACT),
        "contract_sha256": _sha256(DEFAULT_CONTRACT),
        "motion_base": evaluator.motion_base,
        "stand_base_pose": evaluator.stand_base_pose,
        "stand_support_crouch_rad": evaluator.stand_support_crouch_rad,
        "base_walk_policy": (
            str(evaluator.walk_policy_path)
            if evaluator.walk_policy_path is not None
            else None
        ),
        "base_walk_policy_sha256": (
            _sha256(evaluator.walk_policy_path)
            if evaluator.walk_policy_path is not None
            else None
        ),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "seed": args.seed,
        "population": args.population,
        "generations": args.generations,
        "robust_samples": args.robust_samples,
        "initial_manifest": (
            str(args.initial_manifest.resolve())
            if args.initial_manifest is not None
            else None
        ),
        "initial_manifest_sha256": (
            _sha256(args.initial_manifest)
            if args.initial_manifest is not None
            else None
        ),
        "initial_scale": args.initial_scale,
        "initial_parameters_supplied": args.initial_parameters is not None,
        "spec": {
            "target_distance_m": spec.target_distance_m,
            "target_angle_deg": spec.target_angle_deg,
            "requested_ball_speed_mps": spec.requested_ball_speed_mps,
            "desired_arrival_speed_mps": spec.desired_arrival_speed_mps,
            "action_mode": spec.action_mode,
            "duration_s": spec.duration_s,
            "evaluation_duration_s": spec.evaluation_duration_s,
            "control_dt_s": spec.control_dt_s,
            "simulation_dt_s": spec.simulation_dt_s,
        },
        "ball_offset_m": {
            "x": args.ball_x_offset,
            "y": args.ball_y_offset,
        },
        "parameter_names": list(PARAMETER_NAMES),
        "parameters": result.parameters.tolist(),
        "score": result.score,
        "metrics": metrics,
        "history": list(result.history),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
