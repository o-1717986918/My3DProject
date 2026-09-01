#!/usr/bin/env python3
"""Generate exact-CPU walk-to-kick entry states without frame leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_buckets(phases: np.ndarray, bucket_count: int) -> np.ndarray:
    """Map ``sin, cos`` pairs onto equal angular buckets."""
    values = np.asarray(phases, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("locomotion phases must have shape [N, 2]")
    if bucket_count < 2:
        raise ValueError("phase bucket count must be at least two")
    if not np.isfinite(values).all():
        raise ValueError("locomotion phases must be finite")
    angle = np.mod(np.arctan2(values[:, 0], values[:, 1]), 2.0 * np.pi)
    return np.minimum(
        np.floor(angle * bucket_count / (2.0 * np.pi)).astype(np.int32),
        bucket_count - 1,
    )


def stratified_rollout_split(
    buckets: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
) -> np.ndarray:
    """Assign whole rollout rows to train/validation within phase buckets."""
    values = np.asarray(buckets, dtype=np.int32)
    if values.ndim != 1:
        raise ValueError("phase buckets must be one-dimensional")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation fraction must be in (0, 0.5)")
    split = np.zeros(values.shape[0], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for bucket in sorted(set(values.tolist())):
        indices = np.flatnonzero(values == bucket)
        if indices.size < 2:
            continue
        order = rng.permutation(indices)
        validation_count = int(round(indices.size * validation_fraction))
        validation_count = min(max(validation_count, 1), indices.size - 1)
        split[order[:validation_count]] = 1
    return split


def _accepted_condition(source: dict[str, object], condition_index: int) -> dict:
    matches = [
        record
        for record in source.get("records", [])
        if int(record["condition_index"]) == condition_index
        and bool(record["accepted"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "condition index must select exactly one accepted teacher record"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("--condition-index", type=int, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--initial-x-min", type=float, default=-0.45)
    parser.add_argument("--initial-x-max", type=float, default=-0.05)
    parser.add_argument("--initial-y-min", type=float, default=-0.20)
    parser.add_argument("--initial-y-max", type=float, default=0.20)
    parser.add_argument("--initial-yaw-min-deg", type=float, default=-12.0)
    parser.add_argument("--initial-yaw-max-deg", type=float, default=12.0)
    parser.add_argument("--ball-x-jitter", type=float, default=0.01)
    parser.add_argument("--ball-y-jitter", type=float, default=0.015)
    parser.add_argument("--setup-timeout", type=float, default=4.0)
    parser.add_argument("--setup-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--setup-confirmation-cycles",
        type=int,
        help="force one hold length; by default it is randomized per rollout",
    )
    parser.add_argument("--setup-confirmation-min-cycles", type=int, default=5)
    parser.add_argument("--setup-confirmation-max-cycles", type=int, default=35)
    parser.add_argument("--phase-buckets", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.rollouts < 2:
        raise ValueError("rollouts must be at least two")
    ranges = np.asarray(
        [
            args.initial_x_min,
            args.initial_x_max,
            args.initial_y_min,
            args.initial_y_max,
            args.initial_yaw_min_deg,
            args.initial_yaw_max_deg,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(ranges).all():
        raise ValueError("initial pose ranges must be finite")
    if not (
        args.initial_x_min < args.initial_x_max
        and args.initial_y_min < args.initial_y_max
        and args.initial_yaw_min_deg < args.initial_yaw_max_deg
    ):
        raise ValueError("initial pose ranges must be increasing")
    if args.ball_x_jitter < 0.0 or args.ball_y_jitter < 0.0:
        raise ValueError("ball jitter must be non-negative")
    if not (
        1
        <= args.setup_confirmation_min_cycles
        <= args.setup_confirmation_max_cycles
    ):
        raise ValueError("setup confirmation cycle range is invalid")
    if args.setup_confirmation_cycles is not None and (
        args.setup_confirmation_cycles < 1
    ):
        raise ValueError("setup confirmation cycles must be positive")

    npz_path = args.output_prefix.with_suffix(".npz")
    manifest_path = args.output_prefix.with_suffix(".json")
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError("transition corpus outputs already exist")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    record = _accepted_condition(source, args.condition_index)
    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3" or contract.observation_size != 98:
        raise ValueError("transition corpus requires the kick_policy_v3 contract")
    spec = KickTeacherSpec(
        target_distance_m=float(record["distance_m"]),
        target_angle_deg=float(record["angle_deg"]),
        requested_ball_speed_mps=float(record["requested_speed_mps"]),
        desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
        action_mode=str(record["mode"]),
        evaluation_duration_s=args.setup_timeout + 3.0,
    )
    evaluator = KickTeacherEvaluator(spec, contract=contract)
    parameters = np.asarray(record["parameters"], dtype=np.float64)
    base_ball_x = float(record["ball_x_offset_m"])
    base_ball_y = float(record["ball_y_offset_m"])
    rng = np.random.default_rng(args.seed)
    entries: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for rollout_id in range(args.rollouts):
        initial_offset = np.array(
            [
                rng.uniform(args.initial_x_min, args.initial_x_max),
                rng.uniform(args.initial_y_min, args.initial_y_max),
            ],
            dtype=np.float64,
        )
        initial_yaw = float(
            rng.uniform(args.initial_yaw_min_deg, args.initial_yaw_max_deg)
        )
        ball_x = base_ball_x + float(
            rng.uniform(-args.ball_x_jitter, args.ball_x_jitter)
        )
        ball_y = base_ball_y + float(
            rng.uniform(-args.ball_y_jitter, args.ball_y_jitter)
        )
        confirmation_cycles = (
            args.setup_confirmation_cycles
            if args.setup_confirmation_cycles is not None
            else int(
                rng.integers(
                    args.setup_confirmation_min_cycles,
                    args.setup_confirmation_max_cycles + 1,
                )
            )
        )
        metrics = evaluator.rollout(
            parameters,
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
            setup_ball_x_offset_m=base_ball_x,
            setup_ball_y_offset_m=base_ball_y,
            setup_timeout_s=args.setup_timeout,
            setup_tolerance_m=args.setup_tolerance,
            setup_confirmation_cycles=confirmation_cycles,
            initial_robot_offset_m=(float(initial_offset[0]), float(initial_offset[1])),
            initial_robot_yaw_deg=initial_yaw,
            capture_transition_entry=True,
        )
        entry = evaluator.captured_transition_entry
        if (
            not bool(metrics["setup_succeeded"])
            or entry is None
            or entry.torso_height_m < 0.45
            or entry.upright < 0.75
        ):
            rejected.append(
                {
                    "rollout_id": rollout_id,
                    "initial_robot_offset_m": initial_offset.tolist(),
                    "initial_robot_yaw_deg": initial_yaw,
                    "setup_succeeded": bool(metrics["setup_succeeded"]),
                    "setup_timed_out": bool(metrics["setup_timed_out"]),
                    "setup_duration_s": float(metrics["setup_duration_s"]),
                    "setup_confirmation_cycles": confirmation_cycles,
                }
            )
            continue
        entries.append(
            {
                "rollout_id": rollout_id,
                "entry": entry,
                "initial_offset": initial_offset,
                "initial_yaw": initial_yaw,
                "ball_offset": np.array([ball_x, ball_y], dtype=np.float64),
                "setup_duration": float(metrics["setup_duration_s"]),
                "setup_confirmation_cycles": confirmation_cycles,
                "contact": bool(metrics["contact"]),
                "fell": bool(metrics["fell"]),
                "maximum_progress": float(metrics["maximum_progress_m"]),
            }
        )

    if len(entries) < 2:
        raise RuntimeError("fewer than two valid transition entries were generated")
    phases = np.stack([item["entry"].locomotion_phase for item in entries])
    buckets = phase_buckets(phases, args.phase_buckets)
    split = stratified_rollout_split(
        buckets,
        seed=args.seed + 1,
        validation_fraction=args.validation_fraction,
    )

    arrays = {
        "rollout_id": np.asarray([item["rollout_id"] for item in entries], np.int32),
        "qpos": np.stack([item["entry"].qpos for item in entries]).astype(np.float32),
        "qvel": np.stack([item["entry"].qvel for item in entries]).astype(np.float32),
        "joint_position_offset": np.stack(
            [item["entry"].joint_position_offset for item in entries]
        ).astype(np.float32),
        "joint_velocity": np.stack(
            [item["entry"].joint_velocity for item in entries]
        ).astype(np.float32),
        "walk_previous_action": np.stack(
            [item["entry"].walk_previous_action for item in entries]
        ).astype(np.float32),
        "setup_velocity_command": np.stack(
            [item["entry"].setup_velocity_command for item in entries]
        ).astype(np.float32),
        "locomotion_phase": phases.astype(np.float32),
        "support_hint": np.stack(
            [item["entry"].support_hint for item in entries]
        ).astype(np.float32),
        "phase_magnitude_rad": np.asarray(
            [item["entry"].phase_magnitude_rad for item in entries], np.float32
        ),
        "ball_position_local_m": np.stack(
            [item["entry"].ball_position_local_m for item in entries]
        ).astype(np.float32),
        "root_velocity": np.stack(
            [item["entry"].root_velocity for item in entries]
        ).astype(np.float32),
        "initial_robot_offset_m": np.stack(
            [item["initial_offset"] for item in entries]
        ).astype(np.float32),
        "initial_robot_yaw_deg": np.asarray(
            [item["initial_yaw"] for item in entries], np.float32
        ),
        "ball_offset_m": np.stack(
            [item["ball_offset"] for item in entries]
        ).astype(np.float32),
        "setup_duration_s": np.asarray(
            [item["setup_duration"] for item in entries], np.float32
        ),
        "setup_confirmation_cycles": np.asarray(
            [item["setup_confirmation_cycles"] for item in entries], np.int32
        ),
        "outcome_contact": np.asarray(
            [item["contact"] for item in entries], np.uint8
        ),
        "outcome_fell": np.asarray([item["fell"] for item in entries], np.uint8),
        "outcome_maximum_progress_m": np.asarray(
            [item["maximum_progress"] for item in entries], np.float32
        ),
        "phase_bucket": buckets,
        "split": split,
    }
    np.savez_compressed(npz_path, **arrays)

    phase_histogram = {
        str(bucket): {
            "total": int(np.count_nonzero(buckets == bucket)),
            "train": int(np.count_nonzero((buckets == bucket) & (split == 0))),
            "validation": int(
                np.count_nonzero((buckets == bucket) & (split == 1))
            ),
        }
        for bucket in range(args.phase_buckets)
    }
    manifest = {
        "schema_version": 1,
        "purpose": "kick_policy_v3_walk_to_kick_transition_corpus",
        "promotable": False,
        "promotion_blocker": "training input only; policy gates remain required",
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "teacher_condition_index": args.condition_index,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "seed": args.seed,
        "requested_rollouts": args.rollouts,
        "accepted_entries": len(entries),
        "rejected_entries": len(rejected),
        "train_entries": int(np.count_nonzero(split == 0)),
        "validation_entries": int(np.count_nonzero(split == 1)),
        "split_unit": "whole_rollout_entry",
        "split_stratification": "locomotion_phase_bucket",
        "phase_bucket_count": args.phase_buckets,
        "phase_histogram": phase_histogram,
        "sampling": {
            "initial_x_m": [args.initial_x_min, args.initial_x_max],
            "initial_y_m": [args.initial_y_min, args.initial_y_max],
            "initial_yaw_deg": [
                args.initial_yaw_min_deg,
                args.initial_yaw_max_deg,
            ],
            "ball_x_jitter_m": args.ball_x_jitter,
            "ball_y_jitter_m": args.ball_y_jitter,
            "setup_timeout_s": args.setup_timeout,
            "setup_tolerance_m": args.setup_tolerance,
            "setup_confirmation_cycles": (
                args.setup_confirmation_cycles
                if args.setup_confirmation_cycles is not None
                else [
                    args.setup_confirmation_min_cycles,
                    args.setup_confirmation_max_cycles,
                ]
            ),
        },
        "outcomes": {
            "contacts": int(np.count_nonzero(arrays["outcome_contact"])),
            "falls": int(np.count_nonzero(arrays["outcome_fell"])),
        },
        "npz": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "rejections": rejected,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
