#!/usr/bin/env python3
"""Generate rollout-grouped exact-CPU labels for walk-to-kick switch timing."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import (
    KickTeacherEvaluator,
    KickTeacherSpec,
    KickTransitionEntry,
    kick_trial_success,
)
from tools.generate_kick_transition_corpus import phase_buckets


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"
_WORKER_EVALUATOR: KickTeacherEvaluator | None = None
_WORKER_PARAMETERS: np.ndarray | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def rollout_group_split(
    rollout_ids: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
) -> np.ndarray:
    """Split complete approach rollouts so adjacent switch frames never leak."""
    values = np.asarray(rollout_ids, dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("rollout IDs must contain at least two rows")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation fraction must be in (0, 0.5)")
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError("at least two distinct approach rollouts are required")
    order = np.random.default_rng(seed).permutation(unique)
    validation_count = int(round(unique.size * validation_fraction))
    validation_count = min(max(validation_count, 1), unique.size - 1)
    validation = set(order[:validation_count].tolist())
    return np.asarray([value in validation for value in values], dtype=np.uint8)


def successful_rollout_coverage(
    rollout_ids: np.ndarray, successes: np.ndarray
) -> tuple[int, int]:
    """Return covered and total approach rollouts for candidate success labels."""
    ids = np.asarray(rollout_ids, dtype=np.int64)
    labels = np.asarray(successes, dtype=bool)
    if ids.ndim != 1 or labels.shape != ids.shape:
        raise ValueError("rollout IDs and success labels must be aligned vectors")
    unique = np.unique(ids)
    covered = sum(bool(np.any(labels[ids == rollout_id])) for rollout_id in unique)
    return covered, int(unique.size)


def _accepted_condition(source: dict[str, Any], condition_index: int) -> dict[str, Any]:
    records = [
        record
        for record in source.get("records", [])
        if int(record["condition_index"]) == condition_index
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher record")
    return records[0]


def load_prototype(
    manifest_path: Path, *, rollout_id: int
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("purpose") != "exact_cpu_per_transition_kick_teacher_labels"
        or not bool(manifest.get("complete"))
    ):
        raise ValueError("prototype manifest is not a complete transition label set")
    npz_path = Path(str(manifest["npz"]))
    if not npz_path.is_file() or sha256_file(npz_path) != manifest.get("npz_sha256"):
        raise ValueError("prototype NPZ is missing or does not match its manifest")
    with np.load(npz_path, allow_pickle=False) as archive:
        if not {"rollout_id", "parameters", "trained_success"} <= set(archive.files):
            raise ValueError("prototype NPZ is missing required arrays")
        matches = np.flatnonzero(archive["rollout_id"] == rollout_id)
        if matches.size != 1:
            raise ValueError("prototype rollout ID must select exactly one label")
        row = int(matches[0])
        if not bool(archive["trained_success"][row]):
            raise ValueError("selected prototype did not pass its exact training state")
        parameters = np.asarray(archive["parameters"][row], dtype=np.float64)
    if parameters.shape != (14,) or not np.isfinite(parameters).all():
        raise ValueError("prototype parameters have an invalid shape or value")
    return parameters, npz_path, manifest


def _worker_initialize(
    spec: dict[str, Any], contract_path: str, parameters: list[float]
) -> None:
    global _WORKER_EVALUATOR, _WORKER_PARAMETERS
    _WORKER_EVALUATOR = KickTeacherEvaluator(
        KickTeacherSpec(**spec),
        contract=load_policy_contract(Path(contract_path)),
    )
    _WORKER_PARAMETERS = np.asarray(parameters, dtype=np.float64)


def _worker_evaluate(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None or _WORKER_PARAMETERS is None:
        raise RuntimeError("switch-window worker was not initialized")
    metrics = _WORKER_EVALUATOR.rollout(
        _WORKER_PARAMETERS,
        initial_qpos=np.asarray(task["qpos"], dtype=np.float64),
        initial_qvel=np.asarray(task["qvel"], dtype=np.float64),
        capture_targets=True,
    )
    observations = _WORKER_EVALUATOR.captured_observations
    if (
        observations.ndim != 2
        or observations.shape[0] < 1
        or observations.shape[1] != _WORKER_EVALUATOR.contract.observation_size
    ):
        raise RuntimeError("failed to capture the exact kick actor observation")
    return {
        "candidate_id": int(task["candidate_id"]),
        "success": bool(kick_trial_success(metrics)),
        "actor_observation": observations[0].astype(float).tolist(),
        "metrics": metrics,
    }


def _entry_arrays(candidates: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    entries: list[KickTransitionEntry] = [candidate["entry"] for candidate in candidates]
    return {
        "qpos": np.stack([entry.qpos for entry in entries]).astype(np.float32),
        "qvel": np.stack([entry.qvel for entry in entries]).astype(np.float32),
        "joint_position_offset": np.stack(
            [entry.joint_position_offset for entry in entries]
        ).astype(np.float32),
        "joint_velocity": np.stack([entry.joint_velocity for entry in entries]).astype(
            np.float32
        ),
        "walk_previous_action": np.stack(
            [entry.walk_previous_action for entry in entries]
        ).astype(np.float32),
        "setup_velocity_command": np.stack(
            [entry.setup_velocity_command for entry in entries]
        ).astype(np.float32),
        "locomotion_phase": np.stack([entry.locomotion_phase for entry in entries]).astype(
            np.float32
        ),
        "support_hint": np.stack([entry.support_hint for entry in entries]).astype(
            np.float32
        ),
        "phase_magnitude_rad": np.asarray(
            [entry.phase_magnitude_rad for entry in entries], dtype=np.float32
        ),
        "ball_position_local_m": np.stack(
            [entry.ball_position_local_m for entry in entries]
        ).astype(np.float32),
        "root_velocity": np.stack([entry.root_velocity for entry in entries]).astype(
            np.float32
        ),
        "torso_height_m": np.asarray(
            [entry.torso_height_m for entry in entries], dtype=np.float32
        ),
        "upright": np.asarray([entry.upright for entry in entries], dtype=np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("prototype_manifest", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--prototype-rollout-id", type=int, default=302)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--initial-x-min", type=float, default=-0.45)
    parser.add_argument("--initial-x-max", type=float, default=-0.05)
    parser.add_argument("--initial-y-min", type=float, default=-0.20)
    parser.add_argument("--initial-y-max", type=float, default=0.20)
    parser.add_argument("--initial-yaw-min-deg", type=float, default=-12.0)
    parser.add_argument("--initial-yaw-max-deg", type=float, default=12.0)
    parser.add_argument("--ball-x-jitter", type=float, default=0.01)
    parser.add_argument("--ball-y-jitter", type=float, default=0.015)
    parser.add_argument("--setup-timeout", type=float, default=5.0)
    parser.add_argument("--setup-tolerance", type=float, default=0.03)
    parser.add_argument("--confirmation-min-cycles", type=int, default=1)
    parser.add_argument("--confirmation-max-cycles", type=int, default=60)
    parser.add_argument("--confirmation-stride-cycles", type=int, default=2)
    parser.add_argument("--phase-buckets", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.rollouts < 2 or args.workers < 1 or args.phase_buckets < 2:
        raise ValueError("rollouts, workers and phase buckets are out of range")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be in (0, 0.5)")
    if not (
        1 <= args.confirmation_min_cycles <= args.confirmation_max_cycles
        and args.confirmation_stride_cycles >= 1
    ):
        raise ValueError("confirmation-cycle range is invalid")
    ranges = np.asarray(
        [
            args.initial_x_min,
            args.initial_x_max,
            args.initial_y_min,
            args.initial_y_max,
            args.initial_yaw_min_deg,
            args.initial_yaw_max_deg,
            args.ball_x_jitter,
            args.ball_y_jitter,
            args.setup_timeout,
            args.setup_tolerance,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(ranges).all():
        raise ValueError("sampling bounds must be finite")
    if not (
        args.initial_x_min < args.initial_x_max
        and args.initial_y_min < args.initial_y_max
        and args.initial_yaw_min_deg < args.initial_yaw_max_deg
        and args.ball_x_jitter >= 0.0
        and args.ball_y_jitter >= 0.0
        and args.setup_timeout > 0.0
        and args.setup_tolerance > 0.0
    ):
        raise ValueError("sampling bounds are invalid")

    npz_path = args.output_prefix.with_suffix(".npz")
    json_path = args.output_prefix.with_suffix(".json")
    if not json_path.is_absolute() or json_path.is_relative_to(Path.cwd()):
        raise ValueError("output prefix must be absolute and outside the repository")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("switch-window corpus outputs already exist")

    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    record = _accepted_condition(teacher, args.condition_index)
    parameters, prototype_npz, prototype_manifest = load_prototype(
        args.prototype_manifest, rollout_id=args.prototype_rollout_id
    )
    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3" or contract.observation_size != 98:
        raise ValueError("switch-window corpus requires kick_policy_v3")
    if prototype_manifest.get("contract_sha256") != sha256_file(args.contract):
        raise ValueError("prototype and requested policy contracts differ")
    if prototype_manifest.get("teacher_manifest_sha256") != sha256_file(
        args.teacher_manifest
    ):
        raise ValueError("prototype and requested teacher manifests differ")

    shared_spec = {
        "target_distance_m": float(record["distance_m"]),
        "target_angle_deg": float(record["angle_deg"]),
        "requested_ball_speed_mps": float(record["requested_speed_mps"]),
        "desired_arrival_speed_mps": float(record["desired_arrival_speed_mps"]),
        "action_mode": str(record["mode"]),
    }
    capture_spec = KickTeacherSpec(
        **shared_spec,
        evaluation_duration_s=args.setup_timeout + 0.02,
    )
    capture_evaluator = KickTeacherEvaluator(capture_spec, contract=contract)
    setup_parameters = np.asarray(record["parameters"], dtype=np.float64)
    base_ball_x = float(record["ball_x_offset_m"])
    base_ball_y = float(record["ball_y_offset_m"])
    confirmation_cycles = range(
        args.confirmation_min_cycles,
        args.confirmation_max_cycles + 1,
        args.confirmation_stride_cycles,
    )
    rng = np.random.default_rng(args.seed)
    candidates: list[dict[str, Any]] = []
    approaches: list[dict[str, Any]] = []

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
        candidate_ids: list[int] = []
        for cycles in confirmation_cycles:
            metrics = capture_evaluator.rollout(
                setup_parameters,
                ball_x_offset_m=ball_x,
                ball_y_offset_m=ball_y,
                setup_ball_x_offset_m=base_ball_x,
                setup_ball_y_offset_m=base_ball_y,
                setup_timeout_s=args.setup_timeout,
                setup_tolerance_m=args.setup_tolerance,
                setup_confirmation_cycles=cycles,
                initial_robot_offset_m=(float(initial_offset[0]), float(initial_offset[1])),
                initial_robot_yaw_deg=initial_yaw,
                capture_transition_entry=True,
                stop_after_transition_capture=True,
            )
            entry = capture_evaluator.captured_transition_entry
            if (
                not bool(metrics["setup_succeeded"])
                or entry is None
                or entry.torso_height_m < 0.45
                or entry.upright < 0.75
            ):
                continue
            candidate_id = len(candidates)
            candidate_ids.append(candidate_id)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "approach_rollout_id": rollout_id,
                    "confirmation_cycles": cycles,
                    "setup_duration_s": float(metrics["setup_duration_s"]),
                    "initial_robot_offset_m": initial_offset.copy(),
                    "initial_robot_yaw_deg": initial_yaw,
                    "ball_offset_m": np.array([ball_x, ball_y], dtype=np.float64),
                    "entry": entry,
                }
            )
        approaches.append(
            {
                "approach_rollout_id": rollout_id,
                "initial_robot_offset_m": initial_offset.tolist(),
                "initial_robot_yaw_deg": initial_yaw,
                "ball_offset_m": [ball_x, ball_y],
                "candidate_count": len(candidate_ids),
                "candidate_ids": candidate_ids,
            }
        )
        print(
            f"captured approach {rollout_id + 1}/{args.rollouts}: "
            f"{len(candidate_ids)} candidates",
            flush=True,
        )

    if len(candidates) < 2 or sum(bool(node["candidate_count"]) for node in approaches) < 2:
        raise RuntimeError("fewer than two approach rollouts produced switch candidates")

    tasks = [
        {
            "candidate_id": candidate["candidate_id"],
            "qpos": candidate["entry"].qpos,
            "qvel": candidate["entry"].qvel,
        }
        for candidate in candidates
    ]
    results: dict[int, dict[str, Any]] = {}
    evaluation_spec = {**shared_spec, "evaluation_duration_s": 3.0}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_initialize,
        initargs=(evaluation_spec, str(args.contract.resolve()), parameters.tolist()),
    ) as pool:
        futures = {pool.submit(_worker_evaluate, task): task for task in tasks}
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[int(result["candidate_id"])] = result
            if completed_count % 25 == 0 or completed_count == len(tasks):
                print(f"evaluated {completed_count}/{len(tasks)} candidates", flush=True)

    if set(results) != set(range(len(candidates))):
        raise RuntimeError("candidate evaluation did not complete")
    entry_arrays = _entry_arrays(candidates)
    rollout_ids = np.asarray(
        [candidate["approach_rollout_id"] for candidate in candidates], dtype=np.int32
    )
    successes = np.asarray(
        [results[index]["success"] for index in range(len(candidates))], dtype=np.uint8
    )
    split = rollout_group_split(
        rollout_ids, seed=args.seed + 1, validation_fraction=args.validation_fraction
    )
    phases = entry_arrays["locomotion_phase"]
    buckets = phase_buckets(phases, args.phase_buckets)
    metrics = [results[index]["metrics"] for index in range(len(candidates))]
    arrays = {
        "candidate_id": np.arange(len(candidates), dtype=np.int32),
        "approach_rollout_id": rollout_ids,
        "confirmation_cycles": np.asarray(
            [candidate["confirmation_cycles"] for candidate in candidates], np.int32
        ),
        "setup_duration_s": np.asarray(
            [candidate["setup_duration_s"] for candidate in candidates], np.float32
        ),
        "initial_robot_offset_m": np.stack(
            [candidate["initial_robot_offset_m"] for candidate in candidates]
        ).astype(np.float32),
        "initial_robot_yaw_deg": np.asarray(
            [candidate["initial_robot_yaw_deg"] for candidate in candidates], np.float32
        ),
        "ball_offset_m": np.stack(
            [candidate["ball_offset_m"] for candidate in candidates]
        ).astype(np.float32),
        **entry_arrays,
        "actor_observation": np.asarray(
            [results[index]["actor_observation"] for index in range(len(candidates))],
            dtype=np.float32,
        ),
        "phase_bucket": buckets,
        "success": successes,
        "fell": np.asarray([node["fell"] for node in metrics], dtype=np.uint8),
        "contact": np.asarray([node["contact"] for node in metrics], dtype=np.uint8),
        "maximum_progress_m": np.asarray(
            [node["maximum_progress_m"] for node in metrics], dtype=np.float32
        ),
        "range_error_m": np.asarray(
            [node["range_error_m"] for node in metrics], dtype=np.float32
        ),
        "lateral_error_m": np.asarray(
            [node["lateral_error_m"] for node in metrics], dtype=np.float32
        ),
        "speed_error_mps": np.asarray(
            [node["speed_error_mps"] for node in metrics], dtype=np.float32
        ),
        "split": split,
    }
    if set(rollout_ids[split == 0].tolist()) & set(rollout_ids[split == 1].tolist()):
        raise RuntimeError("approach rollout leaked across the train/validation split")

    covered, total = successful_rollout_coverage(rollout_ids, successes)
    train_covered, train_total = successful_rollout_coverage(
        rollout_ids[split == 0], successes[split == 0]
    )
    validation_covered, validation_total = successful_rollout_coverage(
        rollout_ids[split == 1], successes[split == 1]
    )
    for approach in approaches:
        ids = np.asarray(approach["candidate_ids"], dtype=np.int64)
        successful_ids = [int(index) for index in ids if bool(successes[index])]
        approach["successful_candidate_ids"] = successful_ids
        approach["successful_confirmation_cycles"] = [
            int(candidates[index]["confirmation_cycles"]) for index in successful_ids
        ]

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "schema_version": 1,
        "purpose": "exact_cpu_walk_to_kick_switch_window_corpus",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "timing labels only; held-out trigger and server gates remain required",
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "teacher_condition_index": args.condition_index,
        "prototype_manifest": str(args.prototype_manifest.resolve()),
        "prototype_manifest_sha256": sha256_file(args.prototype_manifest),
        "prototype_npz": str(prototype_npz.resolve()),
        "prototype_npz_sha256": sha256_file(prototype_npz),
        "prototype_rollout_id": args.prototype_rollout_id,
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "seed": args.seed,
        "requested_approach_rollouts": args.rollouts,
        "candidate_entries": len(candidates),
        "successful_candidates": int(np.count_nonzero(successes)),
        "fallen_candidates": int(np.count_nonzero(arrays["fell"])),
        "approach_rollouts_with_candidates": total,
        "approach_rollouts_with_success_window": covered,
        "approach_success_window_coverage": covered / total,
        "train_rollouts_with_success_window": train_covered,
        "train_rollouts": train_total,
        "validation_rollouts_with_success_window": validation_covered,
        "validation_rollouts": validation_total,
        "split_unit": "whole_approach_rollout",
        "sampling": {
            "initial_x_m": [args.initial_x_min, args.initial_x_max],
            "initial_y_m": [args.initial_y_min, args.initial_y_max],
            "initial_yaw_deg": [args.initial_yaw_min_deg, args.initial_yaw_max_deg],
            "ball_x_jitter_m": args.ball_x_jitter,
            "ball_y_jitter_m": args.ball_y_jitter,
            "setup_timeout_s": args.setup_timeout,
            "setup_tolerance_m": args.setup_tolerance,
            "confirmation_cycles": [
                args.confirmation_min_cycles,
                args.confirmation_max_cycles,
                args.confirmation_stride_cycles,
            ],
        },
        "npz": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "approaches": approaches,
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
