#!/usr/bin/env python3
"""Generate a labeled multi-condition kick-teacher dataset."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_env import DEFAULT_CONTRACT
from my3d_rl.kick_teacher import (
    DEFAULT_WALK_POLICY,
    KickTeacherEvaluator,
    KickTeacherSpec,
    kick_trial_success,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=float, nargs="+", default=[2.0])
    parser.add_argument("--angles", type=float, nargs="+", default=[-15.0, 0.0, 15.0])
    parser.add_argument("--ball-x", type=float, nargs="+", default=[0.0])
    parser.add_argument("--ball-y", type=float, nargs="+", default=[-0.06, 0.0, 0.06])
    parser.add_argument("--arrival-speed", type=float, default=0.8)
    parser.add_argument(
        "--requested-speed",
        type=float,
        help="fixed launch-speed request; default derives it from distance",
    )
    parser.add_argument("--mode", choices=("pass", "shot", "clear"), default="pass")
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--robust-samples", type=int, default=1)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument(
        "--initial-manifest",
        type=Path,
        nargs="+",
        help="reuse matching per-condition CEM parameters from earlier runs",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/win98/rl_runs/kick-teacher/kick-v2-dataset"),
    )
    args = parser.parse_args()

    conditions = list(
        itertools.product(args.distances, args.angles, args.ball_x, args.ball_y)
    )
    if not conditions:
        raise ValueError("at least one teacher condition is required")
    if args.arrival_speed < 0.0:
        raise ValueError("--arrival-speed must be non-negative")
    if args.restarts < 1:
        raise ValueError("--restarts must be positive")

    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []
    all_condition_vectors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    warm_start: np.ndarray | None = None
    mode_index = {"pass": 0, "shot": 1, "clear": 2}[args.mode]
    initial_parameters_by_condition: dict[
        tuple[float, float, float, float, str], np.ndarray
    ] = {}
    if args.initial_manifest is not None:
        for initial_manifest in args.initial_manifest:
            initial_source = json.loads(initial_manifest.read_text(encoding="utf-8"))
            for record in initial_source.get("records", []):
                if not bool(record.get("accepted", False)):
                    continue
                key = (
                    float(record["distance_m"]),
                    float(record["angle_deg"]),
                    float(record["ball_x_offset_m"]),
                    float(record["ball_y_offset_m"]),
                    str(record["mode"]),
                )
                initial_parameters_by_condition[key] = np.asarray(
                    record["parameters"], dtype=np.float64
                )

    for condition_index, (distance, angle, ball_x, ball_y) in enumerate(conditions):
        requested_speed = (
            float(args.requested_speed)
            if args.requested_speed is not None
            else float(np.sqrt(args.arrival_speed**2 + 2.0 * 0.08 * distance))
        )
        spec = KickTeacherSpec(
            target_distance_m=distance,
            target_angle_deg=angle,
            requested_ball_speed_mps=requested_speed,
            desired_arrival_speed_mps=args.arrival_speed,
            action_mode=args.mode,
        )
        evaluator = KickTeacherEvaluator(spec)
        condition_key = (distance, angle, ball_x, ball_y, args.mode)
        condition_initial = initial_parameters_by_condition.get(condition_key)
        restart_results = []
        for restart in range(args.restarts):
            condition_seed = args.seed + condition_index * args.restarts + restart
            restart_results.append(
                evaluator.optimize(
                    seed=condition_seed,
                    population=args.population,
                    generations=args.generations,
                    robust_samples=args.robust_samples,
                    ball_x_offset_m=ball_x,
                    ball_y_offset_m=ball_y,
                    initial_parameters=(
                        condition_initial
                        if restart == 0 and condition_initial is not None
                        else warm_start if restart == 0 else None
                    ),
                )
            )
        result = max(restart_results, key=lambda candidate: candidate.score)
        condition_seed = (
            args.seed
            + condition_index * args.restarts
            + int(np.argmax([candidate.score for candidate in restart_results]))
        )
        times, observations, actions, targets, metrics = evaluator.demonstration(
            result.parameters,
            ball_x_offset_m=ball_x,
            ball_y_offset_m=ball_y,
        )
        accepted = kick_trial_success(metrics)
        if accepted:
            warm_start = result.parameters.copy()
            all_observations.append(observations)
            all_actions.append(actions)
            all_targets.append(targets)
            all_times.append(times)
            all_episode_ids.append(
                np.full(times.shape, condition_index, dtype=np.int32)
            )
            condition_vector = np.array(
                [
                    distance,
                    angle,
                    requested_speed,
                    args.arrival_speed,
                    ball_x,
                    ball_y,
                    float(mode_index),
                ],
                dtype=np.float32,
            )
            all_condition_vectors.append(
                np.repeat(condition_vector[None, :], times.size, axis=0)
            )
        records.append(
            {
                "condition_index": condition_index,
                "seed": condition_seed,
                "distance_m": distance,
                "angle_deg": angle,
                "requested_speed_mps": requested_speed,
                "desired_arrival_speed_mps": args.arrival_speed,
                "ball_x_offset_m": ball_x,
                "ball_y_offset_m": ball_y,
                "mode": args.mode,
                "accepted": accepted,
                "score": result.score,
                "parameters": result.parameters.tolist(),
                "metrics": metrics,
                "history": list(result.history),
                "restart_scores": [candidate.score for candidate in restart_results],
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_prefix.with_suffix(".npz")
    manifest_path = args.output_prefix.with_suffix(".json")
    accepted_count = len(all_observations)
    if accepted_count:
        np.savez_compressed(
            dataset_path,
            observations=np.concatenate(all_observations),
            actions=np.concatenate(all_actions),
            joint_targets_rad=np.concatenate(all_targets),
            times_s=np.concatenate(all_times),
            episode_ids=np.concatenate(all_episode_ids),
            conditions=np.concatenate(all_condition_vectors),
        )
        dataset_sha256: str | None = _sha256(dataset_path)
        sample_count = int(sum(array.shape[0] for array in all_observations))
    else:
        dataset_sha256 = None
        sample_count = 0

    manifest = {
        "purpose": "kick_v2_conditioned_teacher_dataset",
        "promotable": False,
        "promotion_blocker": "requires supervised fit, held-out physics and server gates",
        "dataset": str(dataset_path) if accepted_count else None,
        "dataset_sha256": dataset_sha256,
        "contract": str(DEFAULT_CONTRACT),
        "contract_sha256": _sha256(DEFAULT_CONTRACT),
        "base_walk_policy": str(DEFAULT_WALK_POLICY),
        "base_walk_policy_sha256": _sha256(DEFAULT_WALK_POLICY),
        "condition_count": len(conditions),
        "accepted_condition_count": accepted_count,
        "sample_count": sample_count,
        "complete": accepted_count == len(conditions),
        "population": args.population,
        "generations": args.generations,
        "robust_samples": args.robust_samples,
        "robust_aggregation": "0.35_mean_plus_0.65_minimum",
        "restarts": args.restarts,
        "seed": args.seed,
        "initial_manifests": (
            [str(path) for path in args.initial_manifest]
            if args.initial_manifest is not None
            else []
        ),
        "initial_manifest_sha256": (
            [_sha256(path) for path in args.initial_manifest]
            if args.initial_manifest is not None
            else []
        ),
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
