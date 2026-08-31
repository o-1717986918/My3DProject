#!/usr/bin/env python3
"""Collect teacher labels on states visited by a kick_policy_v2 learner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("--rollouts-per-condition", type=int, default=3)
    parser.add_argument("--ball-x-jitter", type=float, default=0.01)
    parser.add_argument("--ball-y-jitter", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/win98/rl_runs/kick-dagger/kick-policy-v2-dagger"),
    )
    args = parser.parse_args()
    if args.rollouts_per_condition < 1:
        raise ValueError("--rollouts-per-condition must be positive")
    if args.ball_x_jitter < 0.0 or args.ball_y_jitter < 0.0:
        raise ValueError("jitter bounds must be non-negative")

    with np.load(args.base_dataset) as source:
        base_observations = np.asarray(source["observations"], dtype=np.float32)
        base_actions = np.asarray(source["actions"], dtype=np.float32)
        base_episode_ids = np.asarray(source["episode_ids"], dtype=np.int32)
    teacher_source = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    accepted_records = [
        record for record in teacher_source["records"] if record["accepted"]
    ]
    if not accepted_records:
        raise ValueError("teacher manifest has no accepted conditions")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(args.seed)
    observations = [base_observations]
    actions = [base_actions]
    episode_ids = [base_episode_ids]
    next_episode_id = int(base_episode_ids.max()) + 1
    rollout_records: list[dict[str, object]] = []

    for condition in accepted_records:
        spec = KickTeacherSpec(
            target_distance_m=float(condition["distance_m"]),
            target_angle_deg=float(condition["angle_deg"]),
            requested_ball_speed_mps=float(condition["requested_speed_mps"]),
            desired_arrival_speed_mps=float(condition["desired_arrival_speed_mps"]),
            action_mode=str(condition["mode"]),
        )
        evaluator = KickTeacherEvaluator(spec)
        teacher_parameters = np.asarray(condition["parameters"], dtype=np.float64)
        for rollout_index in range(args.rollouts_per_condition):
            ball_x = float(
                condition["ball_x_offset_m"]
                + rng.uniform(-args.ball_x_jitter, args.ball_x_jitter)
            )
            ball_y = float(
                condition["ball_y_offset_m"]
                + rng.uniform(-args.ball_y_jitter, args.ball_y_jitter)
            )
            times, visited_observations, teacher_actions, metrics = (
                evaluator.dagger_demonstration(
                    teacher_parameters,
                    session,
                    ball_x_offset_m=ball_x,
                    ball_y_offset_m=ball_y,
                )
            )
            observations.append(visited_observations)
            actions.append(teacher_actions)
            episode_ids.append(np.full(times.shape, next_episode_id, dtype=np.int32))
            rollout_records.append(
                {
                    "episode_id": next_episode_id,
                    "condition_index": condition["condition_index"],
                    "rollout_index": rollout_index,
                    "sample_count": int(times.size),
                    "ball_x_offset_m": ball_x,
                    "ball_y_offset_m": ball_y,
                    "learner_metrics": metrics,
                }
            )
            next_episode_id += 1

    output_dataset = args.output_prefix.with_suffix(".npz")
    output_manifest = args.output_prefix.with_suffix(".json")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    combined_observations = np.concatenate(observations)
    combined_actions = np.concatenate(actions)
    combined_episode_ids = np.concatenate(episode_ids)
    np.savez_compressed(
        output_dataset,
        observations=combined_observations,
        actions=combined_actions,
        episode_ids=combined_episode_ids,
    )
    report = {
        "purpose": "kick_policy_v2_dagger_dataset",
        "promotable": False,
        "model": str(args.model),
        "model_sha256": _sha256(args.model),
        "teacher_manifest": str(args.teacher_manifest),
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "base_dataset": str(args.base_dataset),
        "base_dataset_sha256": _sha256(args.base_dataset),
        "output_dataset": str(output_dataset),
        "output_dataset_sha256": _sha256(output_dataset),
        "base_sample_count": int(base_observations.shape[0]),
        "dagger_sample_count": int(
            combined_observations.shape[0] - base_observations.shape[0]
        ),
        "total_sample_count": int(combined_observations.shape[0]),
        "rollout_count": len(rollout_records),
        "rollouts_per_condition": args.rollouts_per_condition,
        "seed": args.seed,
        "validation_episode_suggestion": int(combined_episode_ids.max()),
        "rollouts": rollout_records,
    }
    output_manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
