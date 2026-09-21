#!/usr/bin/env python3
"""Add successful action-bank labels from server-observed kick entries to BC data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import (
    KickTeacherEvaluator, KickTeacherSpec, kick_trial_success,
)
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_motion_state import (
    infer_previous_walk_action, project_server_motion_state,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_successful_teacher(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    """Choose a demonstrated success; never turn a failed trajectory into a label."""
    successful = [
        candidate for candidate in candidates
        if bool(candidate["success"]) and not bool(candidate["fell"])
    ]
    if not successful:
        return None
    return max(successful, key=lambda candidate: float(candidate["metrics"]["score"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("server_corpus", type=Path)
    parser.add_argument("teacher_bank_report", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("--server-weight", type=float, default=12.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    output_npz = args.output_prefix.with_suffix(".npz").resolve()
    output_json = args.output_prefix.with_suffix(".json").resolve()
    inputs = (
        args.model, args.teacher_manifest, args.server_corpus,
        args.teacher_bank_report, args.base_dataset,
    )
    if (
        not all(path.is_file() for path in inputs)
        or not args.output_prefix.is_absolute()
        or output_npz.is_relative_to(REPOSITORY_ROOT.resolve())
        or output_json.is_relative_to(REPOSITORY_ROOT.resolve())
        or output_npz.exists() or output_json.exists()
        or not np.isfinite(args.server_weight) or args.server_weight <= 0.0
    ):
        raise ValueError("inputs must exist and outputs must be new, absolute, and external")

    contract = load_policy_contract(CONTRACT)
    session = ort.InferenceSession(str(args.model.resolve()), providers=["CPUExecutionProvider"])
    if (
        session.get_inputs()[0].shape != list(contract.input_shape)
        or session.get_outputs()[0].shape != list(contract.output_shape)
    ):
        raise ValueError("learner does not match kick_policy_v3")

    corpus_manifest = json.loads(
        (args.server_corpus.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        corpus_manifest.get("schema_version") != 2
        or corpus_manifest.get("archive_sha256") != sha256(args.server_corpus)
    ):
        raise ValueError("server corpus is incomplete or has a hash mismatch")
    with np.load(args.server_corpus, allow_pickle=False) as archive:
        server = {name: np.asarray(archive[name]) for name in archive.files}

    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    records = {
        int(record["condition_index"]): record
        for record in teacher.get("records", []) if bool(record.get("accepted"))
    }
    bank_report = json.loads(args.teacher_bank_report.read_text(encoding="utf-8"))
    if (
        bank_report.get("source_corpus_sha256") != sha256(args.server_corpus)
        or bank_report.get("teacher_manifest_sha256") != sha256(args.teacher_manifest)
        or "teacher_bank" not in bank_report
    ):
        raise ValueError("teacher-bank report is bound to different inputs")
    bank_trials = bank_report["teacher_bank"]["trials"]
    selected = [
        (trial, select_successful_teacher(trial["candidates"]))
        for trial in bank_trials if int(trial["split"]) == 0
    ]
    selected = [(trial, choice) for trial, choice in selected if choice is not None]
    if len(selected) < 2:
        raise ValueError("fewer than two successful training-entry teachers are available")

    condition_record = records[int(selected[0][1]["condition_index"])]
    task_keys = (
        "distance_m", "angle_deg", "requested_speed_mps",
        "desired_arrival_speed_mps", "mode",
    )
    if any(
        any(records[int(choice["condition_index"])][key] != condition_record[key]
            for key in task_keys)
        for _, choice in selected
    ):
        raise ValueError("selected action bank mixes incompatible kick tasks")
    evaluator = KickTeacherEvaluator(KickTeacherSpec(
        target_distance_m=float(condition_record["distance_m"]),
        target_angle_deg=float(condition_record["angle_deg"]),
        requested_ball_speed_mps=float(condition_record["requested_speed_mps"]),
        desired_arrival_speed_mps=float(condition_record["desired_arrival_speed_mps"]),
        action_mode=str(condition_record["mode"]),
    ), contract=contract)
    scene = RcssKickScene(contract)
    trajectories: list[dict[str, object]] = []
    for trial, choice in selected:
        row = int(trial["row"])
        record = records[int(choice["condition_index"])]
        project_server_motion_state(scene, server, row)
        previous_action, previous_valid = infer_previous_walk_action(server, row)
        _, observations, actions, metrics = evaluator.dagger_demonstration(
            np.asarray(record["parameters"], dtype=np.float64), session,
            initial_qpos=scene.data.qpos.copy(), initial_qvel=scene.data.qvel.copy(),
            initial_walk_previous_action=previous_action,
        )
        trajectories.append({
            "row": row,
            "match_id": int(server["match_id"][row]),
            "condition_index": int(choice["condition_index"]),
            "previous_walk_action_reconstructed": previous_valid,
            "observations": observations,
            "actions": actions,
            "learner_metrics": metrics,
        })

    with np.load(args.base_dataset, allow_pickle=False) as archive:
        required = {"observations", "actions", "episode_ids", "split"}
        if not required <= set(archive.files):
            raise ValueError("base dataset lacks observations/actions/episode_ids/split")
        base = {name: np.asarray(archive[name]) for name in required}
        base_weights = (
            np.asarray(archive["sample_weights"], dtype=np.float32)
            if "sample_weights" in archive.files
            else np.ones(base["episode_ids"].shape, dtype=np.float32)
        )
    first_episode = int(np.max(base["episode_ids"])) + 1
    added_episode_ids = np.concatenate([
        np.full(item["observations"].shape[0], first_episode + index, np.int32)
        for index, item in enumerate(trajectories)
    ])
    added_observations = np.concatenate(
        [item["observations"] for item in trajectories]
    ).astype(np.float32)
    added_actions = np.concatenate([item["actions"] for item in trajectories]).astype(np.float32)
    arrays = {
        "observations": np.concatenate([base["observations"], added_observations]),
        "actions": np.concatenate([base["actions"], added_actions]),
        "episode_ids": np.concatenate([base["episode_ids"], added_episode_ids]),
        "split": np.concatenate([
            base["split"], np.zeros(added_episode_ids.shape, dtype=np.uint8)
        ]),
        "sample_weights": np.concatenate([
            base_weights,
            np.full(added_episode_ids.shape, args.server_weight, dtype=np.float32),
        ]),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    report = {
        "schema_version": 1,
        "purpose": "server_observed_kick_entry_dagger_augmentation",
        "promotable": False,
        "promotion_blocker": "retraining and held-out exact physics plus RCSS A/B remain required",
        "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256(args.teacher_manifest),
        "server_corpus": str(args.server_corpus.resolve()),
        "server_corpus_sha256": sha256(args.server_corpus),
        "teacher_bank_report": str(args.teacher_bank_report.resolve()),
        "teacher_bank_report_sha256": sha256(args.teacher_bank_report),
        "base_dataset": str(args.base_dataset.resolve()),
        "base_dataset_sha256": sha256(args.base_dataset),
        "selection": "successful_nonfall_bank_candidate_with_maximum_exact_cpu_score_on_training_matches_only",
        "server_weight": args.server_weight,
        "teacher_episodes": len(trajectories),
        "added_samples": int(added_observations.shape[0]),
        "learner_contacts": sum(bool(item["learner_metrics"]["contact"]) for item in trajectories),
        "learner_falls": sum(bool(item["learner_metrics"]["fell"]) for item in trajectories),
        "learner_successes": sum(
            kick_trial_success(item["learner_metrics"]) for item in trajectories
        ),
        "episodes": [
            {key: item[key] for key in (
                "row", "match_id", "condition_index",
                "previous_walk_action_reconstructed", "learner_metrics",
            )}
            for item in trajectories
        ],
        "dataset": str(output_npz), "dataset_sha256": sha256(output_npz),
    }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "teacher_episodes", "added_samples", "learner_contacts",
        "learner_falls", "learner_successes", "dataset",
    )}, indent=2))


if __name__ == "__main__":
    main()
