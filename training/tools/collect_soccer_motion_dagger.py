#!/usr/bin/env python3
"""Collect exact-CPU phase-teacher labels on states visited by a student."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.soccer_motion_bc import load_soccer_motion_teacher_dataset
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.soccer_motion_dagger import (
    load_selected_teacher_corrections,
    sha256,
)
from my3d_rl.soccer_motion_policy import load_soccer_motion_policy
from my3d_rl.soccer_motion_teacher import SoccerMotionCorrectionEvaluator
from my3d_rl.training_dashboard import TrainingDashboard


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "training/contracts/soccer_motion_policy_v2.yaml"
)


def _clean_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    if status:
        raise RuntimeError("formal DAgger collection requires a clean Git tree")
    return revision


def _external_new_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output directory must stay outside the repository")
    if resolved.exists():
        raise FileExistsError(f"output directory already exists: {resolved}")
    return resolved


def _start_frames(
    length: int,
    samples: int,
    minimum_remaining: int,
    excluded: set[int],
) -> list[int]:
    final_start = max(0, length - minimum_remaining)
    return sorted(
        set(
            int(value)
            for value in np.linspace(0, final_start, samples, dtype=np.int64)
            if value < length - 1 and int(value) not in excluded
        )
    )


def _load_source_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "observation",
            "base_action",
            "teacher_action",
            "qpos",
            "split",
            "motion",
            "start_frame",
            "reference_frame",
        )
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"source dataset lacks {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument("--phase-samples", type=int, default=32)
    parser.add_argument("--minimum-remaining-frames", type=int, default=10)
    parser.add_argument("--minimum-evaluated-starts", type=int, default=24)
    parser.add_argument("--teacher-beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260952)
    args = parser.parse_args()
    if (
        args.phase_samples < 1
        or args.minimum_remaining_frames < 2
        or args.minimum_evaluated_starts < 1
        or not 0.0 <= args.teacher_beta <= 1.0
    ):
        raise ValueError("DAgger collection settings are invalid")
    output_dir = _external_new_directory(args.output_dir)
    revision = _clean_revision()
    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    corpus = load_soccer_motion_corpus(args.corpus_root)
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if selection["combined_dataset"]["sha256"] != sha256(args.source_dataset):
        raise ValueError("source dataset differs from the selected teacher corpus")
    source = _load_source_archive(args.source_dataset)
    load_soccer_motion_teacher_dataset(
        args.source_dataset,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )
    excluded: dict[int, set[int]] = defaultdict(set)
    for motion, start in zip(
        source["motion"].tolist(), source["start_frame"].tolist(), strict=True
    ):
        excluded[int(motion)].add(int(start))
    corrections, teacher_provenance = load_selected_teacher_corrections(
        args.selection_manifest,
        corpus=corpus,
        contract=contract,
        contract_sha256=sha256(args.contract),
    )
    teacher_base_checkpoints = {
        node["teacher_base_checkpoint"] for node in teacher_provenance
    }
    if None in teacher_base_checkpoints or len(teacher_base_checkpoints) != 1:
        raise ValueError("selected teachers do not share one checkpoint base")
    teacher_base_checkpoint = Path(teacher_base_checkpoints.pop())
    teacher_base = load_soccer_motion_policy(
        zero_policy=False,
        checkpoint=teacher_base_checkpoint,
        profile_name=args.profile,
        policy_contract_name=contract.policy_name,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )
    student = load_soccer_motion_policy(
        zero_policy=False,
        checkpoint=args.student_checkpoint,
        profile_name=args.profile,
        policy_contract_name=contract.policy_name,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )
    evaluator = SoccerMotionCorrectionEvaluator(corpus, contract, student)
    rng = np.random.default_rng(args.seed)
    captures: list[tuple[int, int, dict[str, np.ndarray], dict[str, Any]]] = []
    per_motion: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True)
    dashboard = TrainingDashboard(output_dir / "tensorboard")
    started = time.monotonic()
    try:
        for motion, correction in enumerate(corrections):
            starts = _start_frames(
                int(corpus.lengths[motion]),
                args.phase_samples,
                args.minimum_remaining_frames,
                excluded[motion],
            )
            if len(starts) < args.minimum_evaluated_starts:
                raise ValueError(
                    f"motion {motion} retains only {len(starts)} DAgger starts"
                )
            results = []
            for start in starts:
                result = evaluator.rollout(
                    motion,
                    start,
                    correction,
                    capture=True,
                    teacher_execution_probability=args.teacher_beta,
                    teacher_base_policy=teacher_base,
                    rng=rng,
                )
                captures.append((motion, start, result["trajectory"], result))
                results.append(result)
            labels = np.concatenate(
                [node["trajectory"]["teacher_actions"] for node in results]
            )
            students = np.concatenate(
                [node["trajectory"]["base_actions"] for node in results]
            )
            metrics = {
                "dagger/completion_rate": float(
                    np.mean([node["completed"] for node in results])
                ),
                "dagger/mean_survival_fraction": float(
                    np.mean([node["survival_fraction"] for node in results])
                ),
                "dagger/student_teacher_mse": float(
                    np.mean(np.square(students - labels))
                ),
                "dagger/intervention_fraction": float(
                    np.mean(
                        [node["teacher_intervention_fraction"] for node in results]
                    )
                ),
                "dagger/frames": float(labels.shape[0]),
            }
            dashboard.write(motion, metrics)
            per_motion.append(
                {
                    "motion": motion,
                    "relative_path": corpus.relative_paths[motion],
                    "start_frames": starts,
                    "episodes": len(results),
                    "frames": int(labels.shape[0]),
                    **metrics,
                }
            )
            print(json.dumps(per_motion[-1], sort_keys=True), flush=True)
    finally:
        dashboard.close()

    new_arrays = {
        "observation": np.concatenate(
            [node[2]["observations"] for node in captures], axis=0
        ),
        "base_action": np.concatenate(
            [node[2]["base_actions"] for node in captures], axis=0
        ),
        "teacher_action": np.concatenate(
            [node[2]["teacher_actions"] for node in captures], axis=0
        ),
        "qpos": np.concatenate([node[2]["qpos"] for node in captures], axis=0),
        "split": np.zeros(
            sum(node[2]["observations"].shape[0] for node in captures),
            dtype=np.int8,
        ),
        "motion": np.concatenate(
            [
                np.full(node[2]["observations"].shape[0], node[0], dtype=np.int16)
                for node in captures
            ]
        ),
        "start_frame": np.concatenate(
            [
                np.full(node[2]["observations"].shape[0], node[1], dtype=np.int32)
                for node in captures
            ]
        ),
        "reference_frame": np.concatenate(
            [
                np.arange(
                    node[1],
                    node[1] + node[2]["observations"].shape[0],
                    dtype=np.int32,
                )
                for node in captures
            ]
        ),
    }
    aggregate = {
        name: np.concatenate((source[name], new_arrays[name]), axis=0)
        for name in source
    }
    dataset_path = output_dir / "dagger-corpus.npz"
    np.savez_compressed(dataset_path, **aggregate)
    new_frame_count = int(new_arrays["observation"].shape[0])
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "k1_b_exact_cpu_soccer_motion_dagger_collection",
        "promotable": False,
        "promotion_blocker": "requires post-DAgger blind evaluation and three seeds",
        "git_revision": revision,
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "corpus_root": str(args.corpus_root.resolve()),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256(args.contract),
        "profile": args.profile,
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "teacher_base_checkpoint": str(teacher_base_checkpoint.resolve()),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "source_dataset": str(args.source_dataset.resolve()),
        "source_dataset_sha256": sha256(args.source_dataset),
        "output_dataset": str(dataset_path.resolve()),
        "output_dataset_sha256": sha256(dataset_path),
        "source_frames": int(source["observation"].shape[0]),
        "dagger_frames": new_frame_count,
        "aggregate_frames": int(aggregate["observation"].shape[0]),
        "teacher_beta": args.teacher_beta,
        "teacher_query": (
            "fixed selected teacher-base actor plus motion-specific phase "
            "correction evaluated on every student-visited observation"
        ),
        "phase_samples": args.phase_samples,
        "minimum_remaining_frames": args.minimum_remaining_frames,
        "minimum_evaluated_starts": args.minimum_evaluated_starts,
        "seed": args.seed,
        "episode_count": len(captures),
        "completion_rate": float(
            np.mean([node[3]["completed"] for node in captures])
        ),
        "mean_survival_fraction": float(
            np.mean([node[3]["survival_fraction"] for node in captures])
        ),
        "per_motion": per_motion,
        "teacher_provenance": teacher_provenance,
        "visualization": {
            "format": "tensorboard_event",
            "log_dir": str((output_dir / "tensorboard").resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
