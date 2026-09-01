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
from my3d_rl.soccer_motion_reset import derive_case_seed
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


def _validate_source_dataset_lineage(
    *,
    source_dataset: Path,
    selection: dict[str, Any],
    source_manifest: Path | None,
) -> dict[str, Any]:
    """Bind the source archive either to the teacher selection or prior DAgger."""
    source_sha256 = sha256(source_dataset)
    selection_sha256 = selection.get("combined_dataset", {}).get("sha256")
    if source_sha256 == selection_sha256:
        if source_manifest is not None:
            raise ValueError(
                "--source-manifest is only valid for a prior DAgger aggregate"
            )
        return {
            "kind": "selected_teacher_corpus",
            "manifest": None,
            "manifest_sha256": None,
        }
    if source_manifest is None:
        raise ValueError(
            "source dataset differs from the selected teacher corpus; "
            "a prior DAgger --source-manifest is required"
        )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    allowed_purposes = {
        "k1_b_exact_cpu_soccer_motion_dagger_collection",
        "k1_d_cross_fitted_soccer_motion_dagger_collection",
    }
    if manifest.get("status") != "complete":
        raise ValueError("source DAgger manifest is not complete")
    if manifest.get("purpose") not in allowed_purposes:
        raise ValueError("source manifest is not an approved DAgger collection")
    if manifest.get("output_dataset_sha256") != source_sha256:
        raise ValueError("source DAgger manifest does not bind the source dataset")
    return {
        "kind": "prior_dagger_aggregate",
        "manifest": str(source_manifest.resolve()),
        "manifest_sha256": sha256(source_manifest),
        "purpose": manifest["purpose"],
        "git_revision": manifest.get("git_revision"),
    }


def _episode_validation_split(
    *, seed: int, motion: int, start_frame: int, folds: int, fold_index: int
) -> int:
    """Return one stable episode-level train/validation assignment."""
    if min(seed, motion, start_frame, fold_index) < 0 or folds < 2:
        raise ValueError("episode validation fold settings are invalid")
    if fold_index >= folds:
        raise ValueError("validation fold index must be smaller than fold count")
    return int(derive_case_seed(seed, motion, start_frame) % folds == fold_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument("--phase-samples", type=int, default=32)
    parser.add_argument("--minimum-remaining-frames", type=int, default=10)
    parser.add_argument("--minimum-evaluated-starts", type=int, default=24)
    parser.add_argument("--teacher-beta", type=float, default=0.0)
    parser.add_argument("--state-feedback-horizon", type=int, default=0)
    parser.add_argument(
        "--minimum-state-feedback-improvement", type=float, default=0.0
    )
    parser.add_argument(
        "--maximum-state-feedback-action-delta", type=float, default=1.0
    )
    parser.add_argument("--state-feedback-validation-horizon", type=int, default=0)
    parser.add_argument(
        "--minimum-state-feedback-validation-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument("--state-feedback-validation-seed", type=int)
    parser.add_argument("--validation-joint-noise", type=float, default=0.0)
    parser.add_argument(
        "--validation-joint-velocity-noise", type=float, default=0.0
    )
    parser.add_argument("--reset-perturbation-seed", type=int)
    parser.add_argument("--reset-joint-noise", type=float, default=0.0)
    parser.add_argument("--reset-root-velocity-noise", type=float, default=0.0)
    parser.add_argument("--reset-yaw-range", type=float, default=0.0)
    parser.add_argument(
        "--allow-source-start-reuse-with-perturbation", action="store_true"
    )
    parser.add_argument("--validation-folds", type=int, default=5)
    parser.add_argument("--validation-fold-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260952)
    args = parser.parse_args()
    if (
        args.phase_samples < 1
        or args.minimum_remaining_frames < 2
        or args.minimum_evaluated_starts < 1
        or not 0.0 <= args.teacher_beta <= 1.0
        or args.state_feedback_horizon < 0
        or args.state_feedback_validation_horizon < 0
        or args.minimum_state_feedback_improvement < 0.0
        or args.minimum_state_feedback_validation_improvement < 0.0
        or args.maximum_state_feedback_action_delta <= 0.0
        or min(
            args.validation_joint_noise,
            args.validation_joint_velocity_noise,
            args.reset_joint_noise,
            args.reset_root_velocity_noise,
            args.reset_yaw_range,
        )
        < 0.0
        or args.validation_folds < 2
        or not 0 <= args.validation_fold_index < args.validation_folds
        or args.seed < 0
    ):
        raise ValueError("DAgger collection settings are invalid")
    if args.state_feedback_validation_horizon and (
        not args.state_feedback_horizon
        or args.state_feedback_validation_seed is None
    ):
        raise ValueError(
            "cross-fitted state feedback requires search and a validation seed"
        )
    if not args.state_feedback_validation_horizon and (
        args.minimum_state_feedback_validation_improvement > 0.0
        or args.state_feedback_validation_seed is not None
        or args.validation_joint_noise > 0.0
        or args.validation_joint_velocity_noise > 0.0
    ):
        raise ValueError(
            "validation settings require --state-feedback-validation-horizon"
        )
    if args.state_feedback_validation_seed is not None and (
        args.state_feedback_validation_seed < 0
    ):
        raise ValueError("state-feedback validation seed must be non-negative")
    reset_envelope = (
        args.reset_joint_noise,
        args.reset_root_velocity_noise,
        args.reset_yaw_range,
    )
    if args.reset_perturbation_seed is None and any(
        value > 0.0 for value in reset_envelope
    ):
        raise ValueError("reset noise requires --reset-perturbation-seed")
    if args.reset_perturbation_seed is not None and args.reset_perturbation_seed < 0:
        raise ValueError("reset perturbation seed must be non-negative")
    if args.allow_source_start_reuse_with_perturbation and (
        args.reset_perturbation_seed is None
        or not any(value > 0.0 for value in reset_envelope)
    ):
        raise ValueError(
            "source starts may be reused only under a non-zero reset envelope"
        )
    output_dir = _external_new_directory(args.output_dir)
    revision = _clean_revision()
    contract = load_policy_contract(args.contract)
    profile = get_ppo_profile(args.profile)
    if profile.policy_contract != contract.policy_name:
        raise ValueError("PPO profile and policy contract differ")
    corpus = load_soccer_motion_corpus(args.corpus_root)
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    source_lineage = _validate_source_dataset_lineage(
        source_dataset=args.source_dataset,
        selection=selection,
        source_manifest=args.source_manifest,
    )
    source = _load_source_archive(args.source_dataset)
    load_soccer_motion_teacher_dataset(
        args.source_dataset,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )
    excluded: dict[int, set[int]] = defaultdict(set)
    if not args.allow_source_start_reuse_with_perturbation:
        for motion, start in zip(
            source["motion"].tolist(),
            source["start_frame"].tolist(),
            strict=True,
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
                    state_feedback_horizon=args.state_feedback_horizon,
                    minimum_state_feedback_improvement=(
                        args.minimum_state_feedback_improvement
                    ),
                    maximum_state_feedback_action_delta=(
                        args.maximum_state_feedback_action_delta
                    ),
                    state_feedback_validation_horizon=(
                        args.state_feedback_validation_horizon
                    ),
                    minimum_state_feedback_validation_improvement=(
                        args.minimum_state_feedback_validation_improvement
                    ),
                    state_feedback_validation_base_seed=(
                        args.state_feedback_validation_seed
                    ),
                    validation_joint_noise=args.validation_joint_noise,
                    validation_joint_velocity_noise=(
                        args.validation_joint_velocity_noise
                    ),
                    reset_perturbation_base_seed=args.reset_perturbation_seed,
                    reset_joint_noise=args.reset_joint_noise,
                    reset_root_velocity_noise=args.reset_root_velocity_noise,
                    reset_yaw_range=args.reset_yaw_range,
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
            selected_labels = np.concatenate(
                [
                    node["trajectory"]["teacher_label_selected"]
                    for node in results
                ]
            )
            cost_improvements = np.concatenate(
                [
                    node["trajectory"]["teacher_cost_improvement"]
                    for node in results
                ]
            )
            validation_cost_improvements = np.concatenate(
                [
                    node["trajectory"]["teacher_validation_cost_improvement"]
                    for node in results
                ]
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
                "dagger/selected_label_fraction": float(
                    np.mean(selected_labels)
                ),
                "dagger/selected_student_teacher_mse": (
                    float(
                        np.mean(
                            np.square(
                                students[selected_labels] - labels[selected_labels]
                            )
                        )
                    )
                    if np.any(selected_labels)
                    else 0.0
                ),
                "dagger/selected_mean_cost_improvement": (
                    float(np.mean(cost_improvements[selected_labels]))
                    if np.any(selected_labels)
                    else 0.0
                ),
                "dagger/selected_min_cost_improvement": (
                    float(np.min(cost_improvements[selected_labels]))
                    if np.any(selected_labels)
                    else 0.0
                ),
                "dagger/selected_mean_validation_cost_improvement": (
                    float(
                        np.mean(validation_cost_improvements[selected_labels])
                    )
                    if args.state_feedback_validation_horizon
                    and np.any(selected_labels)
                    else 0.0
                ),
                "dagger/selected_min_validation_cost_improvement": (
                    float(
                        np.min(validation_cost_improvements[selected_labels])
                    )
                    if args.state_feedback_validation_horizon
                    and np.any(selected_labels)
                    else 0.0
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

    selected_captures: list[
        tuple[int, int, dict[str, np.ndarray], dict[str, Any]]
    ] = []
    for motion, start, trajectory, result in captures:
        mask = trajectory["teacher_label_selected"]
        if np.any(mask):
            selected_captures.append(
                (
                    motion,
                    start,
                    {
                        name: trajectory[name][mask]
                        for name in (
                            "observations",
                            "base_actions",
                            "teacher_actions",
                            "teacher_cost_improvement",
                            "teacher_validation_cost_improvement",
                            "teacher_validation_seed",
                            "reference_frames",
                            "qpos",
                        )
                    },
                    result,
                )
            )
    if not selected_captures:
        raise RuntimeError("state-feedback teacher selected no useful labels")
    new_arrays = {
        "observation": np.concatenate(
            [node[2]["observations"] for node in selected_captures], axis=0
        ),
        "base_action": np.concatenate(
            [node[2]["base_actions"] for node in selected_captures], axis=0
        ),
        "teacher_action": np.concatenate(
            [node[2]["teacher_actions"] for node in selected_captures], axis=0
        ),
        "qpos": np.concatenate(
            [node[2]["qpos"] for node in selected_captures], axis=0
        ),
        "split": np.concatenate(
            [
                np.full(
                    node[2]["observations"].shape[0],
                    _episode_validation_split(
                        seed=args.seed,
                        motion=node[0],
                        start_frame=node[1],
                        folds=args.validation_folds,
                        fold_index=args.validation_fold_index,
                    ),
                    dtype=np.int8,
                )
                for node in selected_captures
            ]
        ),
        "motion": np.concatenate(
            [
                np.full(node[2]["observations"].shape[0], node[0], dtype=np.int16)
                for node in selected_captures
            ]
        ),
        "start_frame": np.concatenate(
            [
                np.full(node[2]["observations"].shape[0], node[1], dtype=np.int32)
                for node in selected_captures
            ]
        ),
        "reference_frame": np.concatenate(
            [node[2]["reference_frames"] for node in selected_captures]
        ),
    }
    aggregate = {
        name: np.concatenate((source[name], new_arrays[name]), axis=0)
        for name in source
    }
    source_frame_count = int(source["observation"].shape[0])
    aggregate.update(
        {
            "reset_perturbation_seed": np.concatenate(
                (
                    np.full(source_frame_count, -1, dtype=np.int64),
                    np.concatenate(
                        [
                            np.full(
                                node[2]["observations"].shape[0],
                                (
                                    -1
                                    if node[3]["reset_perturbation_seed"] is None
                                    else node[3]["reset_perturbation_seed"]
                                ),
                                dtype=np.int64,
                            )
                            for node in selected_captures
                        ]
                    ),
                )
            ),
            "teacher_search_cost_improvement": np.concatenate(
                (
                    np.full(source_frame_count, np.nan, dtype=np.float32),
                    np.concatenate(
                        [
                            node[2]["teacher_cost_improvement"]
                            for node in selected_captures
                        ]
                    ),
                )
            ),
            "teacher_validation_cost_improvement": np.concatenate(
                (
                    np.full(source_frame_count, np.nan, dtype=np.float32),
                    np.concatenate(
                        [
                            node[2]["teacher_validation_cost_improvement"]
                            for node in selected_captures
                        ]
                    ),
                )
            ),
            "teacher_validation_seed": np.concatenate(
                (
                    np.full(source_frame_count, -1, dtype=np.int64),
                    np.concatenate(
                        [
                            node[2]["teacher_validation_seed"]
                            for node in selected_captures
                        ]
                    ),
                )
            ),
        }
    )
    dataset_path = output_dir / "dagger-corpus.npz"
    np.savez_compressed(dataset_path, **aggregate)
    new_frame_count = int(new_arrays["observation"].shape[0])
    report = {
        "schema_version": 2,
        "status": "complete",
        "purpose": "k1_d_cross_fitted_soccer_motion_dagger_collection",
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
        "source_lineage": source_lineage,
        "output_dataset": str(dataset_path.resolve()),
        "output_dataset_sha256": sha256(dataset_path),
        "source_frames": int(source["observation"].shape[0]),
        "dagger_frames": new_frame_count,
        "aggregate_frames": int(aggregate["observation"].shape[0]),
        "teacher_beta": args.teacher_beta,
        "state_feedback_horizon": args.state_feedback_horizon,
        "minimum_state_feedback_improvement": (
            args.minimum_state_feedback_improvement
        ),
        "maximum_state_feedback_action_delta": (
            args.maximum_state_feedback_action_delta
        ),
        "state_feedback_validation_horizon": (
            args.state_feedback_validation_horizon
        ),
        "minimum_state_feedback_validation_improvement": (
            args.minimum_state_feedback_validation_improvement
        ),
        "state_feedback_validation_seed": (
            args.state_feedback_validation_seed
        ),
        "validation_joint_noise": args.validation_joint_noise,
        "validation_joint_velocity_noise": (
            args.validation_joint_velocity_noise
        ),
        "reset_perturbation": {
            "base_seed": args.reset_perturbation_seed,
            "joint_noise": args.reset_joint_noise,
            "root_velocity_noise": args.reset_root_velocity_noise,
            "yaw_range": args.reset_yaw_range,
            "generator": "numpy_seedsequence_uint63_v2",
            "numpy_version": np.__version__,
        },
        "source_start_reuse_with_perturbation": (
            args.allow_source_start_reuse_with_perturbation
        ),
        "validation_split": {
            "unit": "episode_motion_start",
            "folds": args.validation_folds,
            "fold_index": args.validation_fold_index,
            "seed": args.seed,
            "generator": "numpy_seedsequence_uint63_v2",
        },
        "teacher_query": (
            "cross-fitted exact-CPU state-feedback action search around the "
            "fixed teacher, accepted only on an independently perturbed "
            "validation rollout at every student-visited observation"
            if args.state_feedback_validation_horizon
            else "short-horizon exact-CPU state-feedback action search around "
            "the fixed teacher evaluated on every student-visited observation"
            if args.state_feedback_horizon
            else "fixed selected teacher-base actor plus motion-specific phase "
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
