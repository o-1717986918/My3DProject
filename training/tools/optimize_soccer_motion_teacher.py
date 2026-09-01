#!/usr/bin/env python3
"""Fit and evaluate a low-dimensional phase teacher in exact CPU MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import cem_optimize
from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus
from my3d_rl.soccer_motion_policy import load_soccer_motion_policy
from my3d_rl.soccer_motion_teacher import (
    TEACHER_JOINT_CANDIDATES,
    SoccerMotionCorrectionEvaluator,
    decode_phase_correction,
    robust_teacher_objective,
)
from my3d_rl.training_dashboard import TrainingDashboard


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "training" / "contracts" / "soccer_motion_policy_v2.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_provenance() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    return {
        "revision": revision,
        "dirty": bool(status),
        "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def parse_frame_list(value: str) -> list[int]:
    frames = [int(node.strip()) for node in value.split(",") if node.strip()]
    if not frames or len(frames) != len(set(frames)) or min(frames) < 0:
        raise ValueError("frame list must contain unique non-negative integers")
    return frames


def fixed_phase_starts(
    length: int, *, samples: int, minimum_remaining_frames: int
) -> list[int]:
    if samples < 2 or minimum_remaining_frames < 2:
        raise ValueError("phase grid is too small")
    final_start = max(0, length - minimum_remaining_frames)
    return sorted(
        set(
            int(value)
            for value in np.linspace(0, final_start, samples, dtype=np.int64)
            if value < length - 1
        )
    )


def select_active_joints(
    baseline_results: list[dict[str, Any]],
    *,
    joint_order: tuple[str, ...],
    maximum_count: int,
) -> tuple[list[int], list[str]]:
    """Select high-error lower-body joints without consulting validation."""
    if maximum_count < 1:
        raise ValueError("active-joint count must be positive")
    candidates = [joint_order.index(name) for name in TEACHER_JOINT_CANDIDATES]
    errors = np.mean(
        np.asarray(
            [result["mean_joint_abs_error_by_joint"] for result in baseline_results],
            dtype=np.float64,
        ),
        axis=0,
    )
    order = sorted(candidates, key=lambda index: (-errors[index], index))
    selected = sorted(order[: min(maximum_count, len(order))])
    return selected, [joint_order[index] for index in selected]


def paired_summary(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired evaluation needs equal non-empty result lists")
    for first, second in zip(baseline, candidate, strict=True):
        if first["start_frame"] != second["start_frame"]:
            raise ValueError("paired results use different start frames")
    base_survival = np.asarray([node["survival_fraction"] for node in baseline])
    candidate_survival = np.asarray(
        [node["survival_fraction"] for node in candidate]
    )
    base_score = np.asarray([node["teacher_score"] for node in baseline])
    candidate_score = np.asarray([node["teacher_score"] for node in candidate])
    return {
        "trials": len(baseline),
        "baseline_completions": int(sum(node["completed"] for node in baseline)),
        "candidate_completions": int(sum(node["completed"] for node in candidate)),
        "candidate_only_completions": int(
            sum(
                second["completed"] and not first["completed"]
                for first, second in zip(baseline, candidate, strict=True)
            )
        ),
        "baseline_only_completions": int(
            sum(
                first["completed"] and not second["completed"]
                for first, second in zip(baseline, candidate, strict=True)
            )
        ),
        "baseline_mean_survival_fraction": float(np.mean(base_survival)),
        "candidate_mean_survival_fraction": float(np.mean(candidate_survival)),
        "mean_survival_fraction_delta": float(
            np.mean(candidate_survival - base_survival)
        ),
        "baseline_mean_teacher_score": float(np.mean(base_score)),
        "candidate_mean_teacher_score": float(np.mean(candidate_score)),
        "mean_teacher_score_delta": float(np.mean(candidate_score - base_score)),
        "survival_improvements": int(np.sum(candidate_survival > base_survival)),
        "survival_regressions": int(np.sum(candidate_survival < base_survival)),
    }


def _without_trajectory(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "trajectory"}


def _check_external_new_path(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"{label} must stay outside the repository")
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--phase-samples", type=int, default=8)
    parser.add_argument("--minimum-remaining-frames", type=int, default=10)
    parser.add_argument("--train-start-frames")
    parser.add_argument("--validation-start-frames")
    parser.add_argument("--active-joint-count", type=int, default=8)
    parser.add_argument("--knot-count", type=int, default=5)
    parser.add_argument("--maximum-abs-correction", type=float, default=0.35)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--minimum-weight", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--tensorboard-log-dir", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow smoke/debug runs from an uncommitted tree; formal evidence refuses it",
    )
    args = parser.parse_args()
    _check_external_new_path(args.output, "output")
    if args.dataset_output:
        _check_external_new_path(args.dataset_output, "dataset output")
    if args.tensorboard_log_dir:
        _check_external_new_path(args.tensorboard_log_dir, "TensorBoard log directory")
    if args.zero_policy == (args.checkpoint is not None):
        raise ValueError("select exactly one of --zero-policy or --checkpoint")
    if args.population < 2 or args.generations < 1 or args.knot_count < 2:
        raise ValueError("optimizer dimensions are invalid")
    if not 0.0 < args.maximum_abs_correction <= 1.0:
        raise ValueError("maximum correction must lie in (0, 1]")
    git_provenance = _git_provenance()
    if git_provenance["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "formal teacher evidence requires a clean Git tree; commit or use "
            "--allow-dirty only for smoke/debug runs"
        )

    contract = load_policy_contract(args.contract)
    corpus = load_soccer_motion_corpus(args.corpus_root)
    try:
        motion = list(corpus.relative_paths).index(args.relative_path)
    except ValueError as error:
        raise ValueError("relative path is absent from the corpus") from error
    length = int(corpus.lengths[motion])
    all_starts = fixed_phase_starts(
        length,
        samples=args.phase_samples,
        minimum_remaining_frames=args.minimum_remaining_frames,
    )
    if (args.train_start_frames is None) != (args.validation_start_frames is None):
        raise ValueError("provide both train and validation frame lists")
    if args.train_start_frames:
        train_starts = parse_frame_list(args.train_start_frames)
        validation_starts = parse_frame_list(args.validation_start_frames)
    else:
        train_starts = all_starts[::2]
        validation_starts = all_starts[1::2]
    if (
        set(train_starts) & set(validation_starts)
        or not validation_starts
        or any(frame >= length - 1 for frame in train_starts + validation_starts)
    ):
        raise ValueError("train/validation starts must be disjoint valid frames")

    policy = load_soccer_motion_policy(
        zero_policy=args.zero_policy,
        checkpoint=args.checkpoint,
        profile_name=args.profile,
        policy_contract_name=contract.policy_name,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
    )
    evaluator = SoccerMotionCorrectionEvaluator(corpus, contract, policy)
    baseline_train = [evaluator.rollout(motion, frame) for frame in train_starts]
    baseline_validation = [
        evaluator.rollout(motion, frame) for frame in validation_starts
    ]
    active_indices, active_names = select_active_joints(
        baseline_train,
        joint_order=contract.joint_order,
        maximum_count=args.active_joint_count,
    )
    parameter_count = args.knot_count * len(active_indices)
    phases = np.linspace(0.0, 1.0, length)

    def correction(parameters: np.ndarray) -> np.ndarray:
        return decode_phase_correction(
            parameters,
            phases=phases,
            action_size=contract.action_size,
            joint_indices=active_indices,
            knot_count=args.knot_count,
            maximum_abs_correction=args.maximum_abs_correction,
        )

    def objective(parameters: np.ndarray) -> float:
        residual = correction(parameters)
        results = [evaluator.rollout(motion, frame, residual) for frame in train_starts]
        return robust_teacher_objective(
            np.asarray([result["teacher_score"] for result in results]),
            minimum_weight=args.minimum_weight,
        )

    started = time.monotonic()
    dashboard = (
        TrainingDashboard(args.tensorboard_log_dir)
        if args.tensorboard_log_dir
        else None
    )

    def progress(generation: int, metrics: dict[str, float]) -> None:
        if dashboard is not None:
            dashboard.write(
                generation,
                {f"teacher/{name}": value for name, value in metrics.items()},
            )

    try:
        result = cem_optimize(
            objective,
            initial_mean=np.zeros(parameter_count),
            initial_std=np.full(
                parameter_count, 0.45 * args.maximum_abs_correction
            ),
            lower=np.full(parameter_count, -args.maximum_abs_correction),
            upper=np.full(parameter_count, args.maximum_abs_correction),
            seed=args.seed,
            population=args.population,
            generations=args.generations,
            elite_fraction=0.2,
            smoothing=0.35,
            progress=progress,
        )
    finally:
        if dashboard is not None:
            dashboard.close()
    trained_correction = correction(result.parameters)
    candidate_train = [
        evaluator.rollout(motion, frame, trained_correction) for frame in train_starts
    ]
    candidate_validation = [
        evaluator.rollout(motion, frame, trained_correction)
        for frame in validation_starts
    ]
    train_summary = paired_summary(baseline_train, candidate_train)
    validation_summary = paired_summary(
        baseline_validation, candidate_validation
    )
    gate_checks = {
        "training_score_improved": train_summary["mean_teacher_score_delta"] > 0.0,
        "validation_score_improved": (
            validation_summary["mean_teacher_score_delta"] > 0.0
        ),
        "validation_survival_delta_at_least_0_02": (
            validation_summary["mean_survival_fraction_delta"] >= 0.02
        ),
        "no_validation_completion_regression": (
            validation_summary["baseline_only_completions"] == 0
        ),
    }
    teacher_gate_passed = all(gate_checks.values())

    dataset_sha256 = None
    if args.dataset_output:
        captured: list[tuple[int, int, dict[str, np.ndarray]]] = []
        for split, starts in ((0, train_starts), (1, validation_starts)):
            for start in starts:
                node = evaluator.rollout(
                    motion, start, trained_correction, capture=True
                )
                captured.append((split, start, node["trajectory"]))
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset_output,
            observation=np.concatenate(
                [node["observations"] for _, _, node in captured], axis=0
            ),
            base_action=np.concatenate(
                [node["base_actions"] for _, _, node in captured], axis=0
            ),
            teacher_action=np.concatenate(
                [node["teacher_actions"] for _, _, node in captured], axis=0
            ),
            qpos=np.concatenate(
                [node["qpos"] for _, _, node in captured], axis=0
            ),
            split=np.concatenate(
                [
                    np.full(node["observations"].shape[0], split, dtype=np.int8)
                    for split, _, node in captured
                ]
            ),
            motion=np.full(
                sum(node["observations"].shape[0] for _, _, node in captured),
                motion,
                dtype=np.int16,
            ),
            start_frame=np.concatenate(
                [
                    np.full(node["observations"].shape[0], start, dtype=np.int32)
                    for _, start, node in captured
                ]
            ),
            reference_frame=np.concatenate(
                [
                    np.arange(start, start + node["observations"].shape[0], dtype=np.int32)
                    for _, start, node in captured
                ]
            ),
        )
        dataset_sha256 = _sha256(args.dataset_output)

    payload = {
        "schema_version": 1,
        "purpose": "k1_a_exact_cpu_phase_correction_teacher",
        "status": "complete",
        "promotable": False,
        "promotion_blocker": "single-motion teacher precedes corpus BC and three-seed gates",
        "k1_a_teacher_gate_passed": teacher_gate_passed,
        "gate_checks": gate_checks,
        "git_revision": git_provenance["revision"],
        "git_provenance": git_provenance,
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "corpus_root": str(args.corpus_root.resolve()),
        "relative_path": args.relative_path,
        "motion_sha256": corpus.sha256[motion],
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "base_policy": "zero_residual" if args.zero_policy else "checkpoint",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "profile": args.profile,
        "train_start_frames": train_starts,
        "validation_start_frames": validation_starts,
        "active_joint_indices": active_indices,
        "active_joint_names": active_names,
        "knot_count": args.knot_count,
        "maximum_abs_correction": args.maximum_abs_correction,
        "parameter_count": parameter_count,
        "parameters": result.parameters.tolist(),
        "objective": result.score,
        "history": list(result.history),
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "minimum_weight": args.minimum_weight,
        "elapsed_seconds": time.monotonic() - started,
        "visualization": (
            {
                "format": "tensorboard_event",
                "log_dir": str(args.tensorboard_log_dir.resolve()),
                "launch": (
                    "tensorboard --logdir "
                    f"{args.tensorboard_log_dir.resolve()} --port 6006"
                ),
            }
            if args.tensorboard_log_dir
            else None
        ),
        "train": {
            "summary": train_summary,
            "baseline": [_without_trajectory(node) for node in baseline_train],
            "candidate": [_without_trajectory(node) for node in candidate_train],
        },
        "validation": {
            "summary": validation_summary,
            "baseline": [
                _without_trajectory(node) for node in baseline_validation
            ],
            "candidate": [
                _without_trajectory(node) for node in candidate_validation
            ],
        },
        "dataset": str(args.dataset_output.resolve()) if args.dataset_output else None,
        "dataset_sha256": dataset_sha256,
        "external_method_provenance": {
            "paid_commit": "e72e470230047dedaf66df0983f1d0ab746faeb5",
            "paid_adaptive_sampling": "motion-by-one-second-bin EMA, smoothing, uniform coverage",
            "t1_dagger_commit": "378a12ac7446cd175f973c04e32912eb9acbee10",
            "implementation": "independent exact-MuJoCo bounded phase CEM teacher",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
