#!/usr/bin/env python3
"""Optimize a phase-indexed kick teacher directly in exact CPU MuJoCo."""

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
from my3d_rl.kick_teacher import (
    PARAMETER_LOWER,
    PARAMETER_NAMES,
    PARAMETER_UPPER,
    KickTeacherEvaluator,
    KickTeacherSpec,
    cem_optimize,
    kick_trial_success,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def representative_indices(
    rollout_ids: np.ndarray, maximum_count: int
) -> np.ndarray:
    """Select deterministic, well-spaced rows without duplicating endpoints."""
    ids = np.asarray(rollout_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("rollout IDs must be a non-empty vector")
    if maximum_count < 1:
        raise ValueError("maximum count must be positive")
    order = np.argsort(ids, kind="stable")
    if order.size <= maximum_count:
        return order
    positions = np.rint(np.linspace(0, order.size - 1, maximum_count)).astype(int)
    return order[positions]


def robust_phase_objective(
    scores: np.ndarray,
    successes: np.ndarray,
    *,
    success_bonus: float,
    minimum_weight: float,
) -> float:
    """Combine typical and worst-case exact-physics behavior for CEM."""
    score = np.asarray(scores, dtype=np.float64)
    success = np.asarray(successes, dtype=np.float64)
    if score.ndim != 1 or success.shape != score.shape or score.size == 0:
        raise ValueError("scores and successes must be equal non-empty vectors")
    if not np.isfinite(score).all() or not np.isfinite(success).all():
        return -1.0e9
    if success_bonus < 0.0 or not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("objective weights are invalid")
    shaped = score + success_bonus * success
    return float(
        (1.0 - minimum_weight) * np.mean(shaped)
        + minimum_weight * np.min(shaped)
    )


def _load_inputs(
    teacher_manifest: Path,
    corpus: Path,
    condition_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    teacher = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in teacher["records"]
        if int(record["condition_index"]) == condition_index
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher record")
    manifest_path = corpus.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "kick_policy_v3_walk_to_kick_transition_corpus":
        raise ValueError("transition corpus has the wrong purpose")
    if manifest.get("npz_sha256") != _sha256(corpus):
        raise ValueError("transition corpus hash mismatch")
    if int(manifest["teacher_condition_index"]) != condition_index:
        raise ValueError("teacher condition does not match transition corpus")
    with np.load(corpus, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "split", "rollout_id", "phase_bucket"}
        if not required <= set(archive.files):
            raise ValueError("transition corpus is missing required arrays")
        arrays = {name: np.asarray(archive[name]) for name in required}
    row_count = arrays["qpos"].shape[0]
    if (
        arrays["qpos"].ndim != 2
        or arrays["qvel"].ndim != 2
        or any(arrays[name].shape != (row_count,) for name in required - {"qpos", "qvel"})
        or not np.isfinite(arrays["qpos"]).all()
        or not np.isfinite(arrays["qvel"]).all()
        or not set(arrays["split"].tolist()) <= {0, 1}
    ):
        raise ValueError("transition corpus arrays are invalid")
    return records[0], arrays, manifest


def _evaluate_rows(
    evaluator: KickTeacherEvaluator,
    parameters: np.ndarray,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        metrics = evaluator.rollout(
            parameters,
            initial_qpos=arrays["qpos"][row],
            initial_qvel=arrays["qvel"][row],
        )
        results.append(
            {
                "rollout_id": int(arrays["rollout_id"][row]),
                "phase_bucket": int(arrays["phase_bucket"][row]),
                "success": bool(kick_trial_success(metrics)),
                "metrics": metrics,
            }
        )
    return results


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"trials": 0, "successes": 0, "success_rate": 0.0, "falls": 0}
    return {
        "trials": len(results),
        "successes": sum(bool(row["success"]) for row in results),
        "success_rate": float(np.mean([bool(row["success"]) for row in results])),
        "falls": sum(bool(row["metrics"]["fell"]) for row in results),
        "contacts": sum(bool(row["metrics"]["contact"]) for row in results),
        "mean_score": float(np.mean([row["metrics"]["score"] for row in results])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seed", type=int, default=7302)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--maximum-optimization-states", type=int, default=3)
    parser.add_argument("--success-bonus", type=float, default=50.0)
    parser.add_argument("--minimum-weight", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be an absolute path outside the repository")
    if args.output.exists() and not args.resume:
        raise FileExistsError("output already exists; use --resume explicitly")
    if args.population < 2 or args.generations < 1:
        raise ValueError("population and generations are invalid")

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3":
        raise ValueError("phase-indexed training requires kick_policy_v3")
    record, arrays, corpus_manifest = _load_inputs(
        args.teacher_manifest, args.transition_corpus, args.condition_index
    )
    spec = KickTeacherSpec(
        target_distance_m=float(record["distance_m"]),
        target_angle_deg=float(record["angle_deg"]),
        requested_ball_speed_mps=float(record["requested_speed_mps"]),
        desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
        action_mode=str(record["mode"]),
        evaluation_duration_s=3.0,
    )
    evaluator = KickTeacherEvaluator(spec, contract=contract)
    initial_parameters = np.asarray(record["parameters"], dtype=np.float64)
    completed: dict[int, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        expected = {
            "teacher_manifest_sha256": _sha256(args.teacher_manifest),
            "transition_corpus_sha256": _sha256(args.transition_corpus),
            "contract_sha256": _sha256(args.contract),
            "seed": args.seed,
            "population": args.population,
            "generations": args.generations,
            "maximum_optimization_states": args.maximum_optimization_states,
        }
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("resume manifest does not match the requested run")
        completed = {int(node["phase_bucket"]): node for node in previous["nodes"]}

    phase_buckets = sorted(set(int(value) for value in arrays["phase_bucket"]))
    started = time.time()
    nodes = completed

    def render(status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "purpose": "exact_cpu_phase_indexed_kick_teacher_training",
            "status": status,
            "complete": status == "completed",
            "promotable": False,
            "promotion_blocker": "requires held-out, three-seed and server gates",
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "teacher_manifest": str(args.teacher_manifest.resolve()),
            "teacher_manifest_sha256": _sha256(args.teacher_manifest),
            "transition_corpus": str(args.transition_corpus.resolve()),
            "transition_corpus_sha256": _sha256(args.transition_corpus),
            "transition_corpus_manifest_sha256": _sha256(
                args.transition_corpus.with_suffix(".json")
            ),
            "contract": str(args.contract.resolve()),
            "contract_sha256": _sha256(args.contract),
            "condition_index": args.condition_index,
            "phase_bucket_count": int(corpus_manifest["phase_bucket_count"]),
            "parameter_names": list(PARAMETER_NAMES),
            "initial_parameters": initial_parameters.tolist(),
            "seed": args.seed,
            "population": args.population,
            "generations": args.generations,
            "maximum_optimization_states": args.maximum_optimization_states,
            "success_bonus": args.success_bonus,
            "minimum_weight": args.minimum_weight,
            "elapsed_seconds": time.time() - started,
            "nodes": [nodes[bucket] for bucket in sorted(nodes)],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for bucket in phase_buckets:
        if bucket in completed:
            continue
        train_rows = np.flatnonzero(
            (arrays["split"] == 0) & (arrays["phase_bucket"] == bucket)
        )
        if train_rows.size == 0:
            raise ValueError(f"phase bucket {bucket} has no training entries")
        local = representative_indices(
            arrays["rollout_id"][train_rows], args.maximum_optimization_states
        )
        optimization_rows = train_rows[local]

        def objective(parameters: np.ndarray) -> float:
            evaluations = _evaluate_rows(
                evaluator, parameters, arrays, optimization_rows
            )
            return robust_phase_objective(
                np.asarray([row["metrics"]["score"] for row in evaluations]),
                np.asarray([row["success"] for row in evaluations]),
                success_bonus=args.success_bonus,
                minimum_weight=args.minimum_weight,
            )

        result = cem_optimize(
            objective,
            initial_mean=initial_parameters,
            initial_std=np.maximum(
                0.12 * (PARAMETER_UPPER - PARAMETER_LOWER), 0.03
            ),
            lower=PARAMETER_LOWER,
            upper=PARAMETER_UPPER,
            seed=args.seed + bucket,
            population=args.population,
            generations=args.generations,
            elite_fraction=0.25,
            smoothing=0.35,
        )
        nodes[bucket] = {
            "phase_bucket": bucket,
            "optimization_rollout_ids": arrays["rollout_id"][
                optimization_rows
            ].astype(int).tolist(),
            "training_entry_count": int(train_rows.size),
            "parameters": result.parameters.tolist(),
            "objective": result.score,
            "history": list(result.history),
        }
        args.output.write_text(
            json.dumps(render("running"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "phase_bucket": bucket,
                    "objective": result.score,
                    "completed_buckets": len(nodes),
                }
            ),
            flush=True,
        )

    baseline_parameters = {
        bucket: initial_parameters for bucket in phase_buckets
    }
    trained_parameters = {
        bucket: np.asarray(nodes[bucket]["parameters"], dtype=np.float64)
        for bucket in phase_buckets
    }
    validations: dict[str, Any] = {}
    for split_name, split_value in (("train", 0), ("validation", 1)):
        rows = np.flatnonzero(arrays["split"] == split_value)
        baseline_results: list[dict[str, Any]] = []
        trained_results: list[dict[str, Any]] = []
        for row in rows:
            bucket = int(arrays["phase_bucket"][row])
            baseline_results.extend(
                _evaluate_rows(
                    evaluator,
                    baseline_parameters[bucket],
                    arrays,
                    np.asarray([row]),
                )
            )
            trained_results.extend(
                _evaluate_rows(
                    evaluator,
                    trained_parameters[bucket],
                    arrays,
                    np.asarray([row]),
                )
            )
        validations[split_name] = {
            "baseline": _summarize(baseline_results),
            "trained": _summarize(trained_results),
            "trained_trials": trained_results,
        }

    payload = render("completed")
    payload["evaluation"] = validations
    payload["promotion_observation"] = {
        "held_out_success_rate": validations["validation"]["trained"][
            "success_rate"
        ],
        "held_out_falls": validations["validation"]["trained"]["falls"],
        "gate_passed": (
            validations["validation"]["trained"]["success_rate"] >= 0.9
            and validations["validation"]["trained"]["falls"] == 0
        ),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **payload["promotion_observation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
