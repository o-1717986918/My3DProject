#!/usr/bin/env python3
"""Generate exact-CPU kick parameter labels for every training transition."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
_WORKER_EVALUATOR: KickTeacherEvaluator | None = None


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


def _worker_initialize(spec: dict[str, Any], contract_path: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = KickTeacherEvaluator(
        KickTeacherSpec(**spec),
        contract=load_policy_contract(Path(contract_path)),
    )


def _worker_optimize(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("worker evaluator was not initialized")
    evaluator = _WORKER_EVALUATOR
    qpos = np.asarray(task["qpos"], dtype=np.float64)
    qvel = np.asarray(task["qvel"], dtype=np.float64)
    initial = np.asarray(task["initial_parameters"], dtype=np.float64)

    baseline = evaluator.rollout(
        initial,
        initial_qpos=qpos,
        initial_qvel=qvel,
        capture_targets=True,
    )
    observations = evaluator.captured_observations
    if (
        observations.ndim != 2
        or observations.shape[1] != evaluator.contract.observation_size
    ):
        raise RuntimeError("failed to capture the transition actor observation")
    actor_observation = observations[0]

    def objective(parameters: np.ndarray) -> float:
        metrics = evaluator.rollout(
            parameters, initial_qpos=qpos, initial_qvel=qvel
        )
        return float(
            metrics["score"]
            + task["success_bonus"] * float(kick_trial_success(metrics))
        )

    result = cem_optimize(
        objective,
        initial_mean=initial,
        initial_std=np.maximum(
            task["initial_std_fraction"]
            * (PARAMETER_UPPER - PARAMETER_LOWER),
            0.03,
        ),
        lower=PARAMETER_LOWER,
        upper=PARAMETER_UPPER,
        seed=int(task["seed"]),
        population=int(task["population"]),
        generations=int(task["generations"]),
        elite_fraction=0.25,
        smoothing=0.35,
    )
    trained = evaluator.rollout(
        result.parameters, initial_qpos=qpos, initial_qvel=qvel
    )
    return {
        "corpus_index": int(task["corpus_index"]),
        "rollout_id": int(task["rollout_id"]),
        "phase_bucket": int(task["phase_bucket"]),
        "seed": int(task["seed"]),
        "initial_parameters": initial.tolist(),
        "actor_observation": actor_observation.astype(float).tolist(),
        "parameters": result.parameters.tolist(),
        "baseline_success": bool(kick_trial_success(baseline)),
        "trained_success": bool(kick_trial_success(trained)),
        "baseline_metrics": baseline,
        "trained_metrics": trained,
        "objective": result.score,
        "history": list(result.history),
    }


def _load_source(
    teacher_manifest: Path,
    transition_corpus: Path,
    phase_initializer: Path,
    condition_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[int, np.ndarray]]:
    teacher = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in teacher["records"]
        if int(record["condition_index"]) == condition_index
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher record")

    corpus_manifest_path = transition_corpus.with_suffix(".json")
    corpus_manifest = json.loads(
        corpus_manifest_path.read_text(encoding="utf-8")
    )
    if corpus_manifest.get("npz_sha256") != _sha256(transition_corpus):
        raise ValueError("transition corpus hash mismatch")
    if int(corpus_manifest["teacher_condition_index"]) != condition_index:
        raise ValueError("transition corpus teacher condition mismatch")
    with np.load(transition_corpus, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "split", "rollout_id", "phase_bucket"}
        if not required <= set(archive.files):
            raise ValueError("transition corpus is missing required arrays")
        arrays = {name: np.asarray(archive[name]) for name in required}

    initializer = json.loads(phase_initializer.read_text(encoding="utf-8"))
    if initializer.get("purpose") != "exact_cpu_phase_indexed_kick_teacher_training":
        raise ValueError("phase initializer has the wrong purpose")
    if not bool(initializer.get("complete")):
        raise ValueError("phase initializer is incomplete")
    if int(initializer.get("condition_index", -1)) != condition_index:
        raise ValueError("phase initializer condition mismatch")
    if int(initializer.get("phase_bucket_count", -1)) != int(
        corpus_manifest["phase_bucket_count"]
    ):
        raise ValueError("phase initializer bucket definition mismatch")
    parameters = {
        int(node["phase_bucket"]): np.asarray(node["parameters"], dtype=np.float64)
        for node in initializer["nodes"]
    }
    buckets = set(
        int(value)
        for value in arrays["phase_bucket"][arrays["split"] == 0]
    )
    if set(parameters) != buckets:
        raise ValueError("phase initializer does not cover every training bucket")
    return records[0], arrays, parameters


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("transition_corpus", type=Path)
    parser.add_argument("phase_initializer", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seed", type=int, default=7401)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--success-bonus", type=float, default=50.0)
    parser.add_argument("--initial-std-fraction", type=float, default=0.10)
    parser.add_argument(
        "--repair-source",
        type=Path,
        help="copy successful labels and re-optimize only failures from this manifest",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    json_path = args.output_prefix.with_suffix(".json")
    npz_path = args.output_prefix.with_suffix(".npz")
    if not json_path.is_absolute() or json_path.is_relative_to(Path.cwd()):
        raise ValueError("output prefix must be absolute and outside the repository")
    if (json_path.exists() or npz_path.exists()) and not args.resume:
        raise FileExistsError("output exists; pass --resume explicitly")
    if args.population < 2 or args.generations < 1 or args.workers < 1:
        raise ValueError("population, generations and workers must be positive")
    if args.success_bonus < 0.0 or not 0.0 < args.initial_std_fraction <= 0.5:
        raise ValueError("optimization scale is invalid")

    contract = load_policy_contract(args.contract)
    if contract.policy_name != "kick_policy_v3":
        raise ValueError("transition labels require kick_policy_v3")
    record, arrays, initializers = _load_source(
        args.teacher_manifest,
        args.transition_corpus,
        args.phase_initializer,
        args.condition_index,
    )
    train_rows = np.flatnonzero(arrays["split"] == 0)
    validation_rows = np.flatnonzero(arrays["split"] == 1)
    if train_rows.size < 2 or validation_rows.size < 2:
        raise ValueError("transition corpus needs disjoint train and validation rows")
    if set(arrays["rollout_id"][train_rows].tolist()) & set(
        arrays["rollout_id"][validation_rows].tolist()
    ):
        raise ValueError("transition corpus leaks rollout IDs")

    spec = {
        "target_distance_m": float(record["distance_m"]),
        "target_angle_deg": float(record["angle_deg"]),
        "requested_ball_speed_mps": float(record["requested_speed_mps"]),
        "desired_arrival_speed_mps": float(record["desired_arrival_speed_mps"]),
        "action_mode": str(record["mode"]),
        "evaluation_duration_s": 3.0,
    }
    identity = {
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "transition_corpus_sha256": _sha256(args.transition_corpus),
        "phase_initializer_sha256": _sha256(args.phase_initializer),
        "contract_sha256": _sha256(args.contract),
        "seed": args.seed,
        "population": args.population,
        "generations": args.generations,
        "success_bonus": args.success_bonus,
        "initial_std_fraction": args.initial_std_fraction,
        "repair_source_sha256": (
            _sha256(args.repair_source) if args.repair_source is not None else None
        ),
    }
    completed: dict[int, dict[str, Any]] = {}
    repair_labels: dict[int, dict[str, Any]] = {}
    elapsed_before_resume = 0.0
    if args.repair_source is not None:
        repair = json.loads(args.repair_source.read_text(encoding="utf-8"))
        if (
            repair.get("purpose")
            != "exact_cpu_per_transition_kick_teacher_labels"
            or not bool(repair.get("complete"))
            or repair.get("teacher_manifest_sha256")
            != identity["teacher_manifest_sha256"]
            or repair.get("transition_corpus_sha256")
            != identity["transition_corpus_sha256"]
            or repair.get("contract_sha256") != identity["contract_sha256"]
        ):
            raise ValueError("repair source is incomplete or bound to other inputs")
        repair_labels = {
            int(label["corpus_index"]): label for label in repair["labels"]
        }
        if set(repair_labels) != set(train_rows.tolist()):
            raise ValueError("repair source does not cover every training entry")
        completed = {
            index: label
            for index, label in repair_labels.items()
            if bool(label["trained_success"])
        }
    if args.resume and json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("resume checkpoint does not match requested run")
        completed = {
            int(label["corpus_index"]): label for label in previous["labels"]
        }
        elapsed_before_resume = float(previous.get("elapsed_seconds", 0.0))
    pending = [int(row) for row in train_rows if int(row) not in completed]
    started = time.time()

    def render(status: str) -> dict[str, Any]:
        labels = [completed[index] for index in sorted(completed)]
        return {
            "schema_version": 1,
            "purpose": "exact_cpu_per_transition_kick_teacher_labels",
            "status": status,
            "complete": status == "completed",
            "promotable": False,
            "promotion_blocker": (
                "training labels only; held-out rollout gates remain required"
            ),
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "teacher_manifest": str(args.teacher_manifest.resolve()),
            "transition_corpus": str(args.transition_corpus.resolve()),
            "phase_initializer": str(args.phase_initializer.resolve()),
            "repair_source": (
                str(args.repair_source.resolve())
                if args.repair_source is not None
                else None
            ),
            "contract": str(args.contract.resolve()),
            **identity,
            "workers": args.workers,
            "training_entries": int(train_rows.size),
            "validation_entries_untouched": int(validation_rows.size),
            "completed_entries": len(labels),
            "elapsed_seconds": elapsed_before_resume + time.time() - started,
            "parameter_names": list(PARAMETER_NAMES),
            "labels": labels,
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for row in pending:
        bucket = int(arrays["phase_bucket"][row])
        tasks.append(
            {
                "corpus_index": row,
                "rollout_id": int(arrays["rollout_id"][row]),
                "phase_bucket": bucket,
                "qpos": arrays["qpos"][row],
                "qvel": arrays["qvel"][row],
                "initial_parameters": (
                    repair_labels[row]["parameters"]
                    if row in repair_labels
                    else initializers[bucket]
                ),
                "seed": args.seed + int(arrays["rollout_id"][row]),
                "population": args.population,
                "generations": args.generations,
                "success_bonus": args.success_bonus,
                "initial_std_fraction": args.initial_std_fraction,
            }
        )

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_initialize,
        initargs=(spec, str(args.contract.resolve())),
    ) as executor:
        futures = {executor.submit(_worker_optimize, task): task for task in tasks}
        for future in as_completed(futures):
            label = future.result()
            completed[int(label["corpus_index"])] = label
            _write_checkpoint(json_path, render("running"))
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "total": int(train_rows.size),
                        "rollout_id": label["rollout_id"],
                        "success": label["trained_success"],
                    }
                ),
                flush=True,
            )

    payload = render("completed")
    labels = payload["labels"]
    if len(labels) != train_rows.size:
        raise RuntimeError("not every training transition has a label")
    np.savez_compressed(
        npz_path,
        corpus_index=np.asarray(
            [label["corpus_index"] for label in labels], np.int32
        ),
        rollout_id=np.asarray([label["rollout_id"] for label in labels], np.int32),
        phase_bucket=np.asarray(
            [label["phase_bucket"] for label in labels], np.int32
        ),
        actor_observation=np.asarray(
            [label["actor_observation"] for label in labels], np.float32
        ),
        parameters=np.asarray(
            [label["parameters"] for label in labels], np.float32
        ),
        baseline_success=np.asarray(
            [label["baseline_success"] for label in labels], np.uint8
        ),
        trained_success=np.asarray(
            [label["trained_success"] for label in labels], np.uint8
        ),
    )
    payload["npz"] = str(npz_path.resolve())
    payload["npz_sha256"] = _sha256(npz_path)
    payload["summary"] = {
        "initializer_successes": sum(
            label["baseline_success"] for label in labels
        ),
        "repair_source_successes": (
            sum(label["trained_success"] for label in repair_labels.values())
            if repair_labels
            else None
        ),
        "trained_successes": sum(label["trained_success"] for label in labels),
        "falls": sum(bool(label["trained_metrics"]["fell"]) for label in labels),
        "contacts": sum(
            bool(label["trained_metrics"]["contact"]) for label in labels
        ),
    }
    _write_checkpoint(json_path, payload)
    print(json.dumps({"output": str(json_path), **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
