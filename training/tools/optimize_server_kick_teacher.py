#!/usr/bin/env python3
"""Small exact-CPU teacher search on selected *server-observed* kick entries.

This tunes a low-dimensional trajectory, not a neural policy. Rows not listed
for optimization are evaluated untouched; none of the outcomes alone licenses
competition deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import (
    PARAMETER_LOWER, PARAMETER_NAMES, PARAMETER_UPPER,
    KickTeacherEvaluator, KickTeacherSpec, cem_optimize, kick_trial_success,
)
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_motion_state import (
    infer_previous_walk_action, project_server_motion_state,
)
from tools.evaluate_server_kick_handoff import representative_approaches


REPOSITORY_ROOT = Path(__file__).parents[2]
CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_objective(scores: np.ndarray, successes: np.ndarray) -> float:
    if scores.ndim != 1 or successes.shape != scores.shape or scores.size < 1:
        raise ValueError("objective arrays must be aligned and nonempty")
    return float(0.75 * np.mean(scores) + 0.25 * np.min(scores) + 10.0 * np.mean(successes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("server_corpus", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--train-row", type=int, action="append", required=True)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_921)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        not args.teacher_manifest.is_file() or not args.server_corpus.is_file()
        or not args.output.is_absolute() or output.is_relative_to(REPOSITORY_ROOT.resolve())
        or output.exists() or args.population < 2 or args.generations < 1
        or len(set(args.train_row)) != len(args.train_row)
    ):
        raise ValueError("invalid source, output, optimization count, or train rows")
    source_manifest = json.loads((args.server_corpus.parent / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != 2 or source_manifest.get("archive_sha256") != sha256(args.server_corpus):
        raise ValueError("complete schema-2 server telemetry is required")
    with np.load(args.server_corpus, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    entries, _ = representative_approaches(arrays)
    train = np.asarray(args.train_row, dtype=np.int32)
    if not set(train.tolist()) <= set(entries.tolist()):
        raise ValueError("train rows must be representative observed approaches")
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    records = [
        item for item in teacher.get("records", [])
        if int(item["condition_index"]) == args.condition_index and bool(item["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher")
    record = records[0]
    contract = load_policy_contract(CONTRACT)
    scene = RcssKickScene(contract)
    evaluator = KickTeacherEvaluator(KickTeacherSpec(
        target_distance_m=float(record["distance_m"]),
        target_angle_deg=float(record["angle_deg"]),
        requested_ball_speed_mps=float(record["requested_speed_mps"]),
        desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
        action_mode=str(record["mode"]),
    ), contract=contract)
    states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in entries:
        project_server_motion_state(scene, arrays, int(row))
        previous_action, _ = infer_previous_walk_action(arrays, int(row))
        states[int(row)] = (
            scene.data.qpos.copy(), scene.data.qvel.copy(), previous_action
        )

    def probe(row: int, parameters: np.ndarray) -> dict[str, object]:
        qpos, qvel, previous_action = states[row]
        metrics = evaluator.rollout(
            parameters, initial_qpos=qpos, initial_qvel=qvel,
            initial_walk_previous_action=previous_action,
        )
        return {"success": bool(kick_trial_success(metrics)), "metrics": metrics}

    initial = np.asarray(record["parameters"], dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        values = [probe(int(row), parameters) for row in train]
        return robust_objective(
            np.asarray([value["metrics"]["score"] for value in values]),
            np.asarray([value["success"] for value in values], dtype=np.float64),
        )

    result = cem_optimize(
        objective,
        initial_mean=initial,
        initial_std=np.maximum(0.12 * (PARAMETER_UPPER - PARAMETER_LOWER), 0.03),
        lower=PARAMETER_LOWER,
        upper=PARAMETER_UPPER,
        seed=args.seed,
        population=args.population,
        generations=args.generations,
        elite_fraction=0.25,
        smoothing=0.35,
        progress=lambda generation, metrics: print(
            json.dumps({"generation": generation, "best_score": metrics["best_score"]}),
            flush=True,
        ),
    )
    evaluations: list[dict[str, object]] = []
    for row in entries:
        row = int(row)
        baseline = probe(row, initial)
        trained = probe(row, result.parameters)
        evaluations.append({
            "row": row,
            "match_id": int(arrays["match_id"][row]),
            "player_number": int(arrays["player_number"][row]),
            "used_for_optimization": row in set(train.tolist()),
            "baseline": baseline,
            "trained": trained,
        })
    report = {
        "schema_version": 1,
        "purpose": "server_projected_kick_teacher_parameter_search",
        "promotable": False,
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256(args.teacher_manifest),
        "source_corpus": str(args.server_corpus.resolve()),
        "source_corpus_sha256": sha256(args.server_corpus),
        "contract_sha256": sha256(CONTRACT),
        "condition_index": args.condition_index,
        "train_rows": train.tolist(),
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "parameter_names": PARAMETER_NAMES,
        "initial_parameters": initial.tolist(),
        "trained_parameters": result.parameters.tolist(),
        "optimization_score": result.score,
        "history": result.history,
        "evaluations": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "train_rows": train.tolist(),
        "entries": len(evaluations),
        "baseline_successes": sum(int(item["baseline"]["success"]) for item in evaluations),
        "trained_successes": sum(int(item["trained"]["success"]) for item in evaluations),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
