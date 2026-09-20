#!/usr/bin/env python3
"""Probe a frozen 2 m kick teacher on match-command replay entry states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, kick_trial_success


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representative_rows(
    rollout_ids: np.ndarray,
    ball_local_xyz: np.ndarray,
    release_like: np.ndarray,
    *,
    max_rollouts: int,
) -> np.ndarray:
    """Choose one near-anchor state per rollout, never counting frames as trials."""
    ids = np.asarray(rollout_ids, dtype=np.int32)
    ball = np.asarray(ball_local_xyz, dtype=np.float64)
    eligible = np.asarray(release_like, dtype=bool)
    if (
        ids.ndim != 1
        or ball.shape != (ids.size, 3)
        or eligible.shape != ids.shape
        or max_rollouts < 1
        or not np.isfinite(ball).all()
    ):
        raise ValueError("handoff arrays or max-rollouts are invalid")
    selected: list[int] = []
    for rollout_id in np.unique(ids[eligible])[:max_rollouts]:
        rows = np.flatnonzero((ids == rollout_id) & eligible)
        squared_error = np.sum(
            np.square(ball[rows, :2] - np.array([0.34, 0.01])), axis=1
        )
        selected.append(int(rows[np.argmin(squared_error)]))
    if not selected:
        raise ValueError("corpus has no release-like approach rollouts")
    return np.asarray(selected, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("match_corpus", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--max-rollouts", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        not args.teacher_manifest.is_file()
        or not args.match_corpus.is_file()
        or not args.output.is_absolute()
        or output.is_relative_to(REPOSITORY_ROOT.resolve())
        or output.exists()
    ):
        raise ValueError("inputs must exist; output must be new and outside the repo")
    corpus_manifest_path = args.match_corpus.parent / "manifest.json"
    manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("purpose") != "kick_policy_v3_match_command_replay_handoff_corpus"
        or manifest.get("corpus_sha256") != _sha256(args.match_corpus)
        or manifest.get("kick_contract_sha256") != _sha256(DEFAULT_CONTRACT)
    ):
        raise ValueError("match handoff corpus manifest/hash/contract mismatch")
    with np.load(args.match_corpus, allow_pickle=False) as archive:
        required = {
            "qpos", "qvel", "walk_previous_action", "ball_position_local_m",
            "root_velocity", "rollout_id", "source_trace_id", "split",
            "release_like_geometry",
        }
        if not required <= set(archive.files):
            raise ValueError("match handoff corpus is missing required arrays")
        arrays = {name: np.asarray(archive[name]) for name in required}
    rows = representative_rows(
        arrays["rollout_id"],
        arrays["ball_position_local_m"],
        arrays["release_like_geometry"],
        max_rollouts=args.max_rollouts,
    )
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    matches = [
        record for record in teacher.get("records", [])
        if int(record["condition_index"]) == args.condition_index
        and bool(record["accepted"])
    ]
    if len(matches) != 1:
        raise ValueError("condition index must select one accepted teacher")
    record = matches[0]
    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(
            target_distance_m=float(record["distance_m"]),
            target_angle_deg=float(record["angle_deg"]),
            requested_ball_speed_mps=float(record["requested_speed_mps"]),
            desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
            action_mode=str(record["mode"]),
        ),
        contract=load_policy_contract(DEFAULT_CONTRACT),
    )
    parameters = np.asarray(record["parameters"], dtype=np.float64)
    trials: list[dict[str, object]] = []
    for row in rows:
        metrics = evaluator.rollout(
            parameters,
            initial_qpos=arrays["qpos"][row],
            initial_qvel=arrays["qvel"][row],
            initial_walk_previous_action=arrays["walk_previous_action"][row],
        )
        trials.append({
            "row": int(row),
            "rollout_id": int(arrays["rollout_id"][row]),
            "source_trace_id": int(arrays["source_trace_id"][row]),
            "split": int(arrays["split"][row]),
            "ball_position_local_m": arrays["ball_position_local_m"][row].tolist(),
            "root_planar_speed_mps": float(
                np.linalg.norm(arrays["root_velocity"][row, :2])
            ),
            "contact": bool(metrics["contact"]),
            "fell": bool(metrics["fell"]),
            "success": bool(kick_trial_success(metrics)),
            "maximum_progress_m": float(metrics["maximum_progress_m"]),
            "metrics": metrics,
        })
    report = {
        "schema_version": 1,
        "purpose": "frozen_kick_teacher_match_replay_probe",
        "promotable": False,
        "corpus": str(args.match_corpus.resolve()),
        "corpus_sha256": _sha256(args.match_corpus),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": _sha256(args.teacher_manifest),
        "teacher_condition_index": args.condition_index,
        "unique_rollouts": len(trials),
        "contacts": sum(int(trial["contact"]) for trial in trials),
        "falls": sum(int(trial["fell"]) for trial in trials),
        "successes": sum(int(trial["success"]) for trial in trials),
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("unique_rollouts", "contacts", "falls", "successes", "corpus")}, indent=2))


if __name__ == "__main__":
    main()
