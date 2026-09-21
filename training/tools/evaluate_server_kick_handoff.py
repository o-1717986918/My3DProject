#!/usr/bin/env python3
"""Probe a frozen kick teacher from server-observed near-ball T1 states.

One candidate is selected per contiguous player approach. The full simulator
state is projected from quantized observations, not recovered exactly: foot
contacts, other players, and unobserved ball velocity remain uncertain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_teacher import KickTeacherEvaluator, KickTeacherSpec, kick_trial_success
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


def ball_local_xy(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Project world ball estimates into the team's torso-yaw frame."""
    q = np.asarray(arrays["self_quat_wxyz"], dtype=np.float64)
    yaw = np.arctan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    delta = arrays["ball_xyz"][:, :2] - arrays["self_xyz"][:, :2]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]))


def representative_approaches(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """At most one release-like row per contiguous player near-ball approach."""
    local = ball_local_xy(arrays)
    q = arrays["self_quat_wxyz"]
    yaw_deg = np.rad2deg(np.arctan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    ))
    near = (
        (arrays["ball_valid"] == 1)
        & (arrays["ball_position_age_s"] <= 0.1)
        & (np.linalg.norm(local, axis=1) <= 1.1)
        & (arrays["self_xyz"][:, 2] >= 0.55)
    )
    release = (
        near & (local[:, 0] >= 0.20) & (local[:, 0] <= 0.65)
        & (np.abs(local[:, 1]) <= 0.25)
        & (np.abs(yaw_deg) <= 20.0)
    )
    candidates = np.flatnonzero(release)
    episodes: list[list[int]] = []
    for row in candidates:
        if (
            not episodes
            or arrays["match_id"][row] != arrays["match_id"][episodes[-1][-1]]
            or arrays["player_number"][row] != arrays["player_number"][episodes[-1][-1]]
            or arrays["time_s"][row] - arrays["time_s"][episodes[-1][-1]] > 0.5
        ):
            episodes.append([])
        episodes[-1].append(int(row))
    selected: list[int] = []
    for episode in episodes:
        rows = np.asarray(episode, dtype=np.int32)
        cost = np.sum((local[rows] - [0.34, 0.01]) ** 2, axis=1)
        selected.append(int(rows[np.argmin(cost)]))
    return np.asarray(selected, dtype=np.int32), int(np.sum(release))


def nearest_teacher_records(
    records: list[dict[str, object]], ball_xy: np.ndarray, count: int
) -> list[dict[str, object]]:
    """Select a static-position bank by local ball geometry, without outcome peeking."""
    if count < 1 or np.asarray(ball_xy).shape != (2,):
        raise ValueError("positive bank size and one local ball position are required")
    ranked = sorted(
        records,
        key=lambda item: (
            float(np.linalg.norm(
                np.asarray(ball_xy, dtype=np.float64)
                - np.array([
                    0.32 + float(item["ball_x_offset_m"]),
                    float(item["ball_y_offset_m"]),
                ])
            )),
            int(item["condition_index"]),
        ),
    )
    return ranked[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_manifest", type=Path)
    parser.add_argument("server_corpus", type=Path)
    parser.add_argument("--condition-index", type=int, default=60)
    parser.add_argument("--max-approaches", type=int, default=16)
    parser.add_argument(
        "--teacher-bank-neighbors", type=int, default=0,
        help="also probe this many nearest static-position teachers per approach",
    )
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--replay-model", type=Path)
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        not args.teacher_manifest.is_file() or not args.server_corpus.is_file()
        or not args.output.is_absolute() or output.is_relative_to(REPOSITORY_ROOT.resolve())
        or output.exists() or args.max_approaches < 1
        or args.teacher_bank_neighbors < 0 or args.teacher_bank_neighbors > 16
        or not all(path.is_file() for path in args.model)
        or (args.replay_model is not None and (
            args.replay_output is None
            or args.replay_model.resolve() not in {path.resolve() for path in args.model}
        ))
        or (args.replay_output is not None and (
            not args.replay_output.is_absolute()
            or args.replay_output.resolve().is_relative_to(REPOSITORY_ROOT.resolve())
            or args.replay_output.exists()
        ))
    ):
        raise ValueError("inputs must exist; output must be new, absolute, and outside the repo")
    source_manifest = json.loads((args.server_corpus.parent / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != 2 or source_manifest.get("archive_sha256") != sha256(args.server_corpus):
        raise ValueError("complete schema-2 server telemetry is required")
    with np.load(args.server_corpus, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    selected, candidate_frames = representative_approaches(arrays)
    if not selected.size:
        raise ValueError("no fresh forward-facing release-like player approach")
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    records = [
        record for record in teacher.get("records", [])
        if int(record["condition_index"]) == args.condition_index
        and bool(record["accepted"])
    ]
    if len(records) != 1:
        raise ValueError("condition index must select one accepted teacher")
    record = records[0]
    compatible_bank = [
        item for item in teacher.get("records", [])
        if bool(item.get("accepted"))
        and all(item.get(key) == record.get(key) for key in (
            "distance_m", "angle_deg", "requested_speed_mps",
            "desired_arrival_speed_mps", "mode",
        ))
        and "ball_x_offset_m" in item and "ball_y_offset_m" in item
    ]
    contract = load_policy_contract(CONTRACT)
    scene = RcssKickScene(contract)
    evaluator = KickTeacherEvaluator(
        KickTeacherSpec(
            target_distance_m=float(record["distance_m"]),
            target_angle_deg=float(record["angle_deg"]),
            requested_ball_speed_mps=float(record["requested_speed_mps"]),
            desired_arrival_speed_mps=float(record["desired_arrival_speed_mps"]),
            action_mode=str(record["mode"]),
        ), contract=contract,
    )
    parameters = np.asarray(record["parameters"], dtype=np.float64)
    sessions: list[tuple[Path, ort.InferenceSession]] = []
    for path in args.model:
        session = ort.InferenceSession(
            str(path.resolve()), providers=["CPUExecutionProvider"]
        )
        if (
            session.get_inputs()[0].shape != list(contract.input_shape)
            or session.get_outputs()[0].shape != list(contract.output_shape)
        ):
            raise ValueError(f"ONNX model does not match kick policy v3: {path}")
        sessions.append((path, session))
    local = ball_local_xy(arrays)
    trials: list[dict[str, object]] = []
    for row in selected[: args.max_approaches]:
        project_server_motion_state(scene, arrays, int(row))
        previous_action, previous_action_valid = infer_previous_walk_action(
            arrays, int(row)
        )
        metrics = evaluator.rollout(
            parameters,
            initial_qpos=scene.data.qpos.copy(),
            initial_qvel=scene.data.qvel.copy(),
            initial_walk_previous_action=previous_action,
        )
        trials.append({
            "row": int(row),
            "match_id": int(arrays["match_id"][row]),
            "player_number": int(arrays["player_number"][row]),
            "time_s": float(arrays["time_s"][row]),
            "split": int(arrays["split"][row]),
            "ball_local_xy_m": local[row].tolist(),
            "root_planar_speed_mps": float(np.linalg.norm(arrays["self_velocity_body"][row, :2])),
            "previous_walk_action_reconstructed": previous_action_valid,
            "contact": bool(metrics["contact"]),
            "fell": bool(metrics["fell"]),
            "success": bool(kick_trial_success(metrics)),
            "maximum_progress_m": float(metrics["maximum_progress_m"]),
            "metrics": metrics,
        })
    report = {
        "schema_version": 1,
        "purpose": "frozen_teacher_server_projected_kick_handoff_probe",
        "promotable": False,
        "source_state": "quantized_server_observation_projected_into_single_T1_exact_CPU_scene",
        "previous_walk_action": "decoded_from_previous_stable_Walk_motor_targets_when_contiguous; otherwise_zero",
        "candidate_frames": candidate_frames,
        "approaches_available": len(selected),
        "approaches_probed": len(trials),
        "contacts": sum(int(trial["contact"]) for trial in trials),
        "falls": sum(int(trial["fell"]) for trial in trials),
        "successes": sum(int(trial["success"]) for trial in trials),
        "source_corpus": str(args.server_corpus.resolve()),
        "source_corpus_sha256": sha256(args.server_corpus),
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": sha256(args.teacher_manifest),
        "condition_index": args.condition_index,
        "trials": trials,
    }
    if args.teacher_bank_neighbors:
        bank_trials: list[dict[str, object]] = []
        for row in selected[: args.max_approaches]:
            project_server_motion_state(scene, arrays, int(row))
            qpos, qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
            previous_action, _ = infer_previous_walk_action(arrays, int(row))
            candidates: list[dict[str, object]] = []
            for item in nearest_teacher_records(
                compatible_bank, local[row], args.teacher_bank_neighbors
            ):
                outcome = evaluator.rollout(
                    np.asarray(item["parameters"], dtype=np.float64),
                    initial_qpos=qpos, initial_qvel=qvel,
                    initial_walk_previous_action=previous_action,
                )
                candidates.append({
                    "condition_index": int(item["condition_index"]),
                    "contact": bool(outcome["contact"]),
                    "fell": bool(outcome["fell"]),
                    "success": bool(kick_trial_success(outcome)),
                    "metrics": outcome,
                })
            bank_trials.append({
                "row": int(row), "split": int(arrays["split"][row]),
                "candidates": candidates,
            })
        report["teacher_bank"] = {
            "selection": "nearest_local_ball_position_without_outcome_peeking",
            "neighbors": args.teacher_bank_neighbors,
            "outcome_oracle_is_deployment_policy": False,
            "trials": bank_trials,
        }
    models: list[dict[str, object]] = []
    for path, session in sessions:
        model_trials: list[dict[str, object]] = []
        for row in selected[: args.max_approaches]:
            project_server_motion_state(scene, arrays, int(row))
            previous_action, _ = infer_previous_walk_action(arrays, int(row))
            metrics = evaluator.rollout(
                None,
                initial_qpos=scene.data.qpos.copy(),
                initial_qvel=scene.data.qvel.copy(),
                initial_walk_previous_action=previous_action,
                kick_policy_session=session,
            )
            model_trials.append({
                "row": int(row),
                "contact": bool(metrics["contact"]),
                "fell": bool(metrics["fell"]),
                "success": bool(kick_trial_success(metrics)),
                "maximum_progress_m": float(metrics["maximum_progress_m"]),
                "metrics": metrics,
            })
        models.append({
            "model": str(path.resolve()),
            "model_sha256": sha256(path),
            "contacts": sum(int(trial["contact"]) for trial in model_trials),
            "falls": sum(int(trial["fell"]) for trial in model_trials),
            "successes": sum(int(trial["success"]) for trial in model_trials),
            "trials": model_trials,
        })
    report["models"] = models
    if args.replay_output is not None:
        replay_session = None
        replay_trials = trials
        if args.replay_model is not None:
            replay_session = next(
                session for path, session in sessions
                if path.resolve() == args.replay_model.resolve()
            )
            replay_trials = next(
                model["trials"] for model in models
                if model["model"] == str(args.replay_model.resolve())
            )
        best = max(replay_trials, key=lambda trial: (
            int(trial["success"]), float(trial["metrics"]["score"])
        ))
        best_row = int(best["row"])
        project_server_motion_state(scene, arrays, best_row)
        previous_action, _ = infer_previous_walk_action(arrays, best_row)
        evaluator.rollout(
            parameters if replay_session is None else None,
            initial_qpos=scene.data.qpos.copy(),
            initial_qvel=scene.data.qvel.copy(),
            initial_walk_previous_action=previous_action,
            capture_states=True,
            kick_policy_session=replay_session,
        )
        replay = args.replay_output.resolve()
        replay.parent.mkdir(parents=True, exist_ok=True)
        trajectory = evaluator.captured_qpos
        np.savez_compressed(
            replay, qpos=trajectory,
            split=np.full(
                trajectory.shape[0], int(arrays["split"][best_row]), dtype=np.uint8
            ),
            start_frame=np.full(trajectory.shape[0], best_row, dtype=np.int32),
        )
        report["replay"] = str(replay)
        report["replay_sha256"] = sha256(replay)
        report["replay_row"] = best_row
        report["replay_model"] = (
            str(args.replay_model.resolve()) if args.replay_model is not None
            else "frozen_teacher"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "candidate_frames", "approaches_available", "approaches_probed",
        "contacts", "falls", "successes",
    )} | {"models": [
        {key: model[key] for key in ("model", "contacts", "falls", "successes")}
        for model in models
    ]}, indent=2))


if __name__ == "__main__":
    main()
