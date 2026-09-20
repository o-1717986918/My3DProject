#!/usr/bin/env python3
"""One-control-step CPU parity probe from complete RCSS motor telemetry.

This is a diagnostic of the state adapter and isolated-robot physics, not a
measure of soccer skill. Nearby frames are correlated; report errors, not a
success percentage or policy promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_motion_state import project_server_motion_state


REPOSITORY_ROOT = Path(__file__).parents[2]
CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_pairs(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Keep contiguous PlayOn 50 Hz observations of one unteleported player."""
    times = arrays["time_s"]
    same_player = arrays["player_number"][1:] == arrays["player_number"][:-1]
    same_match = arrays["match_id"][1:] == arrays["match_id"][:-1]
    dt = times[1:] - times[:-1]
    displacement = np.linalg.norm(arrays["self_xyz"][1:] - arrays["self_xyz"][:-1], axis=1)
    complete = np.all(arrays["target_mask"][:-1] == 1, axis=1)
    upright = arrays["self_xyz"][:-1, 2] >= 0.45
    return np.flatnonzero(
        same_player & same_match & (np.abs(dt - 0.02) <= 0.001)
        & (displacement <= 0.10) & complete & upright
    )


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Account for rounded server quaternions and the q/-q equivalence."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or min(np.linalg.norm(a), np.linalg.norm(b)) < 0.8:
        raise ValueError("invalid orientation quaternion")
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def evaluate(
    arrays: dict[str, np.ndarray], *, seed: int, max_rows_per_bucket: int
) -> dict[str, object]:
    if max_rows_per_bucket < 1:
        raise ValueError("max_rows_per_bucket must be positive")
    pairs = eligible_pairs(arrays)
    scene = RcssKickScene(load_policy_contract(CONTRACT))
    root = scene.model.joint(scene.prefix + "root")
    gyro_sensor = scene.model.sensor(scene.prefix + "torso_gyro")
    gyro_start = gyro_sensor.adr[0]
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    near = (
        (arrays["ball_valid"] == 1)
        & (arrays["ball_position_age_s"] <= 0.1)
        & (np.linalg.norm(arrays["ball_xyz"][:, :2] - arrays["self_xyz"][:, :2], axis=1) <= 1.1)
    )
    for split in (0, 1):
        for near_ball in (False, True):
            bucket = pairs[(arrays["split"][pairs] == split) & (near[pairs] == near_ball)]
            if bucket.size:
                chosen.extend(rng.choice(
                    bucket, min(max_rows_per_bucket, bucket.size), replace=False
                ).tolist())
    if not chosen:
        raise ValueError("corpus has no complete contiguous 50 Hz telemetry pairs")

    gyro_errors: dict[str, list[float]] = {"body": [], "world": []}
    for row in chosen[: min(40, len(chosen))]:
        for frame in gyro_errors:
            project_server_motion_state(scene, arrays, row, angular_frame=frame)
            measured = np.deg2rad(arrays["gyro_deg_s"][row])
            gyro_errors[frame].append(float(np.linalg.norm(
                scene.data.sensordata[gyro_start : gyro_start + 3] - measured
            )))
    angular_frame = min(gyro_errors, key=lambda frame: np.median(gyro_errors[frame]))

    trial_rows: list[dict[str, object]] = []
    for row in chosen:
        project_server_motion_state(scene, arrays, row, angular_frame=angular_frame)
        next_row = row + 1
        target = np.deg2rad(arrays["target_position_deg"][row])
        target_velocity = np.deg2rad(arrays["target_velocity_deg_s"][row])
        scene.step_joint_targets(
            target,
            kp=arrays["target_kp"][row],
            kd=arrays["target_kd"][row],
            target_velocity_rad_s=target_velocity,
            feedforward_tau=arrays["target_tau"][row],
        )
        predicted_position = np.rad2deg(scene.joint_state().position)
        predicted_velocity = np.rad2deg(scene.joint_state().velocity)
        observed_position = arrays["joint_position_deg"][next_row]
        observed_velocity = arrays["joint_velocity_deg_s"][next_row]
        predicted_root = scene.data.qpos[root.qposadr[0] : root.qposadr[0] + 7]
        observed_root = arrays["self_quat_wxyz"][next_row]
        trial_rows.append({
            "row": int(row),
            "match_id": int(arrays["match_id"][row]),
            "split": int(arrays["split"][row]),
            "near_ball": bool(near[row]),
            "motion": str(arrays["motion"][row]),
            "joint_position_mae_deg": float(np.mean(np.abs(predicted_position - observed_position))),
            "joint_velocity_mae_deg_s": float(np.mean(np.abs(predicted_velocity - observed_velocity))),
            "joint_no_motion_mae_deg": float(np.mean(np.abs(
                arrays["joint_position_deg"][row] - observed_position
            ))),
            "root_position_error_cm": float(100.0 * np.linalg.norm(
                predicted_root[:3] - arrays["self_xyz"][next_row]
            )),
            "root_orientation_error_deg": quaternion_angle_deg(
                predicted_root[3:7], observed_root
            ),
        })
    partitions: dict[str, dict[str, object]] = {}
    for split in (0, 1):
        for near_ball in (False, True):
            label = f"{'validation' if split else 'train'}_{'near_ball' if near_ball else 'other'}"
            rows = [row for row in trial_rows if row["split"] == split and row["near_ball"] == near_ball]
            partitions[label] = {
                metric: _summary([float(row[metric]) for row in rows])
                for metric in (
                    "joint_position_mae_deg", "joint_velocity_mae_deg_s",
                    "joint_no_motion_mae_deg", "root_position_error_cm",
                    "root_orientation_error_deg",
                )
            }
    return {
        "schema_version": 1,
        "purpose": "server_observed_to_exact_CPU_one_step_parity_diagnostic",
        "promotable": False,
        "pairs_available": int(pairs.size),
        "pairs_probed": len(trial_rows),
        "angular_frame": angular_frame,
        "gyro_reconstruction_error_rad_s": {
            frame: _summary(values) for frame, values in gyro_errors.items()
        },
        "partitions": partitions,
        "trials": trial_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows-per-bucket", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_920)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        not args.corpus.is_file() or not args.output.is_absolute()
        or output.is_relative_to(REPOSITORY_ROOT.resolve()) or output.exists()
    ):
        raise ValueError("corpus must exist; output must be new, absolute, and outside the repo")
    manifest = json.loads((args.corpus.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("archive_sha256") != sha256(args.corpus):
        raise ValueError("complete schema-2 telemetry and matching manifest are required")
    with np.load(args.corpus, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    report = evaluate(arrays, seed=args.seed, max_rows_per_bucket=args.max_rows_per_bucket)
    report["corpus"] = str(args.corpus.resolve())
    report["corpus_sha256"] = sha256(args.corpus)
    report["contract_sha256"] = sha256(CONTRACT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "pairs_available": report["pairs_available"],
        "pairs_probed": report["pairs_probed"],
        "angular_frame": report["angular_frame"],
        "gyro_reconstruction_error_rad_s": report["gyro_reconstruction_error_rad_s"],
        "partitions": report["partitions"],
    }, indent=2))


if __name__ == "__main__":
    main()
