#!/usr/bin/env python3
"""Archive actual RCSS agent observations and motor targets for offline learning.

These are server-observed torso/joint states, not simulator qpos/qvel. In
particular, torso height is not a MuJoCo root height. Preserve the raw frame
and only construct simulator initial states in a separately tested adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).parents[2]
PREFIX = "APOLLO_REBUILD_MOTION_TELEMETRY "
JOINT_COUNT = 23
# Ordered exactly as T1RobotModel::readable_joint_names. The training policy
# contract spells only the first name AAHead_yaw; that is a naming alias.
JOINT_ORDER = (
    "Head_yaw", "Head_pitch", "Left_Shoulder_Pitch", "Left_Shoulder_Roll",
    "Left_Elbow_Pitch", "Left_Elbow_Yaw", "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Waist", "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
VECTOR_SHAPES = {
    "self_xyz": 3,
    "self_quat_wxyz": 4,
    "self_velocity_body": 3,
    "gyro_deg_s": 3,
    "ball_xyz": 3,
    "ball_velocity": 3,
    "joint_position_deg": JOINT_COUNT,
    "joint_velocity_deg_s": JOINT_COUNT,
    "target_position_deg": JOINT_COUNT,
    "target_velocity_deg_s": JOINT_COUNT,
    "target_kp": JOINT_COUNT,
    "target_kd": JOINT_COUNT,
    "target_tau": JOINT_COUNT,
}
REFERENCE_VECTOR_SHAPES = {
    "reference_position_deg": JOINT_COUNT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_telemetry_line(line: str) -> dict[str, object] | None:
    """Return one complete observed frame, or None for unrelated log lines."""
    if not line.startswith(PREFIX):
        return None
    fields = dict(token.split("=", 1) for token in line.split()[1:])
    schema = fields.get("schema")
    if schema not in {"2", "3"}:
        raise ValueError("motion telemetry requires complete schema 2 or 3 controls")
    frame: dict[str, object] = {
        "telemetry_schema": int(schema),
        "time_s": float(fields["t"]),
        "player_number": int(fields["player"]),
        "side": fields["side"],
        "motion": fields["motion"],
        "motion_elapsed_s": float(fields.get("motion_elapsed", -1.0)),
        "ball_valid": int(fields["ball_valid"]),
        "ball_position_age_s": float(fields["ball_age"]),
        "ball_velocity_valid": int(fields["ball_velocity_valid"]),
    }
    for name, size in VECTOR_SHAPES.items():
        vector = np.fromstring(fields[name], sep=",", dtype=np.float64)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid {name} in motion telemetry")
        frame[name] = vector
    for name, size in REFERENCE_VECTOR_SHAPES.items():
        vector = np.fromstring(
            fields[name], sep=",", dtype=np.float64
        ) if schema == "3" else np.zeros(size, dtype=np.float64)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid {name} in motion telemetry")
        frame[name] = vector
    mask = fields["target_mask"]
    if len(mask) != JOINT_COUNT or set(mask) - {"0", "1"}:
        raise ValueError("invalid target mask in motion telemetry")
    frame["target_mask"] = np.fromiter((char == "1" for char in mask), dtype=np.uint8)
    reference_mask = fields.get("reference_mask", "0" * JOINT_COUNT)
    if len(reference_mask) != JOINT_COUNT or set(reference_mask) - {"0", "1"}:
        raise ValueError("invalid reference mask in motion telemetry")
    frame["reference_mask"] = np.fromiter(
        (char == "1" for char in reference_mask), dtype=np.uint8
    )
    if (
        not np.isfinite(frame["time_s"])
        or not np.isfinite(frame["motion_elapsed_s"])
        or frame["player_number"] not in range(1, 12)
        or frame["side"] not in {"left", "right"}
        or frame["ball_valid"] not in {0, 1}
        or frame["ball_velocity_valid"] not in {0, 1}
        or (frame["ball_valid"] and not np.isfinite(frame["ball_position_age_s"]))
    ):
        raise ValueError("invalid scalar in motion telemetry")
    return frame


def collect(
    match_dirs: list[Path],
    *,
    validation_match_count: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for match_id, match_dir in enumerate(match_dirs):
        logs = sorted(match_dir.glob("Apollo-Rebuild-*.log"))
        if not logs:
            raise ValueError(f"no rebuild player logs in {match_dir}")
        for log in logs:
            count = 0
            previous_time = -np.inf
            with log.open(encoding="utf-8") as stream:
                for line in stream:
                    frame = parse_telemetry_line(line)
                    if frame is None:
                        continue
                    if frame["time_s"] <= previous_time:
                        raise ValueError(f"non-increasing telemetry time in {log}")
                    previous_time = frame["time_s"]
                    frame["match_id"] = match_id
                    rows.append(frame)
                    count += 1
            sources.append({"log": str(log.resolve()), "sha256": sha256(log), "frames": count})
    if not rows:
        raise ValueError("no motion telemetry in match logs; enable its interval")
    arrays = {
        name: np.stack([row[name] for row in rows]).astype(np.float32)
        for name in VECTOR_SHAPES | REFERENCE_VECTOR_SHAPES
    }
    for name in ("time_s", "ball_position_age_s", "motion_elapsed_s"):
        arrays[name] = np.asarray([row[name] for row in rows], dtype=np.float64)
    for name in (
        "telemetry_schema", "player_number", "ball_valid",
        "ball_velocity_valid", "match_id",
    ):
        arrays[name] = np.asarray([row[name] for row in rows], dtype=np.int32)
    arrays["target_mask"] = np.stack([row["target_mask"] for row in rows])
    arrays["reference_mask"] = np.stack(
        [row["reference_mask"] for row in rows]
    )
    for name in ("side", "motion"):
        arrays[name] = np.asarray([row[name] for row in rows], dtype="U64")
    # Correlated players and time windows from one match stay on one side.
    unique_matches = np.unique(arrays["match_id"])
    if validation_match_count < 1:
        raise ValueError("validation_match_count must be positive")
    if len(unique_matches) > 1 and validation_match_count >= len(unique_matches):
        raise ValueError("validation matches must leave at least one training match")
    validation_matches = (
        unique_matches[-validation_match_count:]
        if len(unique_matches) > 1
        else np.empty(0, dtype=unique_matches.dtype)
    )
    arrays["split"] = np.isin(
        arrays["match_id"], validation_matches
    ).astype(np.uint8)
    near_ball = (
        (arrays["ball_valid"] == 1)
        & (arrays["ball_position_age_s"] <= 0.1)
        & (np.linalg.norm(arrays["ball_xyz"][:, :2] - arrays["self_xyz"][:, :2], axis=1) <= 1.1)
    )
    summary = {
        "samples": len(rows),
        "match_groups": len(unique_matches),
        "validation_match_groups": int(validation_matches.size),
        "train_samples": int(np.sum(arrays["split"] == 0)),
        "validation_samples": int(np.sum(arrays["split"] == 1)),
        "fresh_near_ball_samples": int(np.sum(near_ball)),
        "complete_target_samples": int(np.sum(np.all(arrays["target_mask"] == 1, axis=1))),
        "complete_reference_samples": int(
            np.sum(np.all(arrays["reference_mask"] == 1, axis=1))
        ),
        "sources": sources,
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-dir", type=Path, action="append", required=True)
    parser.add_argument("--validation-match-count", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if (
        not all(path.is_dir() for path in args.match_dir)
        or not args.run_dir.is_absolute()
        or run_dir.is_relative_to(REPOSITORY_ROOT.resolve())
        or run_dir.exists()
    ):
        raise ValueError("match dirs must exist; run dir must be new, absolute, and outside the repo")
    arrays, summary = collect(
        args.match_dir,
        validation_match_count=args.validation_match_count,
    )
    run_dir.mkdir(parents=True)
    archive = run_dir / "server-motion-telemetry.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {
        "schema_version": 3,
        "purpose": "server_observed_motion_and_sent_targets",
        "promotable": False,
        "units": "server canonical team frame; joint positions/targets degrees; joint velocities degrees/s",
        "joint_order": JOINT_ORDER,
        "policy_first_joint_alias": "AAHead_yaw",
        "simulator_state": False,
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "collector_sha256": sha256(Path(__file__)),
        "runtime_agent_source_sha256": sha256(
            REPOSITORY_ROOT / "runtime" / "apollo_rebuild" / "src" / "app" / "agent_app.cc"
        ),
        **summary,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "samples", "match_groups", "train_samples", "validation_samples",
        "fresh_near_ball_samples", "complete_target_samples",
        "complete_reference_samples",
    )}, indent=2))


if __name__ == "__main__":
    main()
