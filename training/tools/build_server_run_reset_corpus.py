#!/usr/bin/env python3
"""Build sparse real-match Walk reset states for phase-v2 locomotion training.

The server observes the robot, not the full physics state. This corpus uses the
validated state projector and keeps its uncertainty explicit in the manifest.
Nearby 50 Hz frames are thinned, and match-level train/validation labels remain
attached so evaluation cannot silently reuse training matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_motion_state import project_server_motion_state


REPOSITORY_ROOT = Path(__file__).parents[2]
CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_reset_rows(
    arrays: dict[str, np.ndarray], *, stride_s: float, min_speed_m_s: float
) -> np.ndarray:
    """Keep one dynamic upright Walk observation per player/time bucket."""
    if stride_s <= 0.0 or min_speed_m_s < 0.0:
        raise ValueError("stride must be positive and speed non-negative")
    eligible = (
        (arrays["motion"] == "Walk")
        & (arrays["self_xyz"][:, 2] >= 0.55)
        & (np.linalg.norm(arrays["self_velocity_body"][:, :2], axis=1)
           >= min_speed_m_s)
        & np.all(arrays["target_mask"] == 1, axis=1)
    )
    eligible[0] = False
    eligible[1:] &= (
        (arrays["match_id"][1:] == arrays["match_id"][:-1])
        & (arrays["player_number"][1:] == arrays["player_number"][:-1])
        & (np.abs(np.diff(arrays["time_s"]) - 0.02) <= 0.001)
    )
    selected: dict[tuple[int, int, int], int] = {}
    for row in np.flatnonzero(eligible):
        key = (
            int(arrays["match_id"][row]),
            int(arrays["player_number"][row]),
            int(np.floor(float(arrays["time_s"][row]) / stride_s)),
        )
        selected.setdefault(key, int(row))
    if not selected:
        raise ValueError("no dynamic upright Walk states in server corpus")
    return np.asarray(sorted(selected.values()), dtype=np.int32)


def forward_entry_proxy(
    arrays: dict[str, np.ndarray], rows: np.ndarray
) -> np.ndarray:
    """Approximate forward handoff from observed motion, not WalkCommand."""
    body_velocity = arrays["self_velocity_body"][rows]
    quaternion = arrays["self_quat_wxyz"][rows]
    tilt_deg = np.rad2deg(np.arccos(np.clip(
        1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2),
        -1.0, 1.0,
    )))
    return (
        (body_velocity[:, 0] >= 0.5)
        & (np.abs(body_velocity[:, 1]) <= 0.25)
        & (np.max(np.abs(arrays["gyro_deg_s"][rows]), axis=1) <= 55.0)
        & (tilt_deg <= 6.0)
    )


def build_reset_arrays(
    arrays: dict[str, np.ndarray], rows: np.ndarray
) -> dict[str, np.ndarray]:
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("reset rows must be a nonempty vector")
    scene = RcssKickScene(load_policy_contract(CONTRACT), prefix="train_")
    qpos, qvel = [], []
    for row in rows:
        project_server_motion_state(scene, arrays, int(row))
        qpos.append(scene.data.qpos.copy())
        qvel.append(scene.data.qvel.copy())
    near = (
        (arrays["ball_valid"][rows] == 1)
        & (arrays["ball_position_age_s"][rows] <= 0.1)
        & (np.linalg.norm(
            arrays["ball_xyz"][rows, :2] - arrays["self_xyz"][rows, :2],
            axis=1,
        ) <= 1.1)
    )
    # Observed kinematics are only a proxy for runtime eligibility: the
    # actual WalkCommand is not present in this telemetry schema.
    forward_entry = forward_entry_proxy(arrays, rows)
    return {
        "qpos": np.asarray(qpos, dtype=np.float32),
        "qvel": np.asarray(qvel, dtype=np.float32),
        "split": arrays["split"][rows].astype(np.uint8),
        "near_ball": near.astype(np.uint8),
        "forward_entry_proxy": forward_entry.astype(np.uint8),
        "source_row": rows,
        "source_match_id": arrays["match_id"][rows].astype(np.int32),
        "source_player_number": arrays["player_number"][rows].astype(np.int32),
        "source_time_s": arrays["time_s"][rows].astype(np.float64),
        "source_planar_speed_m_s": np.linalg.norm(
            arrays["self_velocity_body"][rows, :2], axis=1
        ).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stride-s", type=float, default=0.2)
    parser.add_argument("--min-speed", type=float, default=0.2)
    args = parser.parse_args()
    output_dir = args.run_dir.resolve()
    if (
        not args.server_corpus.is_file() or not args.run_dir.is_absolute()
        or output_dir.is_relative_to(REPOSITORY_ROOT.resolve())
        or output_dir.exists()
    ):
        raise ValueError("source must exist; new absolute run dir must be outside repo")
    source_manifest = json.loads(
        (args.server_corpus.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("schema_version") != 2
        or source_manifest.get("archive_sha256") != sha256(args.server_corpus)
    ):
        raise ValueError("verified schema-2 server telemetry is required")
    with np.load(args.server_corpus, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    rows = select_reset_rows(
        arrays, stride_s=args.stride_s, min_speed_m_s=args.min_speed
    )
    resets = build_reset_arrays(arrays, rows)
    output_dir.mkdir(parents=True)
    corpus_path = output_dir / "server-run-resets.npz"
    np.savez_compressed(corpus_path, **resets)
    manifest = {
        "schema_version": 2,
        "purpose": "server_projected_phase_v2_fast_walk_reset_states",
        "promotable": False,
        "source_state": "quantized_server_observation_projected_into_single_T1_scene",
        "source_corpus": str(args.server_corpus.resolve()),
        "source_corpus_sha256": sha256(args.server_corpus),
        "contract_sha256": sha256(CONTRACT),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": sha256(corpus_path),
        "stride_s": args.stride_s,
        "min_speed_m_s": args.min_speed,
        "entries": len(rows),
        "train_entries": int(np.sum(resets["split"] == 0)),
        "validation_entries": int(np.sum(resets["split"] == 1)),
        "train_near_ball_entries": int(np.sum(
            (resets["split"] == 0) & (resets["near_ball"] == 1)
        )),
        "validation_near_ball_entries": int(np.sum(
            (resets["split"] == 1) & (resets["near_ball"] == 1)
        )),
        "train_forward_entry_proxy_entries": int(np.sum(
            (resets["split"] == 0)
            & (resets["forward_entry_proxy"] == 1)
        )),
        "validation_forward_entry_proxy_entries": int(np.sum(
            (resets["split"] == 1)
            & (resets["forward_entry_proxy"] == 1)
        )),
        "forward_entry_proxy_rule": {
            "measured_body_vx_min_m_s": 0.5,
            "measured_body_vy_abs_max_m_s": 0.25,
            "gyro_abs_max_deg_s": 55.0,
            "torso_tilt_max_deg": 6.0,
            "runtime_walk_command_observed": False,
        },
        "warning": "not full RCSS state: contacts, other players and some ball velocity are unobserved",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "entries", "train_entries", "validation_entries",
        "train_near_ball_entries", "validation_near_ball_entries",
        "train_forward_entry_proxy_entries",
        "validation_forward_entry_proxy_entries",
    )}, indent=2))


if __name__ == "__main__":
    main()
