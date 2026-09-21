#!/usr/bin/env python3
"""Project deduplicated server kick entries into a grouped exact-physics corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_transition import estimate_locomotion_phase
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_motion_state import (
    infer_previous_walk_action, project_server_motion_state,
)
from my3d_rl.t1_control import APOLLO_DEFAULT_POSE
from tools.evaluate_server_kick_handoff import (
    ball_local_xy, representative_approaches,
)
from tools.generate_kick_transition_corpus import phase_buckets


REPOSITORY_ROOT = Path(__file__).parents[2]
CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_corpus", type=Path)
    parser.add_argument("--teacher-condition-index", type=int, default=60)
    parser.add_argument("--phase-buckets", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    output_npz = args.output_prefix.with_suffix(".npz").resolve()
    output_json = args.output_prefix.with_suffix(".json").resolve()
    if (
        not args.server_corpus.is_file()
        or not args.output_prefix.is_absolute()
        or output_npz.is_relative_to(REPOSITORY_ROOT.resolve())
        or output_npz.exists() or output_json.exists()
        or args.phase_buckets < 2
    ):
        raise ValueError("source must exist and external outputs must be new")
    source_manifest = json.loads(
        (args.server_corpus.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("schema_version") != 2
        or source_manifest.get("archive_sha256") != sha256(args.server_corpus)
    ):
        raise ValueError("complete schema-2 server telemetry is required")
    with np.load(args.server_corpus, allow_pickle=False) as archive:
        server = {name: np.asarray(archive[name]) for name in archive.files}
    rows, candidate_frames = representative_approaches(server)
    if rows.size < 4 or set(server["split"][rows].tolist()) != {0, 1}:
        raise ValueError("server approaches need at least four entries and both splits")

    contract = load_policy_contract(CONTRACT)
    scene = RcssKickScene(contract)
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    walk_previous_action: list[np.ndarray] = []
    walk_previous_action_valid: list[bool] = []
    locomotion_phase: list[np.ndarray] = []
    for row in rows:
        project_server_motion_state(scene, server, int(row))
        action, valid = infer_previous_walk_action(server, int(row))
        joint_state = scene.joint_state()
        phase = estimate_locomotion_phase(
            joint_state.position - APOLLO_DEFAULT_POSE,
            joint_state.velocity, contract.joint_order,
        )
        qpos.append(scene.data.qpos.copy())
        qvel.append(scene.data.qvel.copy())
        walk_previous_action.append(action)
        walk_previous_action_valid.append(valid)
        locomotion_phase.append(phase.sin_cos)
    phases = np.stack(locomotion_phase)
    arrays = {
        "qpos": np.stack(qpos).astype(np.float32),
        "qvel": np.stack(qvel).astype(np.float32),
        "walk_previous_action": np.stack(walk_previous_action).astype(np.float32),
        "walk_previous_action_valid": np.asarray(
            walk_previous_action_valid, dtype=np.uint8
        ),
        "rollout_id": rows.astype(np.int32),
        "phase_bucket": phase_buckets(phases, args.phase_buckets),
        "split": server["split"][rows].astype(np.uint8),
        "source_match_id": server["match_id"][rows].astype(np.int32),
        "source_player_number": server["player_number"][rows].astype(np.int32),
        "source_time_s": server["time_s"][rows].astype(np.float64),
        "source_ball_local_xy_m": ball_local_xy(server)[rows].astype(np.float32),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    manifest = {
        "schema_version": 1,
        "purpose": "kick_policy_v3_walk_to_kick_transition_corpus",
        "promotable": False,
        "source_state": "quantized_server_observation_projected_into_single_T1_exact_CPU_scene",
        "source_corpus": str(args.server_corpus.resolve()),
        "source_corpus_sha256": sha256(args.server_corpus),
        "contract": str(CONTRACT.resolve()), "contract_sha256": sha256(CONTRACT),
        "teacher_condition_index": args.teacher_condition_index,
        "candidate_frames": candidate_frames,
        "entries": int(rows.size),
        "train_entries": int(np.count_nonzero(arrays["split"] == 0)),
        "validation_entries": int(np.count_nonzero(arrays["split"] == 1)),
        "walk_previous_action_valid_entries": int(
            np.count_nonzero(arrays["walk_previous_action_valid"])
        ),
        "split_unit": "source_match",
        "npz": str(output_npz), "npz_sha256": sha256(output_npz),
    }
    output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "entries", "train_entries", "validation_entries",
        "walk_previous_action_valid_entries", "npz",
    )}, indent=2))


if __name__ == "__main__":
    main()
