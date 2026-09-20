#!/usr/bin/env python3
"""Collect kick entry states from exact-CPU replays of real match Walk commands.

Logged commands and the first local ball position are real server observations.
Joint/ball trajectories thereafter are reconstructed in a one-robot MuJoCo
scene; they are *not* direct server-state measurements or success labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from my3d_rl.apollo_walk_cpu import ApolloWalkCpu
from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_transition import estimate_locomotion_phase
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.t1_control import APOLLO_DEFAULT_POSE, apollo_joint_gains
from tools.evaluate_apollo_match_trace import (
    DEFAULT_CONTRACT,
    DEFAULT_MODEL,
    expand_trace,
    load_match_traces,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
KICK_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "kick_policy_v3.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_windows(
    expanded: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    episodes: int,
    steps: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Round-robin source traces, sampling only fresh near-ball starts."""
    if episodes < 1 or steps < 1:
        raise ValueError("episodes and steps must be positive")
    starts: dict[int, np.ndarray] = {}
    for index, (_, commands, _, near) in enumerate(expanded):
        if commands.shape[0] >= steps:
            candidates = np.flatnonzero(near[: commands.shape[0] - steps + 1])
            if candidates.size:
                starts[index] = candidates
    if not starts:
        raise ValueError("match logs contain no complete fresh near-ball Walk window")
    rng = np.random.default_rng(seed)
    trace_ids = rng.permutation(list(starts))
    windows: list[tuple[int, int]] = []
    for index in range(episodes):
        trace_id = int(trace_ids[index % len(trace_ids)])
        windows.append((trace_id, int(rng.choice(starts[trace_id]))))
    return windows


def grouped_split(trace_ids: np.ndarray, *, seed: int) -> np.ndarray:
    """Never split adjacent windows of the same contiguous source trace."""
    ids = np.asarray(trace_ids, dtype=np.int32)
    if ids.ndim != 1 or ids.size < 1:
        raise ValueError("trace IDs must be a nonempty vector")
    unique = np.unique(ids)
    if unique.size < 2:
        return np.zeros(ids.shape, dtype=np.uint8)
    order = np.random.default_rng(seed).permutation(unique)
    validation_count = min(max(round(unique.size * 0.2), 1), unique.size - 1)
    return np.isin(ids, order[:validation_count]).astype(np.uint8)


def release_like_geometry(ball_local_xyz: np.ndarray) -> np.ndarray:
    """Diagnostic only: count plausible contact entries without authorizing kicks."""
    positions = np.asarray(ball_local_xyz, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("ball positions must have shape [N, 3]")
    return (
        (positions[:, 0] >= 0.20)
        & (positions[:, 0] <= 0.65)
        & (np.abs(positions[:, 1]) <= 0.25)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--entry-corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--team-prefix", default="Apollo-Rebuild")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_920)
    args = parser.parse_args()
    if (
        not args.match_dir.is_dir()
        or not args.entry_corpus.is_file()
        or not args.model.is_file()
        or args.episodes < 1
        or args.steps < 1
        or args.capture_stride < 1
        or args.seed < 0
    ):
        raise ValueError("match handoff arguments are invalid")
    run_dir = args.run_dir.resolve()
    if (
        not args.run_dir.is_absolute()
        or run_dir.is_relative_to(REPOSITORY_ROOT.resolve())
        or run_dir.exists()
    ):
        raise ValueError("run-dir must be a new absolute directory outside the repo")

    walk_contract = load_policy_contract(DEFAULT_CONTRACT)
    kick_contract = load_policy_contract(KICK_CONTRACT)
    if walk_contract.joint_order != kick_contract.joint_order:
        raise ValueError("Walk and kick contracts use different joint orders")
    traces = load_match_traces(args.match_dir, args.team_prefix)
    expanded = [(name, *expand_trace(frames)) for name, frames in traces]
    windows = choose_windows(
        expanded, episodes=args.episodes, steps=args.steps, seed=args.seed
    )
    scene = RcssKickScene(walk_contract)
    actor = ApolloWalkCpu(
        scene.model, walk_contract, prefix=scene.prefix, policy_path=args.model
    )
    joint_qpos = np.asarray(
        [scene.model.joint(scene.prefix + name).qposadr[0] for name in walk_contract.joint_order]
    )
    joint_dof = np.asarray(
        [scene.model.joint(scene.prefix + name).dofadr[0] for name in walk_contract.joint_order]
    )
    torso_site = scene.model.site(scene.prefix + "torso").id
    torso_body = scene.model.body(scene.prefix + "torso").id
    root_dof = scene.model.joint(scene.prefix + "root").dofadr[0]
    ball_qpos = scene.model.joint("ball-root").qposadr[0]
    ball_dof = scene.model.joint("ball-root").dofadr[0]
    gains = np.asarray(
        [apollo_joint_gains(name) for name in walk_contract.joint_order],
        dtype=np.float64,
    )
    with np.load(args.entry_corpus, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        previous = np.asarray(archive["last_action"], dtype=np.float64)
        source_command = np.asarray(archive["source_command"], dtype=np.float64)
    count = qpos.shape[0]
    if (
        count < 1
        or qpos.shape != (count, scene.model.nq)
        or qvel.shape != (count, scene.model.nv)
        or previous.shape != (count, walk_contract.action_size)
        or source_command.shape != (count, 3)
        or not all(np.isfinite(array).all() for array in (qpos, qvel, previous, source_command))
    ):
        raise ValueError("entry corpus does not match the exact RCSS T1 scene")

    rng = np.random.default_rng(args.seed + 1)
    records: list[dict[str, np.ndarray | float | int]] = []
    episodes_with_capture = 0
    replay_falls = 0
    for episode, (trace_id, start) in enumerate(windows):
        _, commands, balls, _ = expanded[trace_id]
        episode_commands = commands[start : start + args.steps]
        local_ball = balls[start]
        distances = np.sum(np.square(source_command - episode_commands[0]), axis=1)
        nearest_count = min(32, count)
        nearest = np.argpartition(distances, nearest_count - 1)[:nearest_count]
        entry_index = int(rng.choice(nearest))
        scene.data.qpos[:] = qpos[entry_index]
        scene.data.qvel[:] = qvel[entry_index]
        scene.data.time = 0.0
        scene.data.ctrl[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        torso_xy = scene.data.site_xpos[torso_site, :2]
        scene.data.qpos[ball_qpos : ball_qpos + 3] = [
            torso_xy[0] + c * local_ball[0] - s * local_ball[1],
            torso_xy[1] + s * local_ball[0] + c * local_ball[1],
            0.11,
        ]
        scene.data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        scene.data.qvel[ball_dof : ball_dof + 6] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        last_action = previous[entry_index].copy()
        captured = False
        for step, command in enumerate(episode_commands):
            rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            c, s = math.cos(yaw), math.sin(yaw)
            torso = scene.data.site_xpos[torso_site].copy()
            ball = scene.data.qpos[ball_qpos : ball_qpos + 3]
            local_xy = np.array(
                [c * (ball[0] - torso[0]) + s * (ball[1] - torso[1]),
                 -s * (ball[0] - torso[0]) + c * (ball[1] - torso[1])]
            )
            height = float(scene.data.xpos[torso_body, 2])
            upright = float(rotation[2, 2])
            if step % args.capture_stride == 0 and (
                0.15 <= np.linalg.norm(local_xy) <= 1.10
                and height >= 0.45
                and upright >= 0.75
            ):
                joint_offset = scene.data.qpos[joint_qpos] - APOLLO_DEFAULT_POSE
                joint_velocity = scene.data.qvel[joint_dof]
                phase = estimate_locomotion_phase(
                    joint_offset, joint_velocity, walk_contract.joint_order
                )
                records.append({
                    "qpos": scene.data.qpos.copy(),
                    "qvel": scene.data.qvel.copy(),
                    "joint_position_offset": joint_offset.copy(),
                    "joint_velocity": joint_velocity.copy(),
                    "walk_previous_action": last_action.copy(),
                    "setup_velocity_command": command.copy(),
                    "locomotion_phase": phase.sin_cos.copy(),
                    "support_hint": phase.support_hint.copy(),
                    "phase_magnitude_rad": phase.magnitude_rad,
                    "ball_position_local_m": np.array([*local_xy, ball[2] - torso[2]]),
                    "root_velocity": scene.data.qvel[root_dof : root_dof + 6].copy(),
                    "torso_height_m": height,
                    "upright": upright,
                    "rollout_id": episode,
                    "source_trace_id": trace_id,
                    "source_step": start + step,
                    "initial_command_distance": float(math.sqrt(distances[entry_index])),
                })
                captured = True
            targets, action = actor.target(scene.data, last_action, command)
            targets[:2] = scene.data.qpos[joint_qpos[:2]]
            scene.step_joint_targets(targets, kp=gains[:, 0], kd=gains[:, 1])
            last_action = action
            new_rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
            if scene.data.xpos[torso_body, 2] < 0.35 or new_rotation[2, 2] < 0.20:
                replay_falls += 1
                break
        episodes_with_capture += int(captured)
    if not records:
        raise RuntimeError("replays produced no upright near-ball kick entry states")

    arrays = {
        key: np.asarray([record[key] for record in records], dtype=np.float32)
        for key in records[0]
        if key not in {"rollout_id", "source_trace_id", "source_step"}
    }
    for key in ("rollout_id", "source_trace_id", "source_step"):
        arrays[key] = np.asarray([record[key] for record in records], dtype=np.int32)
    arrays["split"] = grouped_split(arrays["source_trace_id"], seed=args.seed + 2)
    arrays["release_like_geometry"] = release_like_geometry(
        arrays["ball_position_local_m"]
    ).astype(np.uint8)
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("replayed transition corpus contains non-finite values")
    run_dir.mkdir(parents=True)
    npz_path = run_dir / "match-kick-handoff.npz"
    np.savez_compressed(npz_path, **arrays)
    source_paths = sorted(args.match_dir.glob(f"{args.team_prefix}-*.log"))
    report = {
        "schema_version": 1,
        "purpose": "kick_policy_v3_match_command_replay_handoff_corpus",
        "status": "complete",
        "promotable": False,
        "provenance": "exact_CPU_replay_of_logged_Walk_commands_not_server_joint_telemetry",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "walk_model": str(args.model.resolve()),
        "walk_model_sha256": _sha256(args.model),
        "walk_contract_sha256": _sha256(DEFAULT_CONTRACT),
        "kick_contract_sha256": _sha256(KICK_CONTRACT),
        "entry_corpus": str(args.entry_corpus.resolve()),
        "entry_corpus_sha256": _sha256(args.entry_corpus),
        "source_logs": {str(path.resolve()): _sha256(path) for path in source_paths},
        "source_trace_catalog": [
            {
                "trace_id": index,
                "log": name,
                "first_status_time_s": frames[0].time_s,
                "last_status_time_s": frames[-1].time_s,
            }
            for index, (name, frames) in enumerate(traces)
        ],
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "replay_steps": args.steps,
        "capture_stride": args.capture_stride,
        "replay_falls": replay_falls,
        "episodes_with_capture": episodes_with_capture,
        "samples": len(records),
        "source_trace_groups": int(np.unique(arrays["source_trace_id"]).size),
        "train_samples": int(np.sum(arrays["split"] == 0)),
        "validation_samples": int(np.sum(arrays["split"] == 1)),
        "release_like_geometry": {
            "diagnostic_only": True,
            "ball_local_x_m": [0.20, 0.65],
            "absolute_ball_local_y_max_m": 0.25,
            "samples": int(np.sum(arrays["release_like_geometry"])),
            "unique_rollouts": int(
                np.unique(
                    arrays["rollout_id"][arrays["release_like_geometry"] == 1]
                ).size
            ),
            "source_trace_groups": int(
                np.unique(
                    arrays["source_trace_id"][arrays["release_like_geometry"] == 1]
                ).size
            ),
            "validation_samples": int(
                np.sum(
                    (arrays["release_like_geometry"] == 1)
                    & (arrays["split"] == 1)
                )
            ),
        },
        "corpus": str(npz_path.resolve()),
        "corpus_sha256": _sha256(npz_path),
        "ball_position_policy": "logged_local_position_at_start_then_physics_only",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
