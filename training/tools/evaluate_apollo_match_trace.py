#!/usr/bin/env python3
"""Replay real Apollo match command sequences on an Apollo-compatible actor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from my3d_rl.apollo_walk_cpu import ApolloWalkCpu
from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.t1_control import apollo_joint_gains


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "training" / "contracts" / "apollo_walk_policy_v1.yaml"
)
DEFAULT_MODEL = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo_rebuild"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)
POLICY_PERIOD_S = 0.02


@dataclass(frozen=True)
class TraceFrame:
    time_s: float
    command: np.ndarray
    ball_local_xy: np.ndarray
    near_ball: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_new_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("run-dir must be absolute")
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("run-dir must stay outside the repository")
    if resolved.exists():
        raise FileExistsError(f"run-dir already exists: {resolved}")
    return resolved


def _fields(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def _normalise_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def velocity_command_from_status(values: dict[str, str]) -> np.ndarray:
    """Reproduce WalkRunner::compute_velocity_command from one status line."""

    self_yaw_deg = float(values.get("yaw_deg", values.get("self_yaw", "nan")))
    target_x = float(values["walk_target_x"])
    target_y = float(values["walk_target_y"])
    if values.get("walk_target_absolute") == "1":
        world_x = target_x - float(values["x"])
        world_y = target_y - float(values["y"])
        yaw = math.radians(-self_yaw_deg)
        velocity_x = math.cos(yaw) * world_x - math.sin(yaw) * world_y
        velocity_y = math.sin(yaw) * world_x + math.cos(yaw) * world_y
    else:
        velocity_x = target_x
        velocity_y = target_y

    orientation_present = values.get(
        "walk_orientation_present", values.get("walk_orientation_set", "0")
    ) == "1"
    if not orientation_present:
        relative_orientation_deg = (
            math.degrees(math.atan2(velocity_y, velocity_x))
            if math.hypot(velocity_x, velocity_y) > 0.1
            else 0.0
        )
    else:
        orientation_deg = float(
            values["walk_orientation_deg"]
            if "walk_orientation_deg" in values
            else values["walk_orientation"]
        )
        if values.get("walk_orientation_absolute") == "1":
            relative_orientation_deg = _normalise_degrees(
                orientation_deg - self_yaw_deg
            )
        else:
            relative_orientation_deg = _normalise_degrees(orientation_deg)
    command = np.array(
        [
            np.clip(velocity_x, -0.5, 1.0),
            np.clip(velocity_y, -0.5, 0.5),
            np.clip(0.2 * math.radians(relative_orientation_deg), -0.5, 0.5),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(command).all():
        raise ValueError("status line produced a non-finite Walk command")
    return command


def _frame_from_status(values: dict[str, str]) -> TraceFrame:
    self_yaw = math.radians(
        float(values.get("yaw_deg", values.get("self_yaw", "nan")))
    )
    world_ball = np.array(
        [
            float(values["ball_x"]) - float(values["x"]),
            float(values["ball_y"]) - float(values["y"]),
        ]
    )
    c, s = math.cos(self_yaw), math.sin(self_yaw)
    ball_local = np.array(
        [
            c * world_ball[0] + s * world_ball[1],
            -s * world_ball[0] + c * world_ball[1],
        ],
        dtype=np.float32,
    )
    ball_distance = float(values.get("ball_dist", np.linalg.norm(world_ball)))
    time_value = values["t"] if "t" in values else values["server_time"]
    return TraceFrame(
        time_s=float(time_value),
        command=velocity_command_from_status(values),
        ball_local_xy=ball_local,
        near_ball=math.isfinite(ball_distance) and ball_distance <= 1.10,
    )


def load_match_traces(
    match_dir: Path,
    team_prefix: str,
    *,
    maximum_gap_s: float = 0.20,
) -> list[tuple[str, list[TraceFrame]]]:
    """Load contiguous PlayOn Walk traces from per-player runtime logs."""

    traces: list[tuple[str, list[TraceFrame]]] = []
    paths = sorted(match_dir.glob(f"{team_prefix}-*.log"))
    if not paths:
        raise FileNotFoundError(
            f"no {team_prefix}-*.log files found in {match_dir}"
        )
    for path in paths:
        active: list[TraceFrame] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("APOLLO_REBUILD_STATUS "):
                continue
            values = _fields(line)
            eligible = values.get("mode") == "4" and values.get("motion") == "Walk"
            if eligible:
                frame = _frame_from_status(values)
                if active and frame.time_s - active[-1].time_s > maximum_gap_s:
                    if len(active) >= 2:
                        traces.append((path.name, active))
                    active = []
                active.append(frame)
            else:
                if len(active) >= 2:
                    traces.append((path.name, active))
                active = []
        if len(active) >= 2:
            traces.append((path.name, active))
    if not traces:
        raise ValueError("match logs contain no contiguous PlayOn Walk trace")
    return traces


def expand_trace(
    frames: list[TraceFrame],
    *,
    policy_period_s: float = POLICY_PERIOD_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold each status command until the next timestamp at policy frequency."""

    if len(frames) < 2 or policy_period_s <= 0.0:
        raise ValueError("trace expansion requires two frames and a positive period")
    intervals = np.diff([frame.time_s for frame in frames])
    if np.any(intervals <= 0.0):
        raise ValueError("trace timestamps must be strictly increasing")
    final_interval = float(np.median(intervals))
    commands: list[np.ndarray] = []
    balls: list[np.ndarray] = []
    near_ball: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        interval = intervals[index] if index < len(intervals) else final_interval
        hold_steps = max(1, int(round(float(interval) / policy_period_s)))
        commands.append(np.repeat(frame.command[None, :], hold_steps, axis=0))
        balls.append(np.repeat(frame.ball_local_xy[None, :], hold_steps, axis=0))
        near_ball.append(np.full(hold_steps, frame.near_ball, dtype=bool))
    return (
        np.concatenate(commands),
        np.concatenate(balls),
        np.concatenate(near_ball),
    )


def _local_velocity(scene: RcssKickScene, root_dof: int, torso_site: int) -> np.ndarray:
    rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    world = scene.data.qvel[root_dof : root_dof + 3]
    return np.array(
        [c * world[0] + s * world[1], -s * world[0] + c * world[1]],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--team-prefix", default="Apollo-Rebuild")
    parser.add_argument("--entry-corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=28)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20_261_426)
    parser.add_argument(
        "--near-ball-fraction",
        type=float,
        default=0.25,
        help="fraction of episodes forced to start in a logged <=1.1 m ball state",
    )
    args = parser.parse_args()
    if (
        not args.model.is_file()
        or not args.match_dir.is_dir()
        or not args.entry_corpus.is_file()
        or args.episodes < 1
        or args.steps < 1
        or args.seed < 0
        or not 0.0 <= args.near_ball_fraction <= 1.0
    ):
        raise ValueError("match trace evaluation arguments are invalid")
    run_dir = _external_new_directory(args.run_dir)
    traces = load_match_traces(args.match_dir, args.team_prefix)
    expanded = [
        (name, *expand_trace(frames))
        for name, frames in traces
    ]
    eligible = [trace for trace in expanded if trace[1].shape[0] >= args.steps]
    if not eligible:
        raise ValueError("no match trace is long enough for the requested steps")
    near_starts = [
        (trace_index, int(start))
        for trace_index, (_, commands, _, near) in enumerate(eligible)
        for start in np.flatnonzero(
            near[: commands.shape[0] - args.steps + 1]
        )
    ]
    near_episode_count = round(args.episodes * args.near_ball_fraction)
    if near_episode_count and not near_starts:
        raise ValueError("match traces have no near-ball start for requested steps")

    contract = load_policy_contract(DEFAULT_CONTRACT)
    scene = RcssKickScene(contract)
    actor = ApolloWalkCpu(
        scene.model,
        contract,
        prefix=scene.prefix,
        policy_path=args.model,
    )
    joint_qpos = np.asarray(
        [scene.model.joint(scene.prefix + name).qposadr[0] for name in contract.joint_order]
    )
    torso_site = scene.model.site(scene.prefix + "torso").id
    torso_body = scene.model.body(scene.prefix + "torso").id
    root_dof = scene.model.joint(scene.prefix + "root").dofadr[0]
    ball_qpos = scene.model.joint("ball-root").qposadr[0]
    ball_dof = scene.model.joint("ball-root").dofadr[0]
    gyro = scene.model.sensor(scene.prefix + "torso_gyro")
    gyro_z = gyro.adr[0] + 2
    gains = np.asarray(
        [apollo_joint_gains(name) for name in contract.joint_order],
        dtype=np.float64,
    )
    with np.load(args.entry_corpus, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        previous = np.asarray(archive["last_action"], dtype=np.float64)
        source_command = np.asarray(archive["source_command"], dtype=np.float64)
    sample_count = qpos.shape[0]
    if (
        qpos.shape != (sample_count, scene.model.nq)
        or qvel.shape != (sample_count, scene.model.nv)
        or previous.shape != (sample_count, contract.action_size)
        or source_command.shape != (sample_count, 3)
        or sample_count < 1
        or not all(
            np.isfinite(array).all()
            for array in (qpos, qvel, previous, source_command)
        )
    ):
        raise ValueError("entry corpus is incompatible with the RCSS T1 model")

    rng = np.random.default_rng(args.seed)
    falls = np.zeros(args.episodes, dtype=bool)
    completed_steps = np.zeros(args.episodes, dtype=np.int32)
    linear_squared_error = np.zeros(args.episodes, dtype=np.float64)
    yaw_squared_error = np.zeros(args.episodes, dtype=np.float64)
    displacement = np.zeros(args.episodes, dtype=np.float64)
    minimum_height = np.full(args.episodes, np.inf, dtype=np.float64)
    ball_displacement = np.zeros(args.episodes, dtype=np.float64)
    near_ball_fraction = np.zeros(args.episodes, dtype=np.float64)
    initial_near_ball = np.zeros(args.episodes, dtype=bool)
    selected: list[dict[str, object]] = []
    for episode in range(args.episodes):
        if episode < near_episode_count:
            trace_index, start = near_starts[
                int(rng.integers(0, len(near_starts)))
            ]
            name, commands, balls, near = eligible[trace_index]
        else:
            name, commands, balls, near = eligible[episode % len(eligible)]
            start = int(rng.integers(0, commands.shape[0] - args.steps + 1))
        episode_commands = commands[start : start + args.steps]
        episode_balls = balls[start : start + args.steps]
        episode_near = near[start : start + args.steps]
        distances = np.sum(
            np.square(source_command - episode_commands[0]), axis=1
        )
        nearest_count = min(32, sample_count)
        nearest = np.argpartition(distances, nearest_count - 1)[:nearest_count]
        entry_index = int(nearest[int(rng.integers(0, nearest_count))])

        scene.data.qpos[:] = qpos[entry_index]
        scene.data.qvel[:] = qvel[entry_index]
        scene.data.time = 0.0
        scene.data.ctrl[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        local_ball = episode_balls[0]
        torso_xy = scene.data.site_xpos[torso_site, :2]
        scene.data.qpos[ball_qpos : ball_qpos + 3] = [
            torso_xy[0] + c * local_ball[0] - s * local_ball[1],
            torso_xy[1] + s * local_ball[0] + c * local_ball[1],
            0.11,
        ]
        scene.data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        scene.data.qvel[ball_dof : ball_dof + 6] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        initial_xy = scene.data.site_xpos[torso_site, :2].copy()
        initial_ball_xy = scene.data.qpos[ball_qpos : ball_qpos + 2].copy()
        last_action = previous[entry_index].copy()
        sum_linear_error = 0.0
        sum_yaw_error = 0.0
        simulated = 0
        for command in episode_commands:
            targets, action = actor.target(scene.data, last_action, command)
            # The real runner gives both head joints to its visual tracker.
            targets[:2] = scene.data.qpos[joint_qpos[:2]]
            scene.step_joint_targets(targets, kp=gains[:, 0], kd=gains[:, 1])
            last_action = action
            local_velocity = _local_velocity(scene, root_dof, torso_site)
            sum_linear_error += float(np.sum(np.square(local_velocity - command[:2])))
            sum_yaw_error += float((scene.data.sensordata[gyro_z] - command[2]) ** 2)
            rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
            height = float(scene.data.xpos[torso_body, 2])
            upright = float(rotation[2, 2])
            minimum_height[episode] = min(minimum_height[episode], height)
            simulated += 1
            if height < 0.35 or upright < 0.20:
                falls[episode] = True
                break
        completed_steps[episode] = simulated
        linear_squared_error[episode] = sum_linear_error / simulated
        yaw_squared_error[episode] = sum_yaw_error / simulated
        displacement[episode] = float(
            np.linalg.norm(scene.data.site_xpos[torso_site, :2] - initial_xy)
        )
        ball_displacement[episode] = float(
            np.linalg.norm(scene.data.qpos[ball_qpos : ball_qpos + 2] - initial_ball_xy)
        )
        near_ball_fraction[episode] = float(np.mean(episode_near))
        initial_near_ball[episode] = bool(episode_near[0])
        selected.append(
            {
                "trace": name,
                "expanded_start_step": start,
                "entry_index": entry_index,
                "initial_command_distance": float(math.sqrt(distances[entry_index])),
                "initial_near_ball": bool(initial_near_ball[episode]),
            }
        )

    source_paths = sorted(args.match_dir.glob(f"{args.team_prefix}-*.log"))
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "apollo_actor_real_match_command_trace_replay",
        "promotable": False,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip(),
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "match_dir": str(args.match_dir.resolve()),
        "source_logs": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "entry_corpus": str(args.entry_corpus.resolve()),
        "entry_corpus_sha256": _sha256(args.entry_corpus),
        "seed": args.seed,
        "episodes": args.episodes,
        "steps_requested": args.steps,
        "duration_requested_s": args.steps * POLICY_PERIOD_S,
        "requested_near_ball_fraction": args.near_ball_fraction,
        "status_traces": len(traces),
        "eligible_traces": len(eligible),
        "falls": int(np.sum(falls)),
        "fall_rate": float(np.mean(falls)),
        "median_survival_s": float(np.median(completed_steps) * POLICY_PERIOD_S),
        "linear_velocity_rmse_mps": float(np.sqrt(np.mean(linear_squared_error))),
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(yaw_squared_error))),
        "median_displacement_m": float(np.median(displacement)),
        "median_ball_displacement_m": float(np.median(ball_displacement)),
        "maximum_ball_displacement_m": float(np.max(ball_displacement)),
        "episodes_with_ball_motion_over_0_05_m": int(
            np.sum(ball_displacement > 0.05)
        ),
        "mean_trace_near_ball_fraction": float(np.mean(near_ball_fraction)),
        "initial_near_ball_episodes": int(np.sum(initial_near_ball)),
        "initial_near_ball_falls": int(np.sum(falls & initial_near_ball)),
        "initial_near_ball_median_ball_displacement_m": (
            float(np.median(ball_displacement[initial_near_ball]))
            if np.any(initial_near_ball)
            else None
        ),
        "minimum_torso_height_m": float(np.min(minimum_height)),
        "episodes_detail": selected,
    }
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
