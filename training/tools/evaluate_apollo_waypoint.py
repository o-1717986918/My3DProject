#!/usr/bin/env python3
"""Evaluate an Apollo-compatible ONNX actor on closed-loop waypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.apollo_run_env import ApolloWaypointRun
from my3d_rl.apollo_walk_jax import load_apollo_walk_jax


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_MODEL = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo_rebuild"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)


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


def _keep_active(old: Any, new: Any, active: jax.Array) -> Any:
    if (
        not hasattr(new, "ndim")
        or new.ndim == 0
        or new.shape[0] != active.shape[0]
    ):
        return new
    mask = active.reshape(active.shape + (1,) * (new.ndim - active.ndim))
    return jp.where(mask, new, old)


def _bearing_group(bearing: np.ndarray) -> dict[str, np.ndarray]:
    absolute = np.abs(bearing)
    return {
        "front_0_45": absolute <= np.pi / 4.0,
        "oblique_45_90": (absolute > np.pi / 4.0) & (absolute <= np.pi / 2.0),
        "rear_oblique_90_135": (absolute > np.pi / 2.0) & (absolute <= 3.0 * np.pi / 4.0),
        "rear_135_180": absolute > 3.0 * np.pi / 4.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_261_403)
    parser.add_argument("--distance-min", type=float, default=2.0)
    parser.add_argument("--distance-max", type=float, default=6.0)
    parser.add_argument("--bearing-min-deg", type=float, default=-180.0)
    parser.add_argument("--bearing-max-deg", type=float, default=180.0)
    args = parser.parse_args()
    if (
        not args.model.is_file()
        or args.num_envs < 4
        or args.steps < 1
        or args.seed < 0
        or not 0.25 < args.distance_min <= args.distance_max
        or not -180.0 <= args.bearing_min_deg < args.bearing_max_deg <= 180.0
    ):
        raise ValueError("waypoint evaluation arguments are invalid")
    run_dir = _external_new_directory(args.run_dir)
    env = ApolloWaypointRun(
        config_overrides={
            "impl": args.impl,
            "episode_length": args.steps,
            "naconmax": max(2048, 16 * args.num_envs),
            "waypoint_distance_range": [args.distance_min, args.distance_max],
            "waypoint_bearing_range": [
                float(np.deg2rad(args.bearing_min_deg)),
                float(np.deg2rad(args.bearing_max_deg)),
            ],
            "reset_joint_noise": 0.0,
            "reset_joint_velocity_noise": 0.0,
            "reset_policy_action_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "push_enable": False,
            "action_delay_max_steps": 0,
        }
    )
    policy = load_apollo_walk_jax(args.model)
    reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.vmap(env.step)
    initial = reset(jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs))

    def scan_step(carry, unused):
        state, active = carry
        action = policy(state.obs["state"])
        candidate = batched_step(state, action)
        success = active & (
            candidate.metrics["diagnostic/waypoint_success"] > 0.5
        )
        fallen = active & (candidate.metrics["cost/fall"] > 0.5)
        next_active = active & ~success & ~fallen
        next_state = jax.tree.map(
            lambda old, new: _keep_active(old, new, active), state, candidate
        )
        record = {
            "success": success,
            "fallen": fallen,
            "active": active,
            "distance": candidate.metrics["diagnostic/waypoint_distance_m"],
            "height": candidate.metrics["diagnostic/torso_height"],
        }
        return (next_state, next_active), record

    (final, active), rollout = jax.jit(
        lambda state: jax.lax.scan(
            scan_step,
            (state, jp.ones(args.num_envs, dtype=bool)),
            xs=None,
            length=args.steps,
        )
    )(initial)
    arrays = jax.tree.map(np.asarray, rollout)
    success = np.any(arrays["success"], axis=0)
    fallen = np.any(arrays["fallen"], axis=0)
    timeout = np.asarray(active)
    first_success_step = np.argmax(arrays["success"], axis=0) + 1
    success_time = first_success_step[success] * env.dt
    bearing = np.asarray(initial.info["waypoint_bearing"])
    final_distance = np.asarray(final.info["waypoint_distance"])
    minimum_height = np.min(arrays["height"], axis=0)

    def group_summary(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.sum(mask))
        reached = success & mask
        return {
            "episodes": count,
            "successes": int(np.sum(reached)),
            "success_rate": float(np.mean(success[mask])) if count else None,
            "falls": int(np.sum(fallen & mask)),
            "median_final_distance_m": (
                float(np.median(final_distance[mask])) if count else None
            ),
            "median_success_time_s": (
                float(np.median(first_success_step[reached] * env.dt))
                if np.any(reached)
                else None
            ),
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "apollo_closed_loop_waypoint_evaluation",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip(),
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "backend": jax.default_backend(),
        "implementation": args.impl,
        "seed": args.seed,
        "episodes": args.num_envs,
        "steps": args.steps,
        "duration_s": args.steps * env.dt,
        "distance_range_m": [args.distance_min, args.distance_max],
        "bearing_range_deg": [args.bearing_min_deg, args.bearing_max_deg],
        "successes": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "falls": int(np.sum(fallen)),
        "fall_rate": float(np.mean(fallen)),
        "timeouts": int(np.sum(timeout)),
        "median_success_time_s": (
            float(np.median(success_time)) if success_time.size else None
        ),
        "median_final_distance_m": float(np.median(final_distance)),
        "minimum_torso_height_m": float(np.min(minimum_height)),
        "median_minimum_torso_height_m": float(np.median(minimum_height)),
        "bearing_groups": {
            name: group_summary(mask)
            for name, mask in _bearing_group(bearing).items()
        },
    }
    run_dir.mkdir(parents=True)
    report_path = run_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
