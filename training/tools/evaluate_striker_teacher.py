#!/usr/bin/env python3
"""Evaluate a closed-loop striker prior or teacher checkpoint in MJX/Warp."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.striker_env import DEFAULT_CONTRACT, LongHorizonStriker
from tools.train_striker_teacher import STAGES, _load_kick_prior


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _keep_active(old, new, active: jax.Array):
    if (
        not hasattr(new, "ndim")
        or new.ndim == 0
        or new.shape[0] != active.shape[0]
    ):
        return new
    mask = active.reshape(active.shape + (1,) * (new.ndim - active.ndim))
    return jp.where(mask, new, old)


def _first_event(event: np.ndarray, horizon: int) -> np.ndarray:
    happened = event.any(axis=0)
    return np.where(happened, event.argmax(axis=0), horizon)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kick_prior_manifest", type=Path)
    parser.add_argument("--kick-prior-condition-index", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--stage", choices=tuple(STAGES), default="closed_loop"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11_203)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--kick-trigger-threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_rollouts < 1 or args.episode_length < 1:
        raise ValueError("rollout count and episode length must be positive")
    if not 0.0 < args.kick_trigger_threshold <= 1.0:
        raise ValueError("kick trigger threshold must be in (0, 1]")
    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be an absolute path outside the repository")

    contract = load_policy_contract(args.contract)
    prior, prior_metadata = _load_kick_prior(
        args.kick_prior_manifest,
        contract,
        condition_index=args.kick_prior_condition_index,
    )
    overrides = {
        **STAGES[args.stage],
        "impl": args.impl,
        "episode_length": args.episode_length,
        "naconmax": max(2048, 16 * args.num_rollouts),
        "fixed_action_mode": 0,
        "fixed_desired_arrival_speed": 0.8,
        "kick_trigger_threshold": args.kick_trigger_threshold,
    }
    env = LongHorizonStriker(
        config_overrides=overrides,
        contract=contract,
        kick_prior_joint_residuals=prior,
    )
    policy = (
        ppo_checkpoint.load_policy(args.checkpoint)
        if args.checkpoint is not None
        else None
    )
    reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.vmap(env.step)
    state = reset(
        jax.random.split(jax.random.PRNGKey(args.seed), args.num_rollouts)
    )

    def scan_step(carry, key):
        current, already_done = carry
        if policy is None:
            actions = jp.zeros((args.num_rollouts, env.action_size))
        else:
            actions, _ = policy(current.obs, key)
        candidate = batched_step(current, actions)
        active = ~already_done
        next_state = jax.tree.map(
            lambda old, new: _keep_active(old, new, active), current, candidate
        )
        newly_done = active & candidate.done.astype(bool)
        event = jp.stack(
            [
                candidate.metrics["event/contact"],
                candidate.metrics["event/kick_trigger"],
                candidate.metrics["event/success"],
                candidate.metrics["cost/fall"],
            ],
            axis=-1,
        ) * active[:, None]
        return (next_state, already_done | newly_done), event

    keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.episode_length)
    (final_state, done), events = jax.jit(
        lambda initial: jax.lax.scan(
            scan_step,
            (initial, jp.zeros(args.num_rollouts, dtype=bool)),
            keys,
        )
    )(state)
    events_np = np.asarray(events)
    contact = events_np[:, :, 0] > 0
    trigger = events_np[:, :, 1] > 0
    success = events_np[:, :, 2] > 0
    fall = events_np[:, :, 3] > 0
    first_success = _first_event(success, args.episode_length)
    first_fall = _first_event(fall, args.episode_length)
    first_done = np.minimum(first_success, first_fall)
    episode_steps = np.where(
        first_done < args.episode_length,
        first_done + 1,
        args.episode_length,
    )
    contacted = contact.any(axis=0)
    triggered = trigger.any(axis=0)
    succeeded = success.any(axis=0)
    fallen = first_fall < first_success
    final_contact_distance = np.asarray(
        final_state.metrics["diagnostic/contact_distance"]
    )
    final_heading_error = np.asarray(
        final_state.metrics["diagnostic/heading_error"]
    )
    final_activation = np.asarray(
        final_state.metrics["diagnostic/kick_activation"]
    )
    final_goal_distance = np.asarray(
        final_state.metrics["diagnostic/goal_distance"]
    )
    maximum_directional_speed = np.asarray(
        final_state.info["maximum_directional_speed"]
    )
    summary: dict[str, Any] = {
        "rollouts": args.num_rollouts,
        "triggered": int(triggered.sum()),
        "contacted": int(contacted.sum()),
        "succeeded": int(succeeded.sum()),
        "fallen": int(fallen.sum()),
        "trigger_rate": float(triggered.mean()),
        "contact_rate": float(contacted.mean()),
        "success_rate": float(succeeded.mean()),
        "fall_rate": float(fallen.mean()),
        "mean_episode_steps": float(episode_steps.mean()),
        "maximum_directional_speed_mean_mps": float(
            maximum_directional_speed.mean()
        ),
        "maximum_directional_speed_max_mps": float(
            maximum_directional_speed.max()
        ),
        "goal_distance_mean_m": float(
            final_goal_distance.mean()
        ),
        "gate_passed": bool(
            succeeded.mean() >= 0.90 and not fallen.any()
        ),
    }
    rollout_records = []
    initial_fields = {
        name: np.asarray(final_state.info[name])
        for name in (
            "initial_robot_distance",
            "initial_robot_lateral",
            "initial_robot_yaw_error",
            "initial_target_angle",
            "initial_target_distance",
        )
    }
    for index in range(args.num_rollouts):
        rollout_records.append(
            {
                "index": index,
                **{
                    name: float(values[index])
                    for name, values in initial_fields.items()
                },
                "triggered": bool(triggered[index]),
                "contacted": bool(contacted[index]),
                "succeeded": bool(succeeded[index]),
                "fallen": bool(fallen[index]),
                "first_trigger_step": int(
                    _first_event(trigger, args.episode_length)[index]
                ),
                "first_contact_step": int(
                    _first_event(contact, args.episode_length)[index]
                ),
                "episode_steps": int(episode_steps[index]),
                "maximum_directional_speed_mps": float(
                    maximum_directional_speed[index]
                ),
                "final_contact_distance_m": float(final_contact_distance[index]),
                "final_heading_error_rad": float(final_heading_error[index]),
                "final_activation": float(final_activation[index]),
                "final_goal_distance_m": float(final_goal_distance[index]),
            }
        )
    report = {
        "schema_version": 1,
        "purpose": "striker_closed_loop_accelerated_evaluation",
        "promotable": False,
        "promotion_blocker": "requires identical-control exact CPU replay",
        "implementation": args.impl,
        "jax_backend": jax.default_backend(),
        "stage": args.stage,
        "seed": args.seed,
        "episode_length": args.episode_length,
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "kick_prior": prior_metadata,
        "checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "checkpoint_sha256": (
            _checkpoint_fingerprint(args.checkpoint)
            if args.checkpoint is not None
            else None
        ),
        "environment_config": env._config.to_dict(),
        "summary": summary,
        "rollouts": rollout_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
