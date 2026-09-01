#!/usr/bin/env python3
"""Evaluate the deterministic long-horizon striker in exact CPU MuJoCo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.striker_cpu import StrikerCpuEvaluator
from my3d_rl.striker_env import DEFAULT_CONTRACT, default_config
from tools.train_striker_teacher import STAGES, _load_kick_prior_bank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kick_prior_manifest", type=Path)
    parser.add_argument("--kick-prior-condition-index", type=int, default=1)
    parser.add_argument(
        "--kick-prior-bank-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--stage", choices=tuple(STAGES), default="closed_loop")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12_203)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--fixed-kick-prior-index", type=int, default=-1)
    parser.add_argument(
        "--trigger-history-output",
        type=Path,
        help="optional compressed 50-frame trigger histories outside the repository",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_rollouts < 1 or args.episode_length < 1:
        raise ValueError("rollout count and episode length must be positive")
    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be an absolute path outside the repository")
    if args.trigger_history_output is not None and (
        not args.trigger_history_output.is_absolute()
        or args.trigger_history_output.is_relative_to(Path.cwd())
        or args.trigger_history_output.exists()
    ):
        raise ValueError("trigger-history output must be a new absolute external path")

    contract = load_policy_contract(args.contract)
    prior, prior_distances, prior_metadata = _load_kick_prior_bank(
        args.kick_prior_manifest,
        args.kick_prior_bank_manifest,
        contract,
        primary_condition_index=args.kick_prior_condition_index,
    )
    config = default_config()
    config.update(STAGES[args.stage])
    config.episode_length = args.episode_length
    config.fixed_action_mode = 0
    config.fixed_desired_arrival_speed = 0.8
    config.fixed_kick_prior_index = args.fixed_kick_prior_index
    evaluator = StrikerCpuEvaluator(
        contract,
        prior,
        config.to_dict(),
        kick_prior_target_distances=prior_distances,
    )
    results = [
        evaluator.rollout(
            args.seed + index,
            capture_trigger_history=args.trigger_history_output is not None,
        )
        for index in range(args.num_rollouts)
    ]
    succeeded = np.asarray([result.succeeded for result in results])
    contacted = np.asarray([result.contacted for result in results])
    triggered = np.asarray([result.triggered for result in results])
    fallen = np.asarray([result.fallen for result in results])
    rollout_records = []
    trigger_histories = []
    trigger_history_ids = []
    for result in results:
        record = asdict(result)
        history = record.pop("trigger_actor_history")
        rollout_records.append(record)
        if history:
            trigger_histories.append(history)
            trigger_history_ids.append(result.seed)
    trigger_history_evidence = None
    if args.trigger_history_output is not None:
        histories = np.asarray(trigger_histories, dtype=np.float32)
        if histories.shape != (len(trigger_histories), 50, 102):
            raise RuntimeError("captured trigger histories have an invalid shape")
        args.trigger_history_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trigger_history_output,
            rollout_id=np.asarray(trigger_history_ids, dtype=np.int64),
            actor_history=histories,
        )
        trigger_history_evidence = {
            "path": str(args.trigger_history_output.resolve()),
            "sha256": _sha256(args.trigger_history_output),
            "rollouts": int(histories.shape[0]),
            "frames": int(histories.shape[1]),
            "observation_size": int(histories.shape[2]),
        }
    report = {
        "schema_version": 1,
        "purpose": "striker_closed_loop_exact_cpu_evaluation",
        "promotable": False,
        "promotion_blocker": (
            "requires source/ONNX parity, three seeds and RCSSServerMJ replay"
        ),
        "implementation": "exact_cpu_mujoco",
        "seed": args.seed,
        "episode_length": args.episode_length,
        "stage": args.stage,
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "kick_prior": prior_metadata,
        "environment_config": config.to_dict(),
        "success_definition": {
            "goal_radius_m": float(config.success_radius),
            "arrival_speed_tolerance_mps": float(
                config.arrival_speed_tolerance
            ),
            "requires_contact": True,
        },
        "trigger_history": trigger_history_evidence,
        "summary": {
            "rollouts": args.num_rollouts,
            "triggered": int(triggered.sum()),
            "contacted": int(contacted.sum()),
            "succeeded": int(succeeded.sum()),
            "fallen": int(fallen.sum()),
            "trigger_rate": float(triggered.mean()),
            "contact_rate": float(contacted.mean()),
            "success_rate": float(succeeded.mean()),
            "fall_rate": float(fallen.mean()),
            "mean_episode_steps": float(
                np.mean([result.episode_steps for result in results])
            ),
            "maximum_directional_speed_mean_mps": float(
                np.mean(
                    [result.maximum_directional_speed_mps for result in results]
                )
            ),
            "maximum_directional_speed_max_mps": float(
                np.max(
                    [result.maximum_directional_speed_mps for result in results]
                )
            ),
            "goal_distance_mean_m": float(
                np.mean([result.final_goal_distance_m for result in results])
            ),
            "gate_passed": bool(succeeded.mean() >= 0.90 and not fallen.any()),
        },
        "rollouts": rollout_records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
