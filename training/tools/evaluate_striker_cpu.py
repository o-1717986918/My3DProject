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
from tools.train_striker_teacher import STAGES, _load_kick_prior


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
    parser.add_argument("--stage", choices=tuple(STAGES), default="closed_loop")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12_203)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_rollouts < 1 or args.episode_length < 1:
        raise ValueError("rollout count and episode length must be positive")
    if not args.output.is_absolute() or args.output.is_relative_to(Path.cwd()):
        raise ValueError("output must be an absolute path outside the repository")

    contract = load_policy_contract(args.contract)
    prior, prior_metadata = _load_kick_prior(
        args.kick_prior_manifest,
        contract,
        condition_index=args.kick_prior_condition_index,
    )
    config = default_config()
    config.update(STAGES[args.stage])
    config.episode_length = args.episode_length
    config.fixed_action_mode = 0
    config.fixed_desired_arrival_speed = 0.8
    evaluator = StrikerCpuEvaluator(contract, prior, config.to_dict())
    results = [
        evaluator.rollout(args.seed + index) for index in range(args.num_rollouts)
    ]
    succeeded = np.asarray([result.succeeded for result in results])
    contacted = np.asarray([result.contacted for result in results])
    triggered = np.asarray([result.triggered for result in results])
    fallen = np.asarray([result.fallen for result in results])
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
        "rollouts": [asdict(result) for result in results],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
