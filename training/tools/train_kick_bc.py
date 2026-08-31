#!/usr/bin/env python3
"""Fit and export the kick_policy_v2 supervised initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from my3d_rl.kick_bc import (
    export_behavior_clone_onnx,
    load_teacher_dataset,
    train_behavior_clone,
    verify_onnx_parity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--validation-episodes", type=int, nargs="+")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/win98/rl_runs/kick-bc/kick-policy-v2"),
    )
    args = parser.parse_args()

    dataset = load_teacher_dataset(args.dataset)
    result = train_behavior_clone(
        dataset,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_episode_ids=(
            tuple(args.validation_episodes)
            if args.validation_episodes is not None
            else None
        ),
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_prefix.with_suffix(".npz")
    onnx_path = args.output_prefix.with_suffix(".onnx")
    manifest_path = args.output_prefix.with_suffix(".json")
    dense = [
        (
            np.asarray(result.params[f"Dense_{index}"]["kernel"]),
            np.asarray(result.params[f"Dense_{index}"]["bias"]),
        )
        for index in range(3)
    ]
    np.savez_compressed(
        weights_path,
        observation_mean=result.observation_mean,
        observation_std=result.observation_std,
        dense_0_kernel=dense[0][0],
        dense_0_bias=dense[0][1],
        dense_1_kernel=dense[1][0],
        dense_1_bias=dense[1][1],
        dense_2_kernel=dense[2][0],
        dense_2_bias=dense[2][1],
    )
    export_behavior_clone_onnx(result, onnx_path)
    parity = verify_onnx_parity(result, onnx_path, dataset["observations"])
    manifest = {
        "purpose": "kick_policy_v2_supervised_initializer",
        "promotable": False,
        "promotion_blocker": "requires closed-loop physics, multi-seed and server gates",
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "weights": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_episode_ids": list(result.train_episode_ids),
        "validation_episode_ids": list(result.validation_episode_ids),
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "onnx_parity": parity,
        "history": list(result.history),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
