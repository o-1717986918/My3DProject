#!/usr/bin/env python3
"""Replay one captured exact-CPU teacher trajectory in the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from my3d_rl.rcss_scene import build_single_t1_soccer_model


def select_episode(
    dataset: dict[str, np.ndarray],
    *,
    split: int,
    start_frame: int | None,
) -> tuple[np.ndarray, int]:
    required = {"qpos", "split", "start_frame"}
    if not required.issubset(dataset):
        raise ValueError(f"teacher dataset is missing {sorted(required - set(dataset))}")
    split_values = np.asarray(dataset["split"])
    starts = np.asarray(dataset["start_frame"])
    qpos = np.asarray(dataset["qpos"], dtype=np.float64)
    if split_values.shape != starts.shape or split_values.shape != qpos.shape[:1]:
        raise ValueError("teacher replay arrays have inconsistent lengths")
    available = sorted(set(starts[split_values == split].tolist()))
    if not available:
        raise ValueError(f"teacher dataset has no split={split} trajectory")
    selected = available[0] if start_frame is None else start_frame
    if selected not in available:
        raise ValueError(
            f"start frame {selected} is unavailable; choose one of {available}"
        )
    mask = (split_values == split) & (starts == selected)
    trajectory = qpos[mask]
    if trajectory.ndim != 2 or trajectory.shape[0] < 1:
        raise ValueError("selected teacher trajectory is empty")
    if not np.isfinite(trajectory).all():
        raise ValueError("selected teacher trajectory contains non-finite state")
    return trajectory, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0.0 or args.loops < 1:
        raise ValueError("speed must be positive and loops must be at least one")
    with np.load(args.dataset, allow_pickle=False) as archive:
        dataset = {name: archive[name] for name in archive.files}
    split = 0 if args.split == "train" else 1
    trajectory, selected = select_episode(
        dataset, split=split, start_frame=args.start_frame
    )
    model = build_single_t1_soccer_model(
        prefix="soccer_teacher_", robot_x=-10.0, robot_y=0.0
    )
    if trajectory.shape[1] != model.nq:
        raise ValueError(
            f"captured qpos width {trajectory.shape[1]} != model nq {model.nq}"
        )
    data = mujoco.MjData(model)
    frame_period = 0.02 / args.speed
    print(
        f"replaying split={args.split} start_frame={selected} "
        f"frames={trajectory.shape[0]} speed={args.speed}x",
        flush=True,
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(args.loops):
            for qpos in trajectory:
                if not viewer.is_running():
                    return
                started = time.monotonic()
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                viewer.sync()
                remaining = frame_period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        while args.hold and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
