#!/usr/bin/env python3
"""Render a compact, headless MuJoCo contact sheet for a saved 50 Hz replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from my3d_rl.rcss_scene import build_single_t1_soccer_model
from tools.view_soccer_motion_teacher import select_episode


REPOSITORY_ROOT = Path(__file__).parents[2]


def frame_indices(length: int, count: int) -> np.ndarray:
    if length < 1 or count < 1:
        raise ValueError("replay length and frame count must be positive")
    return np.unique(np.rint(np.linspace(0, length - 1, min(length, count))).astype(int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        not args.replay.is_file() or not args.output.is_absolute()
        or output.is_relative_to(REPOSITORY_ROOT.resolve()) or output.exists()
        or args.frames < 1 or args.width < 160 or args.height < 120
    ):
        raise ValueError("replay must exist; output must be new, absolute, and outside the repo")
    with np.load(args.replay, allow_pickle=False) as archive:
        dataset = {name: np.asarray(archive[name]) for name in archive.files}
    trajectory, source_row = select_episode(
        dataset, split=0 if args.split == "train" else 1,
        start_frame=args.start_frame,
    )
    model = build_single_t1_soccer_model(prefix="soccer_teacher_")
    if trajectory.shape[1] != model.nq:
        raise ValueError("replay qpos does not match exact RCSS T1 model")
    data = mujoco.MjData(model)
    root = model.joint("soccer_teacher_root")
    start = trajectory[0, root.qposadr[0] : root.qposadr[0] + 3]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [start[0] + 0.4, start[1], 0.65]
    camera.distance = 2.8
    camera.azimuth = 105.0
    camera.elevation = -20.0
    indices = frame_indices(len(trajectory), args.frames)
    sheet = Image.new("RGB", (args.width * len(indices), args.height + 32), "#101819")
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        for column, index in enumerate(indices):
            data.qpos[:] = trajectory[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            sheet.paste(Image.fromarray(renderer.render()), (column * args.width, 0))
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        draw.text(
            (column * args.width + 8, args.height + 8),
            f"t={index * 0.02:.2f}s  source={source_row}",
            fill="white",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"{output} frames={len(indices)} replay_steps={len(trajectory)}")


if __name__ == "__main__":
    main()
