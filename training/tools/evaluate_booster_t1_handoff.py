#!/usr/bin/env python3
"""Evaluate Booster's official T1 locomotion actor on RCSS handoff states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import yaml

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT / "training" / "contracts" / "apollo_walk_policy_v1.yaml"
)
DEFAULT_REFERENCE_ROOT = Path("/home/win98/reference_sources/booster_gym")


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


def _yaw(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _waypoint_command(
    current_xy: np.ndarray,
    current_yaw: float,
    target_xy: np.ndarray,
) -> np.ndarray:
    delta = target_xy - current_xy
    c, s = np.cos(current_yaw), np.sin(current_yaw)
    local = np.array(
        [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]]
    )
    heading = np.arctan2(local[1], local[0]) if np.linalg.norm(local) > 0.1 else 0.0
    return np.array(
        [
            np.clip(local[0], -0.5, 1.0),
            np.clip(local[1], -0.5, 0.5),
            np.clip(0.2 * heading, -0.5, 0.5),
        ],
        dtype=np.float32,
    )


def _initial_phase(source_step: int, mode: str, policy_interval: float) -> float:
    if mode == "zero":
        return 0.0
    if mode == "source-step":
        return float(source_step * policy_interval % 1.0)
    raise ValueError(f"unsupported phase mode: {mode}")


def _initial_previous_action(
    joint_position: np.ndarray,
    default_pose: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros(12, dtype=np.float32)
    if mode == "pose-offset":
        return np.clip(joint_position[11:] - default_pose[11:], -1.0, 1.0).astype(
            np.float32
        )
    raise ValueError(f"unsupported previous-action mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--model",
        type=Path,
        help="TorchScript .pt or converted ONNX actor (default: official T1.pt)",
    )
    parser.add_argument("--entry-corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_261_419)
    parser.add_argument("--distance-min", type=float, default=2.0)
    parser.add_argument("--distance-max", type=float, default=4.0)
    parser.add_argument("--bearing-min-deg", type=float, default=-180.0)
    parser.add_argument("--bearing-max-deg", type=float, default=180.0)
    parser.add_argument(
        "--phase-mode",
        choices=("zero", "source-step"),
        default="zero",
        help="zero matches a fresh official policy session; source-step tests continuity",
    )
    parser.add_argument(
        "--previous-action-mode",
        choices=("zero", "pose-offset"),
        default="zero",
        help="zero matches the official deployment initialization",
    )
    parser.add_argument(
        "--command-smoothing-step",
        type=float,
        default=None,
        help="maximum command change per policy step (official default: policy interval)",
    )
    args = parser.parse_args()
    model_path = (
        args.model
        if args.model is not None
        else args.reference_root / "deploy" / "models" / "T1.pt"
    )
    config_path = args.reference_root / "deploy" / "configs" / "T1.yaml"
    if (
        not model_path.is_file()
        or not config_path.is_file()
        or not args.entry_corpus.is_file()
        or args.episodes < 1
        or args.steps < 1
        or not 0.25 < args.distance_min <= args.distance_max
        or not -180.0 <= args.bearing_min_deg < args.bearing_max_deg <= 180.0
    ):
        raise ValueError("Booster T1 handoff evaluation arguments are invalid")
    run_dir = _external_new_directory(args.run_dir)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if model_path.suffix == ".pt":
        import torch

        policy = torch.jit.load(str(model_path), map_location="cpu")
        policy.eval()

        def infer(observation: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return policy(torch.from_numpy(observation[None, :]))[0].numpy()

        actor_format = "torchscript"
    elif model_path.suffix == ".onnx":
        import onnxruntime as ort

        policy = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        if policy.get_inputs()[0].shape != [1, 47] or policy.get_outputs()[0].shape != [1, 12]:
            raise ValueError("converted Booster T1 ONNX shape is invalid")

        def infer(observation: np.ndarray) -> np.ndarray:
            return policy.run(["actions"], {"obs": observation[None, :]})[0][0]

        actor_format = "onnx"
    else:
        raise ValueError("model must be a TorchScript .pt or ONNX .onnx actor")
    contract = load_policy_contract(DEFAULT_CONTRACT)
    scene = RcssKickScene(contract)
    joint_qpos = np.asarray(
        [scene.model.joint(scene.prefix + name).qposadr[0] for name in contract.joint_order]
    )
    joint_dof = np.asarray(
        [scene.model.joint(scene.prefix + name).dofadr[0] for name in contract.joint_order]
    )
    torso_site = scene.model.site(scene.prefix + "torso").id
    gyro = scene.model.sensor(scene.prefix + "torso_gyro")
    gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
    default_pose = np.asarray(config["common"]["default_qpos"], dtype=np.float64)
    kp = np.asarray(config["common"]["stiffness"], dtype=np.float64)
    kd = np.asarray(config["common"]["damping"], dtype=np.float64)
    policy_interval = float(config["common"]["dt"]) * int(
        config["policy"]["control"]["decimation"]
    )
    command_smoothing_step = (
        policy_interval
        if args.command_smoothing_step is None
        else args.command_smoothing_step
    )
    if default_pose.shape != (23,) or kp.shape != (23,) or kd.shape != (23,):
        raise ValueError("official Booster T1 deployment config has changed shape")
    if not np.isclose(policy_interval, 1.0 / contract.frequency_hz):
        raise ValueError("official Booster policy period does not match RCSS control")
    if not 0.0 < command_smoothing_step <= 2.0:
        raise ValueError("command-smoothing-step must be in (0, 2]")
    with np.load(args.entry_corpus, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        qvel = np.asarray(archive["qvel"], dtype=np.float64)
        source_command = np.asarray(archive["source_command"], dtype=np.float32)
        source_step = np.asarray(archive["source_step"], dtype=np.int32)
    sample_count = qpos.shape[0]
    if (
        qpos.shape != (sample_count, scene.model.nq)
        or qvel.shape != (sample_count, scene.model.nv)
        or source_command.shape != (sample_count, 3)
        or source_step.shape != (sample_count,)
    ):
        raise ValueError("entry corpus is incompatible with the RCSS T1 model")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, sample_count, size=args.episodes)
    distances = rng.uniform(args.distance_min, args.distance_max, size=args.episodes)
    bearings = np.deg2rad(
        rng.uniform(args.bearing_min_deg, args.bearing_max_deg, size=args.episodes)
    )
    successes = np.zeros(args.episodes, dtype=bool)
    falls = np.zeros(args.episodes, dtype=bool)
    completion_steps = np.full(args.episodes, args.steps, dtype=np.int32)
    final_distances = np.zeros(args.episodes, dtype=np.float64)
    minimum_heights = np.full(args.episodes, np.inf, dtype=np.float64)

    for episode, (sample_index, distance, bearing) in enumerate(
        zip(sample_indices, distances, bearings, strict=True)
    ):
        scene.data.qpos[:] = qpos[sample_index]
        scene.data.qvel[:] = qvel[sample_index]
        scene.data.time = 0.0
        scene.data.ctrl[:] = 0.0
        mujoco.mj_forward(scene.model, scene.data)
        rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
        initial_yaw = _yaw(rotation)
        current_xy = scene.data.site_xpos[torso_site, :2].copy()
        target_xy = current_xy + distance * np.array(
            [np.cos(initial_yaw + bearing), np.sin(initial_yaw + bearing)]
        )
        smoothed_command = source_command[sample_index].copy()
        phase = _initial_phase(
            int(source_step[sample_index]), args.phase_mode, policy_interval
        )
        previous_action = _initial_previous_action(
            scene.data.qpos[joint_qpos], default_pose, args.previous_action_mode
        )

        for step in range(args.steps):
            rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
            raw_command = _waypoint_command(
                scene.data.site_xpos[torso_site, :2], _yaw(rotation), target_xy
            )
            smoothed_command += np.clip(
                raw_command - smoothed_command,
                -command_smoothing_step,
                command_smoothing_step,
            )
            moving = float(np.linalg.norm(smoothed_command) >= 1.0e-5)
            observation = np.zeros(47, dtype=np.float32)
            observation[0:3] = rotation.T @ np.array([0.0, 0.0, -1.0])
            observation[3:6] = scene.data.sensordata[gyro_slice]
            observation[6:9] = smoothed_command * moving
            observation[9:11] = moving * np.array(
                [np.cos(2.0 * np.pi * phase), np.sin(2.0 * np.pi * phase)]
            )
            observation[11:23] = (
                scene.data.qpos[joint_qpos[11:]] - default_pose[11:]
            )
            observation[23:35] = 0.1 * scene.data.qvel[joint_dof[11:]]
            observation[35:47] = previous_action
            action = infer(observation)
            action = np.clip(action, -1.0, 1.0)
            targets = default_pose.copy()
            targets[11:] += action
            scene.step_joint_targets(targets, kp=kp, kd=kd)
            previous_action = action.astype(np.float32)
            phase = (phase + policy_interval * moving) % 1.0

            rotation = scene.data.site_xmat[torso_site].reshape(3, 3)
            height = float(scene.data.xpos[scene.model.body(scene.prefix + "torso").id, 2])
            upright = float(rotation[2, 2])
            remaining = float(
                np.linalg.norm(target_xy - scene.data.site_xpos[torso_site, :2])
            )
            minimum_heights[episode] = min(minimum_heights[episode], height)
            if height < 0.35 or upright < 0.20:
                falls[episode] = True
                completion_steps[episode] = step + 1
                break
            if remaining <= 0.25:
                successes[episode] = True
                completion_steps[episode] = step + 1
                break
        final_distances[episode] = float(
            np.linalg.norm(target_xy - scene.data.site_xpos[torso_site, :2])
        )

    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "official_booster_t1_actor_on_rcss_handoff_states",
        "promotable": False,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip(),
        "reference_revision": subprocess.check_output(
            ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip(),
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "actor_format": actor_format,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "entry_corpus": str(args.entry_corpus.resolve()),
        "entry_corpus_sha256": _sha256(args.entry_corpus),
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "duration_s": 0.02 * args.steps,
        "distance_range_m": [args.distance_min, args.distance_max],
        "bearing_range_deg": [args.bearing_min_deg, args.bearing_max_deg],
        "phase_mode": args.phase_mode,
        "previous_action_mode": args.previous_action_mode,
        "command_smoothing_step": command_smoothing_step,
        "policy_interval_s": policy_interval,
        "successes": int(np.sum(successes)),
        "falls": int(np.sum(falls)),
        "timeouts": int(np.sum(~successes & ~falls)),
        "median_final_distance_m": float(np.median(final_distances)),
        "median_completion_time_s": float(np.median(completion_steps) * 0.02),
        "minimum_torso_height_m": float(np.min(minimum_heights)),
    }
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
