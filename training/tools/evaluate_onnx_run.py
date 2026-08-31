#!/usr/bin/env python3
"""CPU MuJoCo acceptance for deployed 78/80 -> 23 locomotion policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import build_single_t1_soccer_model
from my3d_rl.run_env import NOMINAL_TRAINING_POSE, TRAIN_TO_SERVER_SIGN


REPOSITORY_ROOT = Path(__file__).parents[2]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v1.yaml"
PHASE_CONTRACT = REPOSITORY_ROOT / "training" / "contracts" / "run_policy_v2.yaml"
DEFAULT_MODEL = REPOSITORY_ROOT / "mujococodebase" / "skills" / "walk" / "walk.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _has_contact(data: mujoco.MjData, geom_a: int, geom_b: int) -> bool:
    pairs = data.contact.geom[: data.ncon]
    return bool(
        np.any(
            ((pairs[:, 0] == geom_a) & (pairs[:, 1] == geom_b))
            | ((pairs[:, 0] == geom_b) & (pairs[:, 1] == geom_a))
        )
    )


def _max_consecutive(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _box_geometric_contact(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    geom_id: int,
    *,
    pitch_height: float,
    tolerance: float = 0.01,
) -> bool:
    """Mirror the MJX-Warp contact proxy using an oriented box lower bound."""
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    lowest_z = data.geom_xpos[geom_id, 2] - np.sum(
        np.abs(rotation[2, :]) * model.geom_size[geom_id]
    )
    return bool(lowest_z <= pitch_height + tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4501)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.45)
    parser.add_argument("--gait-frequency", type=float, default=1.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 < args.action_scale <= 1.0:
        raise ValueError("action-scale must be in (0, 1]")

    model = build_single_t1_soccer_model(prefix="accept_", robot_x=-10.0)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    actor_size = input_meta.shape[-1]
    if actor_size not in (78, 80) or output_meta.shape[-1] != 23:
        raise ValueError(
            f"unexpected ONNX boundary {input_meta.shape} -> {output_meta.shape}"
        )
    contract_path = args.contract or (
        PHASE_CONTRACT if actor_size == 80 else DEFAULT_CONTRACT
    )
    contract = load_policy_contract(contract_path)
    if contract.observation_size != actor_size:
        raise ValueError(
            f"contract observation size {contract.observation_size} does not match "
            f"ONNX input {actor_size}"
        )

    prefix = "accept_"
    joint_qpos = np.array(
        [model.joint(prefix + name).qposadr[0] for name in contract.joint_order]
    )
    joint_dof = np.array(
        [model.joint(prefix + name).dofadr[0] for name in contract.joint_order]
    )
    pos_actuator = np.array(
        [model.actuator(prefix + name + "_pos").id for name in contract.effector_order]
    )
    vel_actuator = np.array(
        [model.actuator(prefix + name + "_vel").id for name in contract.effector_order]
    )
    root_joint = model.joint(prefix + "root")
    root_qpos = root_joint.qposadr[0]
    root_dof = root_joint.dofadr[0]
    torso_body = model.body(prefix + "torso").id
    torso_site = model.site(prefix + "torso").id
    gyro = model.sensor(prefix + "torso_gyro")
    gyro_slice = slice(gyro.adr[0], gyro.adr[0] + gyro.dim[0])
    pitch_geom = model.geom("pitch").id
    left_foot_geom = model.geom(prefix + "left_foot").id
    right_foot_geom = model.geom(prefix + "right_foot").id
    pitch_height = float(model.geom_pos[pitch_geom, 2])
    ball_joint = model.joint("ball-root")

    lowers = np.array(
        [model.joint(prefix + name).range[0] for name in contract.joint_order]
    )
    uppers = np.array(
        [model.joint(prefix + name).range[1] for name in contract.joint_order]
    )
    nominal = NOMINAL_TRAINING_POSE.astype(np.float64)
    sign = TRAIN_TO_SERVER_SIGN.astype(np.float64)
    nominal_physical = np.clip(nominal * sign, lowers, uppers)
    model.actuator_gainprm[pos_actuator, 0] = 25.0
    model.actuator_biasprm[pos_actuator, 1] = -25.0
    model.actuator_gainprm[vel_actuator, 0] = 0.6
    model.actuator_biasprm[vel_actuator, 2] = -0.6

    rng = np.random.default_rng(args.seed)
    command = np.array([args.vx, args.vy, args.yaw_rate], dtype=np.float32)
    completed: list[bool] = []
    mean_forward_speeds: list[float] = []
    forward_rmses: list[float] = []
    lateral_drifts: list[float] = []
    flight_phase: list[bool] = []
    max_flight_steps: list[int] = []
    invalid_episodes = 0
    proxy_true_positive = 0
    proxy_true_negative = 0
    proxy_false_positive = 0
    proxy_false_negative = 0
    proxy_flight_frames = 0
    actual_flight_frames = 0
    episode_steps = round(10.0 / 0.02)
    warmup_steps = round(2.0 / 0.02)

    for _ in range(args.episodes):
        mujoco.mj_resetData(model, data)
        joint_noise = rng.uniform(-0.02, 0.02, contract.action_size)
        data.qpos[joint_qpos] = np.clip(nominal_physical + joint_noise, lowers, uppers)
        yaw = rng.uniform(-0.05, 0.05)
        data.qpos[root_qpos + 3 : root_qpos + 7] = [
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        ]
        data.qvel[root_dof : root_dof + 6] = rng.uniform(-0.03, 0.03, 6)
        data.qpos[ball_joint.qposadr[0] : ball_joint.qposadr[0] + 3] = [
            10.0,
            8.0,
            0.11,
        ]
        data.ctrl[pos_actuator] = nominal_physical
        mujoco.mj_forward(model, data)

        initial_xy = data.qpos[root_qpos : root_qpos + 2].copy()
        previous_action = np.zeros(contract.action_size, dtype=np.float32)
        gait_phase = float(rng.uniform())
        forward_velocity: list[float] = []
        airborne_substeps: list[bool] = []
        fell = False
        invalid = False

        for step in range(episode_steps):
            joint_position_training = data.qpos[joint_qpos] * sign
            joint_velocity_training = data.qvel[joint_dof] * sign
            joint_triplets = np.stack(
                [
                    (joint_position_training - nominal) / 4.6,
                    joint_velocity_training / 110.0,
                    previous_action / 10.0,
                ],
                axis=1,
            ).reshape(-1)
            projected_gravity = data.site_xmat[torso_site].reshape(3, 3).T @ np.array(
                [0.0, 0.0, -1.0]
            )
            observation = np.concatenate(
                [
                    joint_triplets,
                    data.sensordata[gyro_slice] / 50.0,
                    command,
                    projected_gravity,
                ]
            ).astype(np.float32)
            if actor_size == 80:
                moving = float(np.linalg.norm(command[:2]) > 1.0e-4)
                angle = 2.0 * np.pi * gait_phase
                observation = np.concatenate(
                    [
                        observation,
                        moving
                        * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32),
                    ]
                )
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
            observation = np.clip(observation, -10.0, 10.0)
            action = session.run(None, {input_meta.name: observation[None]})[0][0]
            action = np.clip(
                np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0),
                -10.0,
                10.0,
            )
            targets = np.clip(
                (nominal + args.action_scale * action) * sign, lowers, uppers
            )
            data.ctrl[pos_actuator] = targets
            previous_action = action.astype(np.float32)
            gait_phase = (gait_phase + 0.02 * args.gait_frequency) % 1.0

            for _ in range(4):
                mujoco.mj_step(model, data)
                left_contact = _has_contact(data, pitch_geom, left_foot_geom)
                right_contact = _has_contact(data, pitch_geom, right_foot_geom)
                airborne_substeps.append(not left_contact and not right_contact)
                proxy_left = _box_geometric_contact(
                    data,
                    model,
                    left_foot_geom,
                    pitch_height=pitch_height,
                )
                proxy_right = _box_geometric_contact(
                    data,
                    model,
                    right_foot_geom,
                    pitch_height=pitch_height,
                )
                for actual, proxy in (
                    (left_contact, proxy_left),
                    (right_contact, proxy_right),
                ):
                    proxy_true_positive += int(actual and proxy)
                    proxy_true_negative += int(not actual and not proxy)
                    proxy_false_positive += int(not actual and proxy)
                    proxy_false_negative += int(actual and not proxy)
                proxy_flight_frames += int(not proxy_left and not proxy_right)
                actual_flight_frames += int(not left_contact and not right_contact)

            torso_xmat = data.site_xmat[torso_site].reshape(3, 3)
            body_yaw = np.arctan2(torso_xmat[1, 0], torso_xmat[0, 0])
            c, s = np.cos(body_yaw), np.sin(body_yaw)
            world_vx, world_vy = data.qvel[root_dof : root_dof + 2]
            forward_velocity.append(c * world_vx + s * world_vy)
            upright = torso_xmat[2, 2]
            height = data.xpos[torso_body, 2]
            invalid = not (
                np.isfinite(data.qpos).all()
                and np.isfinite(data.qvel).all()
                and np.isfinite(action).all()
            )
            fell = height < 0.35 or upright < 0.20
            if invalid or fell:
                break

        forward = np.asarray(forward_velocity, dtype=np.float64)
        after_warmup = forward[min(warmup_steps, forward.size) :]
        if after_warmup.size == 0:
            after_warmup = forward
        delta_world = data.qpos[root_qpos : root_qpos + 2] - initial_xy
        c0, s0 = np.cos(yaw), np.sin(yaw)
        lateral = -s0 * delta_world[0] + c0 * delta_world[1]
        airborne_array = np.asarray(airborne_substeps[warmup_steps * 4 :], dtype=bool)
        longest_flight = _max_consecutive(airborne_array)

        completed.append(
            not fell and not invalid and len(forward_velocity) == episode_steps
        )
        mean_forward_speeds.append(float(np.mean(after_warmup)))
        forward_rmses.append(float(np.sqrt(np.mean(np.square(after_warmup - args.vx)))))
        lateral_drifts.append(abs(float(lateral)))
        # Two consecutive 5 ms physics frames reject one-frame contact noise.
        flight_phase.append(longest_flight >= 2)
        max_flight_steps.append(longest_flight)
        invalid_episodes += int(invalid)

    speeds = np.asarray(mean_forward_speeds)
    rmses = np.asarray(forward_rmses)
    drifts = np.asarray(lateral_drifts)
    completion_rate = float(np.mean(completed))
    flight_rate = float(np.mean(flight_phase))
    payload = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "contract": str(contract_path.resolve()),
        "engine": "MuJoCo CPU",
        "episodes": args.episodes,
        "seed": args.seed,
        "command": command.tolist(),
        "action_scale": args.action_scale,
        "gait_frequency_hz": (args.gait_frequency if actor_size == 80 else None),
        "duration_seconds": 10.0,
        "warmup_seconds": 2.0,
        "upright_completion_rate": completion_rate,
        "invalid_episode_count": invalid_episodes,
        "forward_speed": {
            "median_m_s": _percentile(speeds, 50),
            "p10_m_s": _percentile(speeds, 10),
            "p90_m_s": _percentile(speeds, 90),
        },
        "forward_tracking_rmse": {
            "median_m_s": _percentile(rmses, 50),
            "p90_m_s": _percentile(rmses, 90),
        },
        "absolute_lateral_drift": {
            "median_m": _percentile(drifts, 50),
            "p90_m": _percentile(drifts, 90),
        },
        "flight_phase_episode_rate": flight_rate,
        "max_flight_duration": {
            "median_seconds": _percentile(np.asarray(max_flight_steps) * 0.005, 50),
            "p90_seconds": _percentile(np.asarray(max_flight_steps) * 0.005, 90),
        },
        "training_contact_proxy": {
            "method": "oriented_box_lowest_point_lte_pitch_plus_0.01m",
            "true_positive_frames": proxy_true_positive,
            "true_negative_frames": proxy_true_negative,
            "false_positive_frames": proxy_false_positive,
            "false_negative_frames": proxy_false_negative,
            "proxy_flight_frames": proxy_flight_frames,
            "actual_flight_frames": actual_flight_frames,
        },
    }
    gates = {
        "upright_completion_rate_gte_0_95": completion_rate >= 0.95,
        "median_forward_speed_gte_1_2": payload["forward_speed"]["median_m_s"] >= 1.2,
        "median_forward_rmse_lte_0_35": payload["forward_tracking_rmse"]["median_m_s"]
        <= 0.35,
        "median_lateral_drift_lte_0_25": payload["absolute_lateral_drift"]["median_m"]
        <= 0.25,
        "flight_phase_episode_rate_gte_0_80": flight_rate >= 0.80,
        "all_values_finite": invalid_episodes == 0,
    }
    payload["gates"] = gates
    payload["candidate_gate_passed"] = all(gates.values())

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
