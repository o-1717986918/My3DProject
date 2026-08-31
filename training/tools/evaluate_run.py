#!/usr/bin/env python3
"""Deterministically evaluate a Brax PPO checkpoint on the RCSS run task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from brax.training import types as brax_types
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

from my3d_rl.contract import load_policy_contract
from my3d_rl.ppo_profile import PROFILES, get_ppo_profile
from my3d_rl.run_env import DirectionalRun


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--motion-reference", type=Path)
    parser.add_argument(
        "--network-profile",
        choices=tuple(PROFILES),
        default="t1_tanh_v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profile = get_ppo_profile(args.network_profile)
    contract_path = (
        Path(__file__).parents[1] / "contracts" / f"{profile.policy_contract}.yaml"
    )
    contract = load_policy_contract(contract_path)
    reference_centered = (
        contract.control_mode == "motion_reference_residual_joint_position"
    )
    if reference_centered and args.motion_reference is None:
        raise ValueError("reference-centred policy requires --motion-reference")
    if not reference_centered and args.motion_reference is not None:
        raise ValueError("--motion-reference requires a reference-centred policy")
    env = DirectionalRun(
        config_overrides={
            "impl": args.impl,
            "naconmax": max(2048, 16 * args.episodes),
            "action_clip": max(abs(value) for value in contract.action_clip),
            "use_fixed_command": True,
            "fixed_command": [args.vx, args.vy, args.yaw_rate],
            "reset_joint_noise": 0.005 if reference_centered else 0.03,
            "reset_root_velocity_noise": 0.01 if reference_centered else 0.05,
            "reset_yaw_range": 0.02 if reference_centered else 0.10,
            "reference_init_probability": 1.0 if reference_centered else 0.0,
            "push_enable": False,
            "action_delay_max_steps": 0,
        },
        contract=contract,
        motion_reference=args.motion_reference,
    )
    network_factory = profile.network_factory()
    # Brax 0.14.2's generic load_policy path cannot reconstruct a checkpoint
    # whose serialized default kernel initializer is null. Recreate the exact
    # training network explicitly and then load the numeric parameter tree.
    preprocess_observations_fn = (
        running_statistics.normalize
        if profile.normalize_observations
        else brax_types.identity_observation_preprocessor
    )
    networks = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    params = ppo_checkpoint.load(args.checkpoint)
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    reset = jax.jit(jax.vmap(env.reset))
    env_step = jax.vmap(env.step)
    policy_batch = jax.vmap(policy)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    initial_state = reset(keys)
    initial_xy = initial_state.data.qpos[:, env._root_qpos : env._root_qpos + 2]

    def scan_step(carry: tuple[Any, jax.Array], unused: None):
        del unused
        state, rng = carry
        rng, action_rng = jax.random.split(rng)
        action_keys = jax.random.split(action_rng, args.episodes)
        actions = policy_batch(state.obs, action_keys)[0]
        next_state = env_step(state, actions)
        diagnostics = jp.stack(
            [
                next_state.metrics["diagnostic/local_velocity_x"],
                next_state.metrics["diagnostic/local_velocity_y"],
                next_state.metrics["diagnostic/yaw_rate"],
                next_state.metrics["diagnostic/torso_height"],
                next_state.metrics["cost/fall"],
                next_state.done,
                next_state.data.qpos[:, env._root_qpos],
                next_state.data.qpos[:, env._root_qpos + 1],
            ],
            axis=-1,
        )
        return (next_state, rng), diagnostics

    @jax.jit
    def rollout(state, rng):
        return jax.lax.scan(
            scan_step,
            (state, rng),
            None,
            length=env._config.episode_length,
        )

    (_, _), diagnostics = rollout(
        initial_state, jax.random.PRNGKey(args.seed + 1)
    )
    diagnostics = np.asarray(diagnostics)
    initial_xy_np = np.asarray(initial_xy)
    warmup = round(2.0 / env.dt)
    done = diagnostics[:, :, 5] > 0.0
    has_done = done.any(axis=0)
    first_done = np.argmax(done, axis=0) + 1
    episode_length = np.where(has_done, first_done, env._config.episode_length)
    fall = diagnostics[:, :, 4].max(axis=0) > 0.0
    mean_vx = np.empty(args.episodes)
    rmse_vx = np.empty(args.episodes)
    lateral_speed = np.empty(args.episodes)
    yaw_rate_error = np.empty(args.episodes)
    terminal_xy = np.empty((args.episodes, 2))
    for episode, length_value in enumerate(episode_length):
        length = int(length_value)
        begin = min(warmup, length)
        segment = diagnostics[begin:length, episode]
        if segment.shape[0] == 0:
            segment = diagnostics[:length, episode]
        mean_vx[episode] = np.mean(segment[:, 0])
        rmse_vx[episode] = np.sqrt(np.mean(np.square(segment[:, 0] - args.vx)))
        lateral_speed[episode] = np.mean(np.abs(segment[:, 1]))
        yaw_rate_error[episode] = np.mean(np.abs(segment[:, 2] - args.yaw_rate))
        terminal_xy[episode] = diagnostics[length - 1, episode, 6:8]
    lateral_drift = np.abs(terminal_xy[:, 1] - initial_xy_np[:, 1])
    invalid = ~np.isfinite(diagnostics).all(axis=(0, 2))

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "implementation": args.impl,
        "network_profile": profile.name,
        "episodes": args.episodes,
        "seed": args.seed,
        "command": [args.vx, args.vy, args.yaw_rate],
        "motion_reference": (
            str(args.motion_reference.resolve()) if args.motion_reference else None
        ),
        "duration_seconds": env._config.episode_length * env.dt,
        "warmup_seconds": warmup * env.dt,
        "upright_completion_rate": float(np.mean(~fall)),
        "survival": {
            "median_control_steps": _percentile(episode_length, 50),
            "p10_control_steps": _percentile(episode_length, 10),
            "p90_control_steps": _percentile(episode_length, 90),
            "maximum_control_steps": int(np.max(episode_length)),
        },
        "invalid_episode_count": int(invalid.sum()),
        "forward_speed": {
            "median_m_s": _percentile(mean_vx, 50),
            "p10_m_s": _percentile(mean_vx, 10),
            "p90_m_s": _percentile(mean_vx, 90),
        },
        "forward_tracking_rmse": {
            "median_m_s": _percentile(rmse_vx, 50),
            "p90_m_s": _percentile(rmse_vx, 90),
        },
        "absolute_lateral_drift": {
            "median_m": _percentile(lateral_drift, 50),
            "p90_m": _percentile(lateral_drift, 90),
        },
        "lateral_speed_abs_median_m_s": _percentile(lateral_speed, 50),
        "yaw_rate_abs_error_median_rad_s": _percentile(yaw_rate_error, 50),
        "flight_phase_evaluated": False,
        "flight_phase_note": (
            "MJX training evaluator does not certify contact morphology; "
            "CPU MuJoCo/ONNX acceptance performs the aerial-phase check."
        ),
    }
    gates = {
        "upright_completion_rate_gte_0_95": (
            payload["upright_completion_rate"] >= 0.95
        ),
        "median_forward_speed_gte_1_2": (payload["forward_speed"]["median_m_s"] >= 1.2),
        "median_forward_rmse_lte_0_35": (
            payload["forward_tracking_rmse"]["median_m_s"] <= 0.35
        ),
        "median_lateral_drift_lte_0_25": (
            payload["absolute_lateral_drift"]["median_m"] <= 0.25
        ),
        "all_values_finite": payload["invalid_episode_count"] == 0,
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
