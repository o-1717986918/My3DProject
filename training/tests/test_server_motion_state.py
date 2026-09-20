from __future__ import annotations

from pathlib import Path

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_motion_state import (
    infer_previous_walk_action, project_server_motion_state,
)
from my3d_rl.t1_control import APOLLO_DEFAULT_POSE


CONTRACT = Path(__file__).parents[1] / "contracts" / "kick_policy_v3.yaml"


def test_server_torso_sensor_and_local_gyro_round_trip():
    scene = RcssKickScene(load_policy_contract(CONTRACT))
    yaw_quaternion = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    arrays = {
        "self_xyz": np.array([[1.0, -2.0, 0.72]]),
        "self_quat_wxyz": yaw_quaternion[None],
        "self_velocity_body": np.array([[0.5, 0.0, 0.0]]),
        "gyro_deg_s": np.array([[12.0, 0.0, 0.0]]),
        "joint_position_deg": np.zeros((1, 23)),
        "joint_velocity_deg_s": np.zeros((1, 23)),
        "ball_valid": np.array([1]),
        "ball_position_age_s": np.array([0.02]),
        "ball_velocity_valid": np.array([0]),
        "ball_xyz": np.array([[1.4, -2.0, 0.11]]),
    }

    project_server_motion_state(scene, arrays, 0)

    root = scene.model.joint(scene.prefix + "root")
    ball = scene.model.joint("ball-root")
    gyro = scene.data.sensor(scene.prefix + "torso_gyro").data
    np.testing.assert_allclose(
        scene.data.qpos[root.qposadr[0] : root.qposadr[0] + 3],
        [1.0, -2.0, 0.72], atol=1e-8,
    )
    np.testing.assert_allclose(
        scene.data.qvel[root.dofadr[0] : root.dofadr[0] + 3],
        [0.0, 0.5, 0.0], atol=1e-7,
    )
    np.testing.assert_allclose(gyro, np.deg2rad([12.0, 0.0, 0.0]), atol=1e-7)
    np.testing.assert_allclose(
        scene.data.qpos[ball.qposadr[0] : ball.qposadr[0] + 3],
        [1.4, -2.0, 0.11], atol=1e-8,
    )


def test_previous_stable_walk_action_recovers_motor_decoder_only():
    action = np.linspace(-0.5, 0.5, 23)
    targets = np.rad2deg(APOLLO_DEFAULT_POSE + 0.25 * action)
    arrays = {
        "match_id": np.array([1, 1]),
        "player_number": np.array([7, 7]),
        "time_s": np.array([1.0, 1.02]),
        "motion": np.array(["Walk", "Walk"]),
        "target_mask": np.ones((2, 23), dtype=np.uint8),
        "target_position_deg": np.stack([targets, targets]),
    }

    decoded, valid = infer_previous_walk_action(arrays, 1)

    assert valid
    np.testing.assert_allclose(decoded[:2], 0.0)
    np.testing.assert_allclose(decoded[2:], action[2:], atol=1e-12)
    arrays["motion"][0] = "RapidTurn"
    fallback, valid = infer_previous_walk_action(arrays, 1)
    assert not valid
    np.testing.assert_allclose(fallback, 0.0)
