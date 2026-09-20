"""Project server-observed T1 motion frames into the exact CPU soccer scene.

RCSS framepos/framequat observe the T1 torso site, which is coincident with
its free-joint root. Joint positions/velocities are direct MuJoCo sensors but
quantized by the server. Torso linear velocity is an agent finite difference,
not a direct server sensor; the gyro is a torso-local angular velocity.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .rcss_scene import RcssKickScene
from .t1_control import APOLLO_DEFAULT_POSE


def infer_previous_walk_action(
    arrays: dict[str, np.ndarray], row: int
) -> tuple[np.ndarray, bool]:
    """Recover the prior stable-Walk action from its sent position targets.

    The head targets are overwritten by the tracker in runtime and the Walk
    observation masks those two previous-action slots, so set them to zero.
    Other motion modes are not invertible with this decoder.
    """
    action = np.zeros(23, dtype=np.float64)
    if row < 1 or any(
        name not in arrays for name in (
            "match_id", "player_number", "time_s", "motion", "target_mask",
            "target_position_deg",
        )
    ):
        return action, False
    previous = row - 1
    if (
        arrays["match_id"][previous] != arrays["match_id"][row]
        or arrays["player_number"][previous] != arrays["player_number"][row]
        or abs(arrays["time_s"][row] - arrays["time_s"][previous] - 0.02) > 0.001
        or arrays["motion"][previous] != "Walk"
        or not np.all(arrays["target_mask"][previous] == 1)
    ):
        return action, False
    action = (
        np.deg2rad(np.asarray(arrays["target_position_deg"][previous], dtype=np.float64))
        - APOLLO_DEFAULT_POSE
    ) / 0.25
    action[:2] = 0.0
    if not np.isfinite(action).all() or np.max(np.abs(action[2:])) > 5.01:
        return np.zeros(23, dtype=np.float64), False
    return np.clip(action, -5.0, 5.0), True


def project_server_motion_state(
    scene: RcssKickScene,
    arrays: dict[str, np.ndarray],
    row: int,
    *,
    angular_frame: str = "body",
) -> None:
    """Set scene data from one observed frame; no contact correction is hidden."""
    if angular_frame not in {"body", "world"}:
        raise ValueError("angular_frame must be body or world")
    root = scene.model.joint(scene.prefix + "root")
    ball = scene.model.joint("ball-root")
    joint_qpos = [
        scene.model.joint(scene.prefix + name).qposadr[0]
        for name in scene.contract.joint_order
    ]
    joint_dof = [
        scene.model.joint(scene.prefix + name).dofadr[0]
        for name in scene.contract.joint_order
    ]
    quaternion = np.asarray(arrays["self_quat_wxyz"][row], dtype=np.float64)
    if not np.isfinite(quaternion).all() or not 0.8 <= np.linalg.norm(quaternion) <= 1.2:
        raise ValueError("invalid server torso quaternion")
    quaternion /= np.linalg.norm(quaternion)
    mujoco.mj_resetData(scene.model, scene.data)
    scene.data.qpos[root.qposadr[0] : root.qposadr[0] + 7] = [
        *np.asarray(arrays["self_xyz"][row], dtype=np.float64), *quaternion
    ]
    scene.data.qpos[joint_qpos] = np.deg2rad(arrays["joint_position_deg"][row])
    scene.data.qvel[joint_dof] = np.deg2rad(arrays["joint_velocity_deg_s"][row])
    world_velocity = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(
        world_velocity,
        np.asarray(arrays["self_velocity_body"][row], dtype=np.float64),
        quaternion,
    )
    scene.data.qvel[root.dofadr[0] : root.dofadr[0] + 3] = world_velocity
    gyro = np.deg2rad(np.asarray(arrays["gyro_deg_s"][row], dtype=np.float64))
    if angular_frame == "world":
        angular_velocity = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(angular_velocity, gyro, quaternion)
    else:
        angular_velocity = gyro
    scene.data.qvel[root.dofadr[0] + 3 : root.dofadr[0] + 6] = angular_velocity
    fresh_ball = (
        arrays["ball_valid"][row] == 1
        and np.isfinite(arrays["ball_position_age_s"][row])
        and arrays["ball_position_age_s"][row] <= 0.1
    )
    if fresh_ball:
        ball_xyz = np.asarray(arrays["ball_xyz"][row], dtype=np.float64)
    else:
        ball_xyz = np.asarray(arrays["self_xyz"][row], dtype=np.float64) + [10.0, 10.0, -0.5]
        ball_xyz[2] = 0.11
    scene.data.qpos[ball.qposadr[0] : ball.qposadr[0] + 7] = [
        *ball_xyz, 1.0, 0.0, 0.0, 0.0
    ]
    if fresh_ball and arrays["ball_velocity_valid"][row] == 1:
        scene.data.qvel[ball.dofadr[0] : ball.dofadr[0] + 3] = arrays["ball_velocity"][row]
    scene.data.time = 0.0
    mujoco.mj_forward(scene.model, scene.data)
