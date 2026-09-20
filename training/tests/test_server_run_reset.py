from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from my3d_rl.contract import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene
from my3d_rl.server_run_env import ServerFastWalkRun
from tools.build_server_run_reset_corpus import select_reset_rows
from tools.train_server_fastwalk import fastwalk_overrides


CONTRACT = Path(__file__).parents[1] / "contracts" / "run_policy_v2.yaml"


def _telemetry() -> dict[str, np.ndarray]:
    count = 100
    xyz = np.zeros((count, 3), dtype=np.float32)
    xyz[:, 2] = 0.65
    speed = np.zeros((count, 3), dtype=np.float32)
    speed[:, 0] = 0.4
    return {
        "motion": np.full(count, "Walk", dtype="U64"),
        "self_xyz": xyz,
        "self_velocity_body": speed,
        "target_mask": np.ones((count, 23), dtype=np.uint8),
        "match_id": np.zeros(count, dtype=np.int32),
        "player_number": np.full(count, 3, dtype=np.int32),
        "time_s": np.arange(count, dtype=np.float64) * 0.02,
    }


def test_reset_row_selection_thins_dynamic_walk_per_player_time_bucket():
    arrays = _telemetry()
    rows = select_reset_rows(arrays, stride_s=0.2, min_speed_m_s=0.2)
    assert len(rows) == 10
    assert len(set((arrays["time_s"][rows] / 0.2).astype(int))) == 10
    assert np.all(rows > 0)
    arrays["motion"][:] = "GetUp"
    with pytest.raises(ValueError, match="no dynamic"):
        select_reset_rows(arrays, stride_s=0.2, min_speed_m_s=0.2)


def test_server_fastwalk_reset_preserves_projected_pose_and_runtime_handoff(tmp_path):
    contract = load_policy_contract(CONTRACT)
    scene = RcssKickScene(contract, prefix="train_")
    qpos = np.tile(scene.data.qpos, (2, 1)).astype(np.float32)
    qvel = np.zeros((2, scene.model.nv), dtype=np.float32)
    root = scene.model.joint("train_root")
    qvel[:, root.dofadr[0]] = 0.4
    corpus = tmp_path / "server-run-resets.npz"
    np.savez_compressed(
        corpus, qpos=qpos, qvel=qvel,
        split=np.array([0, 1], dtype=np.uint8),
        near_ball=np.zeros(2, dtype=np.uint8),
        source_row=np.array([17, 29], dtype=np.int32),
    )
    env = ServerFastWalkRun(
        corpus, split=0, contract=contract,
        config_overrides=fastwalk_overrides(
            impl="jax", graph_mode="auto", num_envs=1
        ),
    )
    state = env.reset(jax.random.PRNGKey(2))
    assert int(state.info["entry_source_row"]) == 17
    np.testing.assert_allclose(np.asarray(state.data.qpos), qpos[0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(state.data.qvel), qvel[0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(state.info["command"]), [1.5, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(state.info["last_action"]), 0.0)
    assert float(state.info["gait_phase"]) == 0.0
