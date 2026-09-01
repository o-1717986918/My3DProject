from pathlib import Path

import jax
import numpy as np

from my3d_rl.kick_bc import (
    apply_behavior_clone_numpy,
    export_behavior_clone_onnx,
    train_behavior_clone,
)
from my3d_rl.kick_bc_jax import load_kick_behavior_clone_jax


def test_jax_loader_matches_exported_behavior_clone(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    observations = rng.normal(size=(12, 98)).astype(np.float32)
    dataset = {
        "observations": observations,
        "actions": np.tanh(rng.normal(size=(12, 23))).astype(np.float32),
        "episode_ids": np.repeat(np.arange(3), 4).astype(np.int32),
    }
    result = train_behavior_clone(
        dataset,
        seed=3,
        steps=2,
        batch_size=4,
        learning_rate=1.0e-3,
        validation_episode_ids=(2,),
    )
    path = tmp_path / "policy.onnx"
    export_behavior_clone_onnx(result, path)
    policy = load_kick_behavior_clone_jax(
        path, observation_size=98, action_size=23
    )

    expected = apply_behavior_clone_numpy(result, observations[:5])
    actual = np.asarray(jax.vmap(policy)(observations[:5]))

    np.testing.assert_allclose(actual, expected, atol=2.0e-6)
