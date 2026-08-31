from pathlib import Path

import numpy as np
import onnxruntime as ort

from my3d_rl.apollo_walk_jax import load_apollo_walk_jax


REPOSITORY_ROOT = Path(__file__).parents[2]
WALK_POLICY = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)


def test_apollo_walk_jax_matches_cpu_onnx_runtime():
    rng = np.random.default_rng(5301)
    observations = rng.normal(0.0, 0.25, size=(16, 78)).astype(np.float32)
    policy = load_apollo_walk_jax(WALK_POLICY)
    expected = np.asarray(policy(observations))
    session = ort.InferenceSession(str(WALK_POLICY), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    actual = np.concatenate(
        [session.run(None, {input_name: row[None, :]})[0] for row in observations],
        axis=0,
    )
    np.testing.assert_allclose(expected, actual, atol=2.0e-5, rtol=1.0e-5)
