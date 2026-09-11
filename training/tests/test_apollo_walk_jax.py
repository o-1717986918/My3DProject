from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

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


def test_apollo_walk_jax_loads_fused_training_export(tmp_path: Path):
    source = onnx.load(str(WALK_POLICY))
    arrays = {
        initializer.name: np.asarray(
            numpy_helper.to_array(initializer), dtype=np.float32
        ).copy()
        for initializer in source.graph.initializer
    }
    mean = arrays["normalizer._mean"].reshape(78)
    scale = arrays["add"].reshape(78)
    initializers = []
    nodes = []
    previous = "observations"
    for index, source_name in enumerate(
        ("actor.0", "actor.2", "actor.4", "actor.6"), start=1
    ):
        weight = arrays[f"{source_name}.weight"]
        bias = arrays[f"{source_name}.bias"]
        if index == 1:
            bias = bias - weight @ (mean / scale)
            weight = weight / scale[None, :]
        weight_name = f"fc{index}.weight"
        bias_name = f"fc{index}.bias"
        raw = "actions_raw" if index == 4 else f"fc{index}_raw"
        initializers.extend(
            [
                numpy_helper.from_array(weight, weight_name),
                numpy_helper.from_array(bias, bias_name),
            ]
        )
        nodes.append(
            helper.make_node(
                "Gemm",
                [previous, weight_name, bias_name],
                [raw],
                transB=1,
            )
        )
        if index < 4:
            previous = f"fc{index}_elu"
            nodes.append(helper.make_node("Elu", [raw], [previous]))
    initializers.extend(
        [
            numpy_helper.from_array(np.array(-5.0, dtype=np.float32), "minimum"),
            numpy_helper.from_array(np.array(5.0, dtype=np.float32), "maximum"),
        ]
    )
    nodes.append(
        helper.make_node(
            "Clip", ["actions_raw", "minimum", "maximum"], ["actions"]
        )
    )
    graph = helper.make_graph(
        nodes,
        "fused_apollo_actor",
        [helper.make_tensor_value_info("observations", TensorProto.FLOAT, [None, 78])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, 23])],
        initializer=initializers,
    )
    fused_path = tmp_path / "fused_apollo.onnx"
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, fused_path)

    observations = np.random.default_rng(5302).normal(
        0.0, 0.25, size=(16, 78)
    ).astype(np.float32)
    actual = np.asarray(load_apollo_walk_jax(fused_path)(observations))
    session = ort.InferenceSession(
        str(fused_path), providers=["CPUExecutionProvider"]
    )
    expected = session.run(None, {"observations": observations})[0]
    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=1.0e-5)
