#!/usr/bin/env python3
"""Export a legacy-compatible Brax PPO checkpoint to runtime ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from brax.training import types as brax_types
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from my3d_rl.ppo_profile import PROFILES, get_ppo_profile


def _tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-output", type=Path)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument(
        "--network-profile",
        choices=tuple(PROFILES),
        default="legacy_warmstart_v1",
    )
    args = parser.parse_args()

    profile = get_ppo_profile(args.network_profile)
    params = ppo_checkpoint.load(args.checkpoint)
    normalizer_params, policy_params = params[0], params[1]
    learned = policy_params["params"]
    actor_size = int(learned["fc1"]["kernel"].shape[0])
    if actor_size not in (78, 80):
        raise ValueError(f"unsupported actor observation size {actor_size}")
    contract_name = "run_policy_v2" if actor_size == 80 else "run_policy_v1"

    initializers = []
    nodes = []
    previous = "observations"
    for index, layer in enumerate(("fc1", "fc2", "fc3", "fc4"), start=1):
        weight_name = f"{layer}.weight"
        bias_name = f"{layer}.bias"
        output_name = "actions_raw" if layer == "fc4" else f"{layer}_raw"
        initializers.extend(
            [
                _tensor(weight_name, np.asarray(learned[layer]["kernel"]).T),
                _tensor(bias_name, np.asarray(learned[layer]["bias"])),
            ]
        )
        nodes.append(
            helper.make_node(
                "Gemm",
                [previous, weight_name, bias_name],
                [output_name],
                name=f"{layer}_gemm",
                transB=1,
            )
        )
        if layer == "fc1":
            initializers.extend(
                [
                    _tensor(
                        "layer_norm.weight",
                        np.asarray(learned["layer_norm"]["scale"]),
                    ),
                    _tensor(
                        "layer_norm.bias",
                        np.asarray(learned["layer_norm"]["bias"]),
                    ),
                ]
            )
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    [output_name, "layer_norm.weight", "layer_norm.bias"],
                    ["fc1_norm"],
                    name="layer_norm",
                    axis=-1,
                    epsilon=1.0e-6,
                )
            )
            nodes.append(
                helper.make_node("Elu", ["fc1_norm"], ["fc1_elu"], name="fc1_elu")
            )
            previous = "fc1_elu"
        elif layer != "fc4":
            activated = f"{layer}_elu"
            nodes.append(
                helper.make_node("Elu", [output_name], [activated], name=activated)
            )
            previous = activated
        else:
            initializers.extend(
                [
                    _tensor("action_min", np.array(-10.0, dtype=np.float32)),
                    _tensor("action_max", np.array(10.0, dtype=np.float32)),
                ]
            )
            nodes.append(
                helper.make_node(
                    "Clip",
                    [output_name, "action_min", "action_max"],
                    ["actions"],
                    name="action_clip",
                )
            )

    graph = helper.make_graph(
        nodes,
        f"my3d_{contract_name}",
        [
            helper.make_tensor_value_info(
                "observations", TensorProto.FLOAT, [None, actor_size]
            )
        ],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [None, 23])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="My3DProject training/tools/export_run_onnx.py",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "network_profile": profile.name,
        "observation_contract": f"{contract_name}:{actor_size}",
        "action_contract": f"{contract_name}:23",
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    networks = profile.network_factory()(
        {"state": (actor_size,), "privileged_state": (actor_size + 6,)},
        23,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )
    rng = np.random.default_rng(args.seed)
    observations = rng.normal(0.0, 0.25, (256, actor_size)).astype(np.float32)
    expected = networks.policy_network.apply(
        normalizer_params,
        policy_params,
        {
            "state": jp.asarray(observations),
            "privileged_state": jp.zeros((256, actor_size + 6), dtype=jp.float32),
        },
    )[0]
    session = ort.InferenceSession(
        str(args.output.resolve()), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"observations": observations})[0]
    difference = np.abs(np.asarray(expected) - actual)
    parity = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx": str(args.output.resolve()),
        "samples": observations.shape[0],
        "seed": args.seed,
        "max_abs_error": float(np.max(difference)),
        "mean_abs_error": float(np.mean(difference)),
        "passed": bool(np.max(difference) <= 2.0e-5),
    }
    if not parity["passed"]:
        raise ValueError(f"ONNX parity failed: {parity}")
    rendered = json.dumps(parity, indent=2, sort_keys=True) + "\n"
    if args.parity_output:
        args.parity_output.parent.mkdir(parents=True, exist_ok=True)
        args.parity_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
