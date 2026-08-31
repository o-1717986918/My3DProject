#!/usr/bin/env python3
"""Export legacy and standard Brax PPO checkpoints to runtime ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from brax.training.acme import running_statistics
from brax.training import types as brax_types
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from my3d_rl.contract import load_policy_contract
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
    legacy_layout = "fc1" in learned
    standard_normal_layout = "MLP_0" in learned and "Dense_0" in learned
    if legacy_layout:
        actor_size = int(learned["fc1"]["kernel"].shape[0])
    elif standard_normal_layout:
        actor_size = int(learned["MLP_0"]["hidden_0"]["kernel"].shape[0])
    else:
        raise ValueError(
            "unsupported actor parameter layout; expected legacy fc layers or "
            "the standard Brax normal-policy MLP"
        )
    if actor_size not in (78, 80):
        raise ValueError(f"unsupported actor observation size {actor_size}")
    contract_name = profile.policy_contract
    contract = load_policy_contract(
        Path(__file__).parents[1] / "contracts" / f"{contract_name}.yaml"
    )
    if contract.observation_size != actor_size or contract.action_size != 23:
        raise ValueError("checkpoint shape differs from the profile policy contract")

    initializers = []
    nodes = []
    previous = "observations"
    if profile.normalize_observations:
        state_mean = np.asarray(normalizer_params.mean["state"])
        state_std = np.asarray(normalizer_params.std["state"])
        if state_mean.shape != (actor_size,) or state_std.shape != (actor_size,):
            raise ValueError("checkpoint observation normalizer has incompatible shape")
        initializers.extend(
            [
                _tensor("observation_mean", state_mean),
                _tensor("observation_std", state_std),
            ]
        )
        nodes.append(
            helper.make_node(
                "Sub",
                [previous, "observation_mean"],
                ["observations_centered"],
                name="observation_centering",
            )
        )
        nodes.append(
            helper.make_node(
                "Div",
                ["observations_centered", "observation_std"],
                ["observations_normalized"],
                name="observation_normalization",
            )
        )
        previous = "observations_normalized"

    if legacy_layout:
        for layer in ("fc1", "fc2", "fc3", "fc4"):
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
                    helper.make_node(
                        "Elu", ["fc1_norm"], ["fc1_elu"], name="fc1_elu"
                    )
                )
                previous = "fc1_elu"
            elif layer != "fc4":
                activated = f"{layer}_elu"
                nodes.append(
                    helper.make_node(
                        "Elu", [output_name], [activated], name=activated
                    )
                )
                previous = activated
    else:
        if profile.distribution_type != "normal":
            raise ValueError("standard ONNX export currently requires a normal policy")
        hidden_params = learned["MLP_0"]
        for index, expected_size in enumerate(profile.policy_hidden_layer_sizes):
            layer = f"hidden_{index}"
            if layer not in hidden_params:
                raise ValueError(f"checkpoint is missing policy layer {layer}")
            weight_name = f"policy.{layer}.weight"
            bias_name = f"policy.{layer}.bias"
            raw_name = f"policy.{layer}.raw"
            sigmoid_name = f"policy.{layer}.sigmoid"
            activated_name = f"policy.{layer}.swish"
            values = hidden_params[layer]
            if int(values["kernel"].shape[1]) != expected_size:
                raise ValueError(f"checkpoint layer {layer} differs from PPO profile")
            initializers.extend(
                [
                    _tensor(weight_name, np.asarray(values["kernel"]).T),
                    _tensor(bias_name, np.asarray(values["bias"])),
                ]
            )
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [previous, weight_name, bias_name],
                    [raw_name],
                    name=f"policy_{layer}_gemm",
                    transB=1,
                )
            )
            nodes.append(
                helper.make_node(
                    "Sigmoid",
                    [raw_name],
                    [sigmoid_name],
                    name=f"policy_{layer}_sigmoid",
                )
            )
            nodes.append(
                helper.make_node(
                    "Mul",
                    [raw_name, sigmoid_name],
                    [activated_name],
                    name=f"policy_{layer}_swish",
                )
            )
            previous = activated_name
        output_values = learned["Dense_0"]
        initializers.extend(
            [
                _tensor("policy.output.weight", np.asarray(output_values["kernel"]).T),
                _tensor("policy.output.bias", np.asarray(output_values["bias"])),
            ]
        )
        nodes.append(
            helper.make_node(
                "Gemm",
                [previous, "policy.output.weight", "policy.output.bias"],
                ["actions_raw"],
                name="policy_output_gemm",
                transB=1,
            )
        )

    initializers.extend(
        [
            _tensor("action_min", np.array(contract.action_clip[0], dtype=np.float32)),
            _tensor("action_max", np.array(contract.action_clip[1], dtype=np.float32)),
        ]
    )
    nodes.append(
        helper.make_node(
            "Clip",
            ["actions_raw", "action_min", "action_max"],
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

    preprocess_observations_fn = (
        running_statistics.normalize
        if profile.normalize_observations
        else brax_types.identity_observation_preprocessor
    )
    networks = profile.network_factory()(
        {"state": (actor_size,), "privileged_state": (actor_size + 6,)},
        23,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    rng = np.random.default_rng(args.seed)
    if profile.normalize_observations:
        # Exercise the graph over the checkpoint's actual training domain.
        # Fixed command channels can have a 1e-6 standard deviation, so a raw
        # N(0, 0.25) sample would create artificial 250,000-sigma inputs.
        observations = (
            np.asarray(normalizer_params.mean["state"])[None]
            + np.asarray(normalizer_params.std["state"])[None]
            * rng.normal(0.0, 1.0, (256, actor_size))
        ).astype(np.float32)
    else:
        observations = rng.normal(0.0, 0.25, (256, actor_size)).astype(np.float32)
    # Compare CPU with CPU: GPU matmul/autotuning can differ by a few 1e-4
    # for this network even when the graph is mathematically exact.
    with jax.default_device(jax.devices("cpu")[0]):
        expected = networks.policy_network.apply(
            normalizer_params,
            policy_params,
            {
                "state": jp.asarray(observations),
                "privileged_state": jp.zeros(
                    (256, actor_size + 6), dtype=jp.float32
                ),
            },
        )[0]
    expected = np.clip(
        np.asarray(expected), contract.action_clip[0], contract.action_clip[1]
    )
    session = ort.InferenceSession(
        str(args.output.resolve()), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"observations": observations})[0]
    difference = np.abs(expected - actual)
    parity = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx": str(args.output.resolve()),
        "samples": observations.shape[0],
        "seed": args.seed,
        "comparison_device": "JAX CPU vs ONNX Runtime CPU",
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
