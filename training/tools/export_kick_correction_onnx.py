#!/usr/bin/env python3
"""Export a formal kick-correction PPO checkpoint to deterministic ONNX."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from my3d_rl.contract import load_policy_contract
from my3d_rl.kick_env import TRANSITION_CONTRACT


def _tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _network_factory():
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="normal",
        noise_std_type="log",
        init_noise_std=0.05,
        mean_clip_scale=1.0,
        mean_kernel_init_fn=jax.nn.initializers.normal,
        mean_kernel_init_kwargs={"stddev": 0.0},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--contract", type=Path, default=TRANSITION_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-output", type=Path)
    parser.add_argument("--seed", type=int, default=6201)
    args = parser.parse_args()

    contract = load_policy_contract(args.contract)

    params = ppo_checkpoint.load(args.checkpoint)
    normalizer_params, policy_params = params[0], params[1]
    learned = policy_params["params"]
    if set(learned) != {"Dense_0", "MLP_0", "std_logparam"}:
        raise ValueError("checkpoint is not the formal kick-correction network")
    actor_size = int(learned["MLP_0"]["hidden_0"]["kernel"].shape[0])
    if (
        actor_size != contract.observation_size
        or learned["Dense_0"]["bias"].shape != (contract.action_size,)
    ):
        raise ValueError("checkpoint differs from the selected kick contract")
    observation_mean = np.asarray(normalizer_params.mean["state"])
    observation_std = np.asarray(normalizer_params.std["state"])
    if (
        observation_mean.shape != (actor_size,)
        or observation_std.shape != (actor_size,)
    ):
        raise ValueError("checkpoint observation normalizer is incompatible")

    initializers = [
        _tensor("observation_mean", observation_mean),
        _tensor("observation_std", observation_std),
    ]
    nodes = [
        helper.make_node(
            "Sub", ["observations", "observation_mean"], ["observations_centered"]
        ),
        helper.make_node(
            "Div",
            ["observations_centered", "observation_std"],
            ["observations_normalized"],
        ),
    ]
    previous = "observations_normalized"
    for index, expected_size in enumerate((256, 128, 128)):
        values = learned["MLP_0"][f"hidden_{index}"]
        if int(values["kernel"].shape[1]) != expected_size:
            raise ValueError("checkpoint hidden-layer shape is incompatible")
        weight = f"hidden_{index}.weight"
        bias = f"hidden_{index}.bias"
        raw = f"hidden_{index}.raw"
        sigmoid = f"hidden_{index}.sigmoid"
        activated = f"hidden_{index}.swish"
        initializers.extend(
            [
                _tensor(weight, np.asarray(values["kernel"]).T),
                _tensor(bias, np.asarray(values["bias"])),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Gemm", [previous, weight, bias], [raw], transB=1),
                helper.make_node("Sigmoid", [raw], [sigmoid]),
                helper.make_node("Mul", [raw, sigmoid], [activated]),
            ]
        )
        previous = activated

    output_values = learned["Dense_0"]
    initializers.extend(
        [
            _tensor("output.weight", np.asarray(output_values["kernel"]).T),
            _tensor("output.bias", np.asarray(output_values["bias"])),
            _tensor("one", np.array(1.0, dtype=np.float32)),
        ]
    )
    nodes.extend(
        [
            helper.make_node(
                "Gemm",
                [previous, "output.weight", "output.bias"],
                ["actions_raw"],
                transB=1,
            ),
            helper.make_node("Abs", ["actions_raw"], ["actions_abs"]),
            helper.make_node("Add", ["actions_abs", "one"], ["actions_denominator"]),
            helper.make_node(
                "Div", ["actions_raw", "actions_denominator"], ["actions"]
            ),
        ]
    )
    graph = helper.make_graph(
        nodes,
        contract.policy_name + "_correction",
        [
            helper.make_tensor_value_info(
                "observations", TensorProto.FLOAT, list(contract.input_shape)
            )
        ],
        [
            helper.make_tensor_value_info(
                "actions", TensorProto.FLOAT, list(contract.output_shape)
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="My3DProject training/tools/export_kick_correction_onnx.py",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    for key, value in {
        "checkpoint": str(args.checkpoint.resolve()),
        "policy": contract.policy_name,
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
        "composition": "apollo_walk_plus_teacher_table_plus_bounded_correction",
    }.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    networks = _network_factory()(
        {"state": (actor_size,), "privileged_state": (actor_size + 10,)},
        contract.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    rng = np.random.default_rng(args.seed)
    observations = (
        observation_mean[None]
        + observation_std[None] * rng.normal(0.0, 1.0, (256, actor_size))
    ).astype(np.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = networks.policy_network.apply(
            normalizer_params,
            policy_params,
            {
                "state": jp.asarray(observations),
                "privileged_state": jp.zeros(
                    (256, actor_size + 10), dtype=jp.float32
                ),
            },
        )[0]
    session = ort.InferenceSession(
        str(args.output.resolve()), providers=["CPUExecutionProvider"]
    )
    actual = np.concatenate(
        [
            session.run(None, {"observations": row[None]})[0]
            for row in observations
        ],
        axis=0,
    )
    difference = np.abs(np.asarray(expected) - actual)
    parity = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx": str(args.output.resolve()),
        "contract": str(args.contract.resolve()),
        "contract_sha256": _sha256(args.contract),
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
    if args.parity_output is not None:
        args.parity_output.parent.mkdir(parents=True, exist_ok=True)
        args.parity_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
