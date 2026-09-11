#!/usr/bin/env python3
"""Convert the official Booster T1 TorchScript actor to validated ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch


DEFAULT_REFERENCE_ROOT = Path("/home/win98/reference_sources/booster_gym")
SOURCE_REPOSITORY = "https://github.com/BoosterRobotics/booster_gym"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_output_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20_261_423)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_model = args.reference_root / "deploy" / "models" / "T1.pt"
    source_config = args.reference_root / "deploy" / "configs" / "T1.yaml"
    source_license = args.reference_root / "LICENSE"
    output_model = args.output_model.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_model.with_suffix(".manifest.json")
    )
    if not all(path.is_file() for path in (source_model, source_config, source_license)):
        raise FileNotFoundError("official Booster T1 source assets are incomplete")
    if args.samples < 1 or args.seed < 0:
        raise ValueError("samples and seed must be positive")
    if not args.overwrite and (output_model.exists() or manifest_path.exists()):
        raise FileExistsError("output model or manifest already exists")

    policy = torch.jit.load(str(source_model), map_location="cpu")
    policy.eval()
    probe = torch.zeros((1, 47), dtype=torch.float32)
    with torch.no_grad():
        output = policy(probe)
    if tuple(output.shape) != (1, 12):
        raise ValueError("official Booster T1 actor shape has changed")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=output_model.name + ".", suffix=".tmp", dir=output_model.parent
    )
    os.close(temporary_fd)
    temporary_model = Path(temporary_name)
    try:
        torch.onnx.export(
            policy,
            probe,
            str(temporary_model),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamo=False,
        )
        model = onnx.load(str(temporary_model))
        onnx.checker.check_model(model)
        input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
        if input_shape != [1, 47] or output_shape != [1, 12]:
            raise ValueError("exported Booster T1 ONNX shape is invalid")

        rng = np.random.default_rng(args.seed)
        observations = np.vstack(
            [
                np.zeros((1, 47), dtype=np.float32),
                np.clip(
                    rng.normal(0.0, 0.75, size=(args.samples, 47)), -10.0, 10.0
                ).astype(np.float32),
            ]
        )
        session = ort.InferenceSession(
            str(temporary_model), providers=["CPUExecutionProvider"]
        )
        maximum_error = 0.0
        for observation in observations:
            with torch.no_grad():
                expected = policy(torch.from_numpy(observation[None, :])).numpy()
            actual = session.run(["actions"], {"obs": observation[None, :]})[0]
            maximum_error = max(
                maximum_error, float(np.max(np.abs(expected - actual)))
            )
        parity_threshold = 5.0e-6
        if maximum_error > parity_threshold:
            raise RuntimeError(
                f"TorchScript/ONNX parity {maximum_error:.3e} exceeds "
                f"{parity_threshold:.3e}"
            )

        os.replace(temporary_model, output_model)
    finally:
        temporary_model.unlink(missing_ok=True)

    reference_revision = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "official_booster_t1_torchscript_to_onnx",
        "promotable": False,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": reference_revision,
        "source_model_relative_path": "deploy/models/T1.pt",
        "source_model_sha256": _sha256(source_model),
        "source_config_relative_path": "deploy/configs/T1.yaml",
        "source_config_sha256": _sha256(source_config),
        "source_license_sha256": _sha256(source_license),
        "output_model": _portable_output_path(output_model),
        "output_model_sha256": _sha256(output_model),
        "input_shape": [1, 47],
        "output_shape": [1, 12],
        "opset": 17,
        "parity": {
            "seed": args.seed,
            "samples": int(observations.shape[0]),
            "maximum_absolute_error": maximum_error,
            "required_maximum_absolute_error": parity_threshold,
            "passed": True,
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
