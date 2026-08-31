#!/usr/bin/env python3
"""Import patched Holosoma T1 output into the local-only motion-prior schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.holosoma_motion import build_motion_reference
from my3d_rl.motion_reference import validate_motion_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Holosoma qpos NPZ")
    parser.add_argument("output", type=Path, help="local-only output NPZ")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("training/contracts/run_policy_v2.yaml"),
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end-inclusive", type=int)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--source-contact-height", type=float, default=0.02)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.output_fps != 50.0:
        raise SystemExit("motion_reference_v1 requires exactly 50 Hz")
    if len(args.source_sha256) != 64:
        raise SystemExit("--source-sha256 must be a 64-character SHA-256")
    command = " ".join(shlex.quote(value) for value in sys.argv)
    provenance = {
        "source_url": args.source_url,
        "source_version": args.source_version,
        "source_license": args.source_license,
        "source_sha256": args.source_sha256,
        "holosoma_input_sha256": _sha256(args.input),
        "conversion_command": command,
        "holosoma_required_patch": "training/patches/holosoma-t1-retargeting.patch",
    }
    contract = load_policy_contract(args.contract)
    arrays, _ = build_motion_reference(
        args.input,
        contract,
        output_fps=args.output_fps,
        frame_start=args.frame_start,
        frame_end_inclusive=args.frame_end_inclusive,
        source_height_threshold=args.source_contact_height,
        provenance=provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    result = validate_motion_reference(args.output)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
