#!/usr/bin/env python3
"""Export accepted teacher parameters as a provenance-carrying runtime table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [record for record in source["records"] if record["accepted"]]
    if not records or len(records) != source["condition_count"]:
        raise ValueError("runtime export requires a complete accepted condition grid")
    nodes = []
    for record in records:
        parameters = [float(value) for value in record["parameters"]]
        if len(parameters) != 14:
            raise ValueError("teacher parameter vector must contain 14 values")
        nodes.append(
            {
                "condition_index": int(record["condition_index"]),
                "distance_m": float(record["distance_m"]),
                "angle_deg": float(record["angle_deg"]),
                "requested_speed_mps": float(record["requested_speed_mps"]),
                "ball_x_offset_m": float(record["ball_x_offset_m"]),
                "ball_y_offset_m": float(record["ball_y_offset_m"]),
                "mode": str(record["mode"]),
                "parameters": parameters,
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "experimental_kick_residual_parameter_table",
        "promotable": False,
        "promotion_blocker": "held-out exact-physics success remains below 90 percent",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": _sha256(args.manifest),
        "evaluations": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.evaluation
        ],
        "nodes": sorted(nodes, key=lambda node: node["condition_index"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "node_count": len(nodes)}))


if __name__ == "__main__":
    main()
