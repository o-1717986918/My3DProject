"""Create a single-node Apollo asset tree for RCSSServerMJ kick calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


PARAMETER_COUNT = 14


def parse_parameter_delta(value: str) -> tuple[int, float]:
    try:
        raw_index, raw_delta = value.split("=", 1)
        index = int(raw_index)
        delta = float(raw_delta)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "parameter deltas must use INDEX=DELTA"
        ) from exc
    if not 0 <= index < PARAMETER_COUNT:
        raise argparse.ArgumentTypeError(
            f"parameter index must be in [0, {PARAMETER_COUNT - 1}]"
        )
    return index, delta


def create_calibration_table(
    source: dict[str, object],
    *,
    condition_index: int,
    parameter_deltas: Iterable[tuple[int, float]] = (),
    source_sha256: str = "",
) -> dict[str, object]:
    nodes = source.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("source table has no node list")
    matches = [
        node
        for node in nodes
        if isinstance(node, dict)
        and node.get("condition_index") == condition_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one node for condition {condition_index}"
        )
    node = copy.deepcopy(matches[0])
    parameters = node.get("parameters")
    if not isinstance(parameters, list) or len(parameters) != PARAMETER_COUNT:
        raise ValueError("source node does not contain 14 parameters")
    applied: dict[str, float] = {}
    for index, delta in parameter_deltas:
        parameters[index] = float(parameters[index]) + delta
        applied[str(index)] = applied.get(str(index), 0.0) + delta

    return {
        "schema_version": 1,
        "purpose": "rcssservermj_kick_calibration_candidate",
        "promotable": False,
        "promotion_blocker": "single-node server calibration asset",
        "declared_condition_count": 1,
        "accepted_condition_count": 1,
        "complete_grid": False,
        "cpu_gate_passed": bool(source.get("cpu_gate_passed", False)),
        "source_manifest": source.get("source_manifest", ""),
        "source_table_sha256": source_sha256,
        "calibration": {
            "condition_index": condition_index,
            "parameter_deltas": applied,
        },
        "nodes": [node],
    }


def create_asset_tree(
    base_asset_root: Path,
    output_root: Path,
    *,
    condition_index: int,
    parameter_deltas: Iterable[tuple[int, float]] = (),
) -> Path:
    source_table_path = base_asset_root / "keyframes" / "kick_residual_table.yaml"
    source_bytes = source_table_path.read_bytes()
    source = json.loads(source_bytes)
    table = create_calibration_table(
        source,
        condition_index=condition_index,
        parameter_deltas=parameter_deltas,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )

    if output_root.exists():
        raise FileExistsError(f"output asset root already exists: {output_root}")
    shutil.copytree(base_asset_root, output_root)
    output_table_path = output_root / "keyframes" / "kick_residual_table.yaml"
    output_table_path.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_table_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-asset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--condition-index", required=True, type=int)
    parser.add_argument(
        "--parameter-delta",
        action="append",
        default=[],
        type=parse_parameter_delta,
        metavar="INDEX=DELTA",
    )
    args = parser.parse_args()
    output = create_asset_tree(
        args.base_asset_root,
        args.output_root,
        condition_index=args.condition_index,
        parameter_deltas=args.parameter_delta,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
