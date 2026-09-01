import json
from pathlib import Path

import pytest

from training.tools.create_kick_server_calibration_assets import (
    create_asset_tree,
    create_calibration_table,
    parse_parameter_delta,
)


def _source_table() -> dict[str, object]:
    return {
        "schema_version": 1,
        "cpu_gate_passed": True,
        "source_manifest": "teacher.json",
        "nodes": [
            {
                "condition_index": 60,
                "distance_m": 2.0,
                "angle_deg": 0.0,
                "requested_speed_mps": 1.43,
                "ball_x_offset_m": 0.02375,
                "ball_y_offset_m": 0.01,
                "mode": "pass",
                "parameters": [0.0] * 14,
            }
        ],
    }


def test_create_calibration_table_applies_accumulated_deltas() -> None:
    result = create_calibration_table(
        _source_table(),
        condition_index=60,
        parameter_deltas=[(8, -0.1), (8, -0.05), (2, 0.2)],
        source_sha256="abc",
    )
    node = result["nodes"][0]
    assert node["parameters"][8] == pytest.approx(-0.15)
    assert node["parameters"][2] == pytest.approx(0.2)
    assert result["calibration"]["parameter_deltas"] == {
        "2": 0.2,
        "8": pytest.approx(-0.15),
    }
    assert result["source_table_sha256"] == "abc"


def test_create_asset_tree_copies_runtime_assets(tmp_path: Path) -> None:
    base = tmp_path / "base"
    (base / "keyframes").mkdir(parents=True)
    (base / "networks" / "walk").mkdir(parents=True)
    (base / "keyframes" / "kick_residual_table.yaml").write_text(
        json.dumps(_source_table()), encoding="utf-8"
    )
    (base / "networks" / "walk" / "policy.onnx").write_bytes(b"model")
    output = tmp_path / "candidate"

    table_path = create_asset_tree(
        base,
        output,
        condition_index=60,
        parameter_deltas=[(8, -0.2)],
    )

    table = json.loads(table_path.read_text(encoding="utf-8"))
    assert table["nodes"][0]["parameters"][8] == pytest.approx(-0.2)
    assert (output / "networks" / "walk" / "policy.onnx").read_bytes() == b"model"


def test_parse_parameter_delta_rejects_out_of_range_index() -> None:
    assert parse_parameter_delta("8=-0.2") == (8, -0.2)
    with pytest.raises(Exception):
        parse_parameter_delta("14=0.1")
