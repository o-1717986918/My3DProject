from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from my3d_rl.soccer_motion_dagger import (
    load_selected_teacher_corrections,
    sha256,
)


def _selection(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "k1_a_teacher_gate_passed": True,
                "relative_path": "motion.t1.npz",
                "contract_sha256": "c" * 64,
                "parameters": [0.0, 0.2],
                "active_joint_indices": [1],
                "knot_count": 2,
                "maximum_abs_correction": 0.5,
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "complete_teacher_corpus_selected",
                "teacher_gates_passed": 1,
                "motion_count": 1,
                "selection": [
                    {
                        "motion": 0,
                        "relative_path": "motion.t1.npz",
                        "report": str(report_path),
                        "report_sha256": sha256(report_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return selection_path, report_path


def test_selected_teacher_loader_decodes_hash_bound_report(tmp_path):
    selection_path, _ = _selection(tmp_path)
    corpus = SimpleNamespace(
        motion_count=1,
        relative_paths=("motion.t1.npz",),
        lengths=np.array([5]),
    )
    contract = SimpleNamespace(action_size=3)

    corrections, provenance = load_selected_teacher_corrections(
        selection_path,
        corpus=corpus,
        contract=contract,
        contract_sha256="c" * 64,
    )

    assert corrections[0].shape == (5, 3)
    np.testing.assert_array_equal(corrections[0][:, [0, 2]], 0.0)
    np.testing.assert_allclose(corrections[0][[0, -1], 1], [0.0, 0.2])
    assert provenance[0]["relative_path"] == "motion.t1.npz"


def test_selected_teacher_loader_rejects_tampered_report(tmp_path):
    selection_path, report_path = _selection(tmp_path)
    report_path.write_text("{}", encoding="utf-8")
    corpus = SimpleNamespace(
        motion_count=1,
        relative_paths=("motion.t1.npz",),
        lengths=np.array([5]),
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        load_selected_teacher_corrections(
            selection_path,
            corpus=corpus,
            contract=SimpleNamespace(action_size=3),
            contract_sha256="c" * 64,
        )
