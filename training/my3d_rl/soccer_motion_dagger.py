"""Provenance-checked phase teachers for soccer-motion DAgger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import PolicyContract
from .soccer_motion_corpus import SoccerMotionCorpus
from .soccer_motion_teacher import decode_phase_correction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected_teacher_corrections(
    selection_manifest: Path,
    *,
    corpus: SoccerMotionCorpus,
    contract: PolicyContract,
    contract_sha256: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Load one hash-bound correction teacher for every corpus motion."""
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "complete_teacher_corpus_selected"
        or selection.get("teacher_gates_passed") != corpus.motion_count
        or selection.get("motion_count") != corpus.motion_count
    ):
        raise ValueError("teacher selection manifest is incomplete")
    nodes = selection.get("selection")
    if not isinstance(nodes, list) or len(nodes) != corpus.motion_count:
        raise ValueError("teacher selection does not cover the corpus")

    corrections: list[np.ndarray | None] = [None] * corpus.motion_count
    provenance: list[dict[str, Any] | None] = [None] * corpus.motion_count
    for node in nodes:
        motion = int(node["motion"])
        if not 0 <= motion < corpus.motion_count or corrections[motion] is not None:
            raise ValueError("teacher selection has duplicate or invalid motion IDs")
        if node["relative_path"] != corpus.relative_paths[motion]:
            raise ValueError("teacher selection motion path differs from the corpus")
        report_path = Path(node["report"])
        if not report_path.is_file() or sha256(report_path) != node["report_sha256"]:
            raise ValueError("selected teacher report hash mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            not report.get("k1_a_teacher_gate_passed")
            or report.get("relative_path") != corpus.relative_paths[motion]
            or report.get("contract_sha256") != contract_sha256
        ):
            raise ValueError("selected teacher report is incompatible")
        length = int(corpus.lengths[motion])
        corrections[motion] = decode_phase_correction(
            np.asarray(report["parameters"], dtype=np.float64),
            phases=np.linspace(0.0, 1.0, length),
            action_size=contract.action_size,
            joint_indices=report["active_joint_indices"],
            knot_count=int(report["knot_count"]),
            maximum_abs_correction=float(report["maximum_abs_correction"]),
        )
        provenance[motion] = {
            "motion": motion,
            "relative_path": corpus.relative_paths[motion],
            "report": str(report_path.resolve()),
            "report_sha256": node["report_sha256"],
            "teacher_base_checkpoint": report.get("checkpoint"),
        }

    if any(value is None for value in corrections + provenance):
        raise ValueError("teacher selection leaves a motion uncovered")
    return (
        [value for value in corrections if value is not None],
        [value for value in provenance if value is not None],
    )
