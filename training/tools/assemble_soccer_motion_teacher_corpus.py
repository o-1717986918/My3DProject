#!/usr/bin/env python3
"""Select passing per-motion teachers and assemble the K1-A BC corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from tools.optimize_soccer_motion_teacher_corpus import (
    aggregate_teacher_datasets,
    require_external_path,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_override(value: str) -> tuple[int, Path]:
    motion_text, separator, path_text = value.partition("=")
    if not separator or not motion_text.isdigit() or not path_text:
        raise ValueError("override must have the form MOTION=/absolute/report.json")
    return int(motion_text), Path(path_text)


def _clean_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    if status:
        raise RuntimeError("formal teacher-corpus assembly requires a clean Git tree")
    return revision


def aggregate_split(reports: list[dict[str, Any]], split: str) -> dict[str, Any]:
    summaries = [report[split]["summary"] for report in reports]
    trials = np.asarray([summary["trials"] for summary in summaries], dtype=np.int64)
    if np.any(trials < 1):
        raise ValueError("teacher report contains an empty split")

    def weighted(name: str) -> float:
        values = np.asarray([summary[name] for summary in summaries])
        return float(np.average(values, weights=trials))

    return {
        "trials": int(np.sum(trials)),
        "baseline_completions": int(
            sum(summary["baseline_completions"] for summary in summaries)
        ),
        "candidate_completions": int(
            sum(summary["candidate_completions"] for summary in summaries)
        ),
        "baseline_only_completions": int(
            sum(summary["baseline_only_completions"] for summary in summaries)
        ),
        "candidate_only_completions": int(
            sum(summary["candidate_only_completions"] for summary in summaries)
        ),
        "baseline_mean_survival_fraction": weighted(
            "baseline_mean_survival_fraction"
        ),
        "candidate_mean_survival_fraction": weighted(
            "candidate_mean_survival_fraction"
        ),
        "mean_survival_fraction_delta": weighted(
            "mean_survival_fraction_delta"
        ),
        "baseline_mean_teacher_score": weighted("baseline_mean_teacher_score"),
        "candidate_mean_teacher_score": weighted("candidate_mean_teacher_score"),
        "mean_teacher_score_delta": weighted("mean_teacher_score_delta"),
        "survival_improvements": int(
            sum(summary["survival_improvements"] for summary in summaries)
        ),
        "survival_regressions": int(
            sum(summary["survival_regressions"] for summary in summaries)
        ),
    }


def select_reports(
    batch: dict[str, Any], overrides: dict[int, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    motion_count = int(batch["motion_count"])
    if any(motion < 0 or motion >= motion_count for motion in overrides):
        raise ValueError("override motion lies outside the batch corpus")
    reports: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for motion in range(motion_count):
        batch_node = batch["motions"][str(motion)]
        path = overrides.get(motion, Path(batch_node["report"]))
        path = require_external_path(path, f"motion {motion} report")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"motion {motion} report is incomplete")
        if report.get("relative_path") != batch_node["relative_path"]:
            raise ValueError(f"motion {motion} report selects the wrong reference")
        if not report.get("k1_a_teacher_gate_passed"):
            raise ValueError(f"motion {motion} teacher gate did not pass")
        reports.append(report)
        selection.append(
            {
                "motion": motion,
                "relative_path": report["relative_path"],
                "source": "override" if motion in overrides else "batch",
                "report": str(path),
                "report_sha256": _sha256(path),
                "dataset": report["dataset"],
                "dataset_sha256": report["dataset_sha256"],
                "phase_samples": len(report["train"]["baseline"])
                + len(report["validation"]["baseline"]),
                "population": report["population"],
                "generations": report["generations"],
                "seed": report["seed"],
            }
        )
    return reports, selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_manifest", type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    batch_path = require_external_path(args.batch_manifest, "batch manifest")
    dataset_output = require_external_path(args.dataset_output, "dataset output")
    manifest_output = require_external_path(args.manifest_output, "manifest output")
    if dataset_output.exists() or manifest_output.exists():
        raise FileExistsError("assembly outputs must not already exist")
    overrides: dict[int, Path] = {}
    for value in args.override:
        motion, path = parse_override(value)
        if motion in overrides:
            raise ValueError(f"duplicate override for motion {motion}")
        overrides[motion] = path
    revision = _clean_revision()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    reports, selection = select_reports(batch, overrides)
    dataset = aggregate_teacher_datasets(reports, dataset_output)
    payload = {
        "schema_version": 1,
        "status": "complete_teacher_corpus_selected",
        "purpose": "k1_a_selected_full_corpus_teacher_demonstrations",
        "git_revision": revision,
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": _sha256(batch_path),
        "motion_count": len(reports),
        "teacher_gates_passed": len(reports),
        "selection": selection,
        "train": aggregate_split(reports, "train"),
        "validation": aggregate_split(reports, "validation"),
        "combined_dataset": dataset,
        "promotable": False,
        "promotion_blocker": "teacher corpus precedes BC, blind dense-grid evaluation, DAgger and three seeds",
        "selection_boundary": (
            "teacher validation selected the demonstrations; BC must use a new "
            "dense or randomized phase grid for model selection"
        ),
        "finished_wall_time_unix": time.time(),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
