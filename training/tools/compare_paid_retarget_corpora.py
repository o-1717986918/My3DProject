#!/usr/bin/env python3
"""Pair and compare two complete PAiD-to-T1 K0 corpus reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    def values(name: str) -> np.ndarray:
        return np.asarray([record[name] for record in records], dtype=np.float64)

    return {
        "passed_count": int(sum(bool(record["passed"]) for record in records)),
        "clipped_value_count_total": int(np.sum(values("clipped_value_count"))),
        "maximum_abs_joint_limit_correction_rad": float(
            np.max(values("maximum_abs_joint_limit_correction_rad"))
        ),
        "maximum_joint_velocity_rad_s": float(
            np.max(values("maximum_joint_velocity_rad_s"))
        ),
        "mean_peak_kick_foot_relative_speed_m_s": float(
            np.mean(values("peak_kick_foot_relative_speed_m_s"))
        ),
        "minimum_peak_kick_foot_relative_speed_m_s": float(
            np.min(values("peak_kick_foot_relative_speed_m_s"))
        ),
        "minimum_kick_to_other_foot_peak_speed_ratio": float(
            np.min(values("kick_to_other_foot_peak_speed_ratio"))
        ),
        "maximum_root_tilt_rad": float(np.max(values("maximum_root_tilt_rad"))),
        "maximum_ground_offset_step_m": float(
            np.max(values("ground_offset_max_step_m"))
        ),
        "non_foot_pitch_contact_frames_total": int(
            np.sum(values("non_foot_pitch_contact_frames"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_records = {
        record["relative_path"]: record for record in baseline["records"]
    }
    candidate_records = {
        record["relative_path"]: record for record in candidate["records"]
    }
    if set(baseline_records) != set(candidate_records):
        raise SystemExit("corpus reports contain different relative paths")
    pairs = []
    for relative_path in sorted(baseline_records):
        first = baseline_records[relative_path]
        second = candidate_records[relative_path]
        if first.get("source_sha256") != second.get("source_sha256"):
            raise SystemExit(f"source hash mismatch for {relative_path}")
        pairs.append(
            {
                "relative_path": relative_path,
                "source_sha256": first.get("source_sha256"),
                "baseline_passed": first["passed"],
                "candidate_passed": second["passed"],
                "clipped_value_count_delta": (
                    second["clipped_value_count"] - first["clipped_value_count"]
                ),
                "maximum_joint_velocity_delta_rad_s": (
                    second["maximum_joint_velocity_rad_s"]
                    - first["maximum_joint_velocity_rad_s"]
                ),
                "peak_kick_speed_delta_m_s": (
                    second["peak_kick_foot_relative_speed_m_s"]
                    - first["peak_kick_foot_relative_speed_m_s"]
                ),
                "root_tilt_delta_rad": (
                    second["maximum_root_tilt_rad"]
                    - first["maximum_root_tilt_rad"]
                ),
            }
        )

    baseline_aggregate = _aggregate(list(baseline_records.values()))
    candidate_aggregate = _aggregate(list(candidate_records.values()))
    all_pass = (
        baseline_aggregate["passed_count"] == len(pairs)
        and candidate_aggregate["passed_count"] == len(pairs)
    )
    candidate_reduces_clipping = (
        candidate_aggregate["clipped_value_count_total"]
        < baseline_aggregate["clipped_value_count_total"]
    )
    preferred = candidate["method"] if all_pass and candidate_reduces_clipping else None
    payload = {
        "schema_version": 1,
        "purpose": "paid_to_t1_paired_k0_method_comparison",
        "status": "complete",
        "baseline_method": baseline["method"],
        "candidate_method": candidate["method"],
        "baseline_report": str(args.baseline.resolve()),
        "baseline_report_sha256": _sha256(args.baseline),
        "candidate_report": str(args.candidate.resolve()),
        "candidate_report_sha256": _sha256(args.candidate),
        "motion_count": len(pairs),
        "all_paired_sources_identical": True,
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "candidate_reduces_total_limit_clipping": candidate_reduces_clipping,
        "k1_primary_candidate": preferred,
        "selection_scope": (
            "K0 kinematic primary only; dynamic trackability and ball outcome "
            "remain unproven until K1/K2"
        ),
        "baseline_retained_for_ablation_and_fallback": True,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "pairs": pairs,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if preferred is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
