#!/usr/bin/env python3
"""Paired statistical comparison for fixed-grid soccer policy evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def _one_sided_exact_p(improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, count)
        for count in range(improvements, discordant + 1)
    ) / (2.0**discordant)


def _record_key(record: dict) -> tuple[str, int, int, int | None]:
    return (
        record["relative_path"],
        record["start_frame"],
        record["length"],
        record.get("perturbation_seed"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if baseline.get("purpose") != "k1_exact_cpu_fixed_motion_phase_grid":
        raise ValueError("baseline is not an exact CPU fixed-grid report")
    if candidate.get("purpose") != baseline["purpose"]:
        raise ValueError("candidate report has another purpose")

    baseline_records = {
        _record_key(record): record for record in baseline["records"]
    }
    candidate_records = {
        _record_key(record): record for record in candidate["records"]
    }
    if set(baseline_records) != set(candidate_records):
        raise ValueError("fixed evaluation grids differ")
    transitions = {
        "both_completed": 0,
        "candidate_only_completed": 0,
        "baseline_only_completed": 0,
        "both_failed": 0,
    }
    pairs = []
    for record_key in sorted(baseline_records):
        first = baseline_records[record_key]
        second = candidate_records[record_key]
        if first["completed"] and second["completed"]:
            transition = "both_completed"
        elif second["completed"]:
            transition = "candidate_only_completed"
        elif first["completed"]:
            transition = "baseline_only_completed"
        else:
            transition = "both_failed"
        transitions[transition] += 1
        pairs.append(
            {
                "relative_path": record_key[0],
                "start_frame": record_key[1],
                "perturbation_seed": record_key[3],
                "transition": transition,
                "survival_fraction_delta": (
                    second["survival_fraction"] - first["survival_fraction"]
                ),
            }
        )
    trials = len(pairs)
    baseline_successes = sum(record["completed"] for record in baseline_records.values())
    candidate_successes = sum(record["completed"] for record in candidate_records.values())
    improvements = transitions["candidate_only_completed"]
    regressions = transitions["baseline_only_completed"]
    p_value = _one_sided_exact_p(improvements, regressions)
    candidate_interval = _wilson(candidate_successes, trials)
    tracking_ok = (
        candidate["mean_joint_tracking_rmse_rad"]
        <= baseline["mean_joint_tracking_rmse_rad"] + 0.01
        and candidate["mean_foot_contact_agreement"]
        >= baseline["mean_foot_contact_agreement"] - 0.02
    )
    promotion_passed = (
        improvements > regressions and p_value <= 0.05 and tracking_ok
    )
    survival_deltas = np.asarray(
        [pair["survival_fraction_delta"] for pair in pairs], dtype=np.float64
    )
    survival_improvements = int(np.sum(survival_deltas > 1.0e-12))
    survival_regressions = int(np.sum(survival_deltas < -1.0e-12))
    survival_ties = trials - survival_improvements - survival_regressions
    survival_sign_p = _one_sided_exact_p(
        survival_improvements, survival_regressions
    )
    mean_survival_delta = float(np.mean(survival_deltas))
    curriculum_advance_passed = (
        candidate_successes >= baseline_successes
        and mean_survival_delta >= 0.015
        and survival_improvements > survival_regressions
        and survival_sign_p <= 0.01
        and tracking_ok
    )
    payload = {
        "schema_version": 1,
        "purpose": "k1_paired_exact_cpu_policy_comparison",
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": _sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "paired_trials": trials,
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "completion_rate_delta": (
            candidate_successes - baseline_successes
        ) / trials,
        "candidate_completion_wilson_95": {
            "lower": candidate_interval[0],
            "upper": candidate_interval[1],
        },
        "transitions": transitions,
        "one_sided_exact_mcnemar_p": p_value,
        "tracking_tolerance_passed": tracking_ok,
        "mean_survival_fraction_delta": mean_survival_delta,
        "survival_transitions": {
            "improvements": survival_improvements,
            "regressions": survival_regressions,
            "ties": survival_ties,
        },
        "one_sided_exact_survival_sign_p": survival_sign_p,
        "promotion_rule": (
            "paired improvement exceeds regressions, one-sided exact McNemar "
            "p<=0.05, RMSE regression<=0.01 rad, contact regression<=0.02"
        ),
        "promotion_passed": promotion_passed,
        "curriculum_advance_rule": (
            "no net completion loss, mean survival delta>=0.015, survival "
            "improvements exceed regressions with one-sided exact sign p<=0.01, "
            "and tracking tolerance passes; authorizes PPO initialization only"
        ),
        "curriculum_advance_passed": curriculum_advance_passed,
        "decision": "promote" if promotion_passed else "retain_experimental",
        "pairs": pairs,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
