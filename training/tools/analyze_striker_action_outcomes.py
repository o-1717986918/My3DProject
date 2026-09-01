#!/usr/bin/env python3
"""Audit continuous action outcomes from aligned exact-CPU striker reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_aligned_reports(
    paths: list[Path],
) -> tuple[list[int], list[int], list[list[dict[str, Any]]]]:
    if len(paths) < 2:
        raise ValueError("at least two action reports are required")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = sources[0]
    if reference.get("purpose") != "striker_closed_loop_exact_cpu_evaluation":
        raise ValueError("reports must be exact CPU striker evaluations")
    reference_rows = [
        row for row in reference.get("rollouts", []) if bool(row.get("triggered"))
    ]
    seeds = [int(row["seed"]) for row in reference_rows]
    if not seeds:
        raise ValueError("reference report has no triggered rollouts")
    action_indices: list[int] = []
    aligned: list[list[dict[str, Any]]] = []
    for source in sources:
        action_index = int(source["environment_config"]["fixed_kick_prior_index"])
        rows_by_seed = {int(row["seed"]): row for row in source["rollouts"]}
        try:
            rows = [rows_by_seed[seed] for seed in seeds]
        except KeyError as error:
            raise ValueError("reports do not contain the same rollout seeds") from error
        if any(not bool(row.get("triggered")) for row in rows):
            raise ValueError("an aligned action report contains an untriggered rollout")
        action_indices.append(action_index)
        aligned.append(rows)
    if len(set(action_indices)) != len(action_indices):
        raise ValueError("action reports contain duplicate prior indices")
    return seeds, action_indices, aligned


def _selection_summary(
    action_indices: list[int],
    reports: list[list[dict[str, Any]]],
    metric: str,
) -> dict[str, Any]:
    values = np.asarray(
        [[float(row[metric]) for row in action_rows] for action_rows in reports]
    )
    if not np.isfinite(values).all():
        raise ValueError(f"metric {metric!r} contains non-finite values")
    success = np.asarray(
        [[bool(row["succeeded"]) for row in action_rows] for action_rows in reports]
    )
    fall = np.asarray(
        [[bool(row["fallen"]) for row in action_rows] for action_rows in reports]
    )
    selected = np.argmin(values, axis=0)
    columns = np.arange(values.shape[1])
    selected_success = success[selected, columns]
    selected_fall = fall[selected, columns]
    by_action = []
    for position, action_index in enumerate(action_indices):
        mask = selected == position
        by_action.append(
            {
                "action_prior_index": action_index,
                "selected": int(mask.sum()),
                "successes": int(selected_success[mask].sum()),
                "falls": int(selected_fall[mask].sum()),
            }
        )
    return {
        "metric": metric,
        "rollouts": int(values.shape[1]),
        "successes": int(selected_success.sum()),
        "success_rate": float(selected_success.mean()),
        "falls": int(selected_fall.sum()),
        "oracle_successes": int(success.any(axis=0).sum()),
        "oracle_success_rate": float(success.any(axis=0).mean()),
        "by_action": by_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument(
        "--metric", action="append", default=["final_goal_distance_m"]
    )
    args = parser.parse_args()
    seeds, action_indices, reports = _load_aligned_reports(args.reports)
    result = {
        "purpose": "striker_action_outcome_proxy_audit",
        "rollout_seed_start": min(seeds),
        "rollout_seed_end": max(seeds),
        "action_prior_indices": action_indices,
        "selectors": [
            _selection_summary(action_indices, reports, metric)
            for metric in dict.fromkeys(args.metric)
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
