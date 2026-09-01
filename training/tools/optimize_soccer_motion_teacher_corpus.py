#!/usr/bin/env python3
"""Run and aggregate exact-CPU phase teachers across the locked corpus."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from my3d_rl.soccer_motion_corpus import load_soccer_motion_corpus


REPOSITORY_ROOT = Path(__file__).parents[2]
SINGLE_TEACHER_TOOL = Path(__file__).with_name("optimize_soccer_motion_teacher.py")
DATASET_KEYS = (
    "observation",
    "base_action",
    "teacher_action",
    "qpos",
    "split",
    "motion",
    "start_frame",
    "reference_frame",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision_and_clean() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
        encoding="utf-8",
    )
    if status:
        raise RuntimeError("formal corpus teacher generation requires a clean Git tree")
    return revision


def require_external_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{label} must stay outside the repository")


def build_motion_command(
    *,
    corpus_root: Path,
    relative_path: str,
    checkpoint: Path,
    profile: str,
    phase_samples: int,
    population: int,
    generations: int,
    seed: int,
    motion_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SINGLE_TEACHER_TOOL),
        str(corpus_root),
        "--relative-path",
        relative_path,
        "--checkpoint",
        str(checkpoint),
        "--profile",
        profile,
        "--phase-samples",
        str(phase_samples),
        "--population",
        str(population),
        "--generations",
        str(generations),
        "--seed",
        str(seed),
        "--output",
        str(motion_dir / "report.json"),
        "--dataset-output",
        str(motion_dir / "teacher-dataset.npz"),
        "--tensorboard-log-dir",
        str(motion_dir / "tensorboard"),
    ]


def aggregate_teacher_datasets(
    reports: list[dict[str, Any]], output: Path
) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in DATASET_KEYS}
    for report in reports:
        dataset = Path(report["dataset"])
        if _sha256(dataset) != report["dataset_sha256"]:
            raise ValueError(f"teacher dataset hash mismatch: {dataset}")
        with np.load(dataset, allow_pickle=False) as archive:
            missing = set(DATASET_KEYS) - set(archive.files)
            if missing:
                raise ValueError(f"teacher dataset is missing {sorted(missing)}")
            for key in DATASET_KEYS:
                arrays[key].append(archive[key])
    combined = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(temporary, **combined)
    temporary.replace(output)
    return {
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "frames": int(combined["observation"].shape[0]),
        "motions": int(len(set(combined["motion"].tolist()))),
        "train_frames": int(np.sum(combined["split"] == 0)),
        "validation_frames": int(np.sum(combined["split"] == 1)),
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run_motion(
    *,
    motion: int,
    relative_path: str,
    command: list[str],
    motion_dir: Path,
    revision: str,
    population: int,
    generations: int,
    seed: int,
) -> dict[str, Any]:
    report_path = motion_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = (
            report.get("git_revision") == revision
            and report.get("relative_path") == relative_path
            and report.get("population") == population
            and report.get("generations") == generations
            and report.get("seed") == seed
            and report.get("status") == "complete"
        )
        if not expected:
            raise ValueError(f"existing motion report is incompatible: {report_path}")
        return report
    if motion_dir.exists():
        preserved = motion_dir.with_name(
            f"{motion_dir.name}.incomplete-{time.time_ns()}"
        )
        if preserved.parent != motion_dir.parent:
            raise RuntimeError("refusing to preserve an incomplete run outside its parent")
        motion_dir.replace(preserved)
    motion_dir.mkdir(parents=True, exist_ok=True)
    console_path = motion_dir / "console.log"
    environment = os.environ.copy()
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["PYTHONPATH"] = "training"
    started = time.monotonic()
    with console_path.open("w", encoding="utf-8") as console:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"motion {motion} teacher failed with exit {result.returncode}; "
            f"see {console_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["batch_elapsed_seconds"] = time.monotonic() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", default="soccer_motion_residual_v3")
    parser.add_argument("--phase-samples", type=int, default=8)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260910)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    if min(
        args.phase_samples,
        args.population,
        args.generations,
        args.max_workers,
    ) < 1:
        raise ValueError("batch teacher counts must be positive")
    if args.population < 2 or args.phase_samples < 2:
        raise ValueError("teacher search requires population and phase samples >= 2")
    run_dir = require_external_path(args.run_dir, "run directory")
    corpus_root = require_external_path(args.corpus_root, "corpus root")
    checkpoint = require_external_path(args.checkpoint, "checkpoint")
    revision = _git_revision_and_clean()
    corpus = load_soccer_motion_corpus(corpus_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "batch-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "purpose": "k1_a_full_corpus_exact_cpu_phase_teachers",
        "git_revision": revision,
        "corpus_root": str(corpus_root),
        "checkpoint": str(checkpoint),
        "profile": args.profile,
        "phase_samples": args.phase_samples,
        "population": args.population,
        "generations": args.generations,
        "base_seed": args.base_seed,
        "max_workers": args.max_workers,
        "motion_count": corpus.motion_count,
        "motions": {},
        "started_wall_time_unix": time.time(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "git_revision",
            "corpus_root",
            "checkpoint",
            "profile",
            "phase_samples",
            "population",
            "generations",
            "base_seed",
            "motion_count",
        )
        if any(previous.get(key) != manifest[key] for key in invariant_keys):
            raise ValueError("existing batch manifest has incompatible configuration")
        manifest["motions"] = previous.get("motions", {})
        manifest["started_wall_time_unix"] = previous.get(
            "started_wall_time_unix", manifest["started_wall_time_unix"]
        )
    _write_manifest(manifest_path, manifest)

    jobs: dict[Any, tuple[int, str, Path]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for motion, relative_path in enumerate(corpus.relative_paths):
            motion_dir = run_dir / f"motion-{motion:02d}"
            seed = args.base_seed + motion
            command = build_motion_command(
                corpus_root=corpus_root,
                relative_path=relative_path,
                checkpoint=checkpoint,
                profile=args.profile,
                phase_samples=args.phase_samples,
                population=args.population,
                generations=args.generations,
                seed=seed,
                motion_dir=motion_dir,
            )
            future = pool.submit(
                _run_motion,
                motion=motion,
                relative_path=relative_path,
                command=command,
                motion_dir=motion_dir,
                revision=revision,
                population=args.population,
                generations=args.generations,
                seed=seed,
            )
            jobs[future] = (motion, relative_path, motion_dir)
        try:
            for future in as_completed(jobs):
                motion, relative_path, motion_dir = jobs[future]
                report = future.result()
                manifest["motions"][str(motion)] = {
                    "relative_path": relative_path,
                    "report": str((motion_dir / "report.json").resolve()),
                    "report_sha256": _sha256(motion_dir / "report.json"),
                    "dataset": report["dataset"],
                    "dataset_sha256": report["dataset_sha256"],
                    "teacher_gate_passed": report["k1_a_teacher_gate_passed"],
                    "train": report["train"]["summary"],
                    "validation": report["validation"]["summary"],
                }
                _write_manifest(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            "motion": motion,
                            "relative_path": relative_path,
                            "teacher_gate_passed": report[
                                "k1_a_teacher_gate_passed"
                            ],
                            "validation_survival_delta": report["validation"][
                                "summary"
                            ]["mean_survival_fraction_delta"],
                            "completed": len(manifest["motions"]),
                            "total": corpus.motion_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except BaseException as error:
            manifest.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_wall_time_unix": time.time(),
                }
            )
            _write_manifest(manifest_path, manifest)
            raise

    ordered_reports = [
        json.loads(
            Path(manifest["motions"][str(motion)]["report"]).read_text(
                encoding="utf-8"
            )
        )
        for motion in range(corpus.motion_count)
    ]
    passed = sum(report["k1_a_teacher_gate_passed"] for report in ordered_reports)
    combined_dataset = None
    if passed == corpus.motion_count:
        combined_dataset = aggregate_teacher_datasets(
            ordered_reports, run_dir / "teacher-corpus.npz"
        )
    manifest.update(
        {
            "status": (
                "complete_teacher_gate_passed"
                if passed == corpus.motion_count
                else "complete_teacher_gate_incomplete"
            ),
            "teacher_gates_passed": int(passed),
            "teacher_gates_total": corpus.motion_count,
            "combined_dataset": combined_dataset,
            "promotable": False,
            "promotion_blocker": "full-corpus teachers precede BC, DAgger and three-seed gates",
            "finished_wall_time_unix": time.time(),
        }
    )
    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
