#!/usr/bin/env python3
"""Audit a local PAiD clone without copying its non-commercial motion assets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys

from my3d_rl.paid_motion import (
    PAID_SCHEMA_REVISION,
    PAID_SOURCE_LICENSE,
    file_sha256,
    load_paid_motion,
    source_foot_contact,
    validate_paid_clip,
)


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paid_root", type=Path, help="external PAiD repository clone")
    parser.add_argument(
        "--motion-root", type=Path, help="defaults to PAiD_ROOT/motions"
    )
    parser.add_argument("--revision", default=PAID_SCHEMA_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=13)
    args = parser.parse_args()

    paid_root = args.paid_root.resolve()
    motion_root = (args.motion_root or paid_root / "motions").resolve()
    try:
        motion_root.relative_to(paid_root)
    except ValueError as exc:
        raise SystemExit("motion-root must stay inside the audited PAiD clone") from exc
    actual_revision = _git_revision(paid_root)
    if actual_revision != args.revision:
        raise SystemExit(
            f"PAiD revision {actual_revision} != schema revision {args.revision}"
        )
    license_path = paid_root / "LICENSE.md"
    license_text = license_path.read_text(encoding="utf-8")
    if "Attribution-NonCommercial 4.0 International" not in license_text:
        raise SystemExit("PAiD LICENSE.md does not declare the audited CC BY-NC 4.0")

    paths = sorted(motion_root.rglob("*.npz"))
    if len(paths) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} PAiD motions, found {len(paths)}"
        )
    records = []
    for path in paths:
        clip = load_paid_motion(path)
        validation = validate_paid_clip(clip)
        _, contact = source_foot_contact(clip)
        records.append(
            {
                "relative_path": str(path.relative_to(paid_root)),
                "sha256": file_sha256(path),
                **validation,
                "source_contact": contact,
            }
        )

    labels = Counter(record["kick_leg"] for record in records)
    groups = Counter(Path(record["relative_path"]).parent.name for record in records)
    payload = {
        "schema_version": 1,
        "purpose": "paid_local_motion_provenance_and_schema_audit",
        "status": "complete",
        "passed": True,
        "repository": "https://github.com/TeleHuman/HumanoidSoccer",
        "external_root": str(paid_root),
        "revision": actual_revision,
        "license": PAID_SOURCE_LICENSE,
        "license_file": str(license_path),
        "license_sha256": file_sha256(license_path),
        "reuse_policy": (
            "local attributed non-commercial research only; motion files and "
            "derived artifacts must not be committed or redistributed"
        ),
        "conversion_boundary": (
            "array order is valid only for the pinned official exporter revision"
        ),
        "motion_count": len(records),
        "group_counts": dict(sorted(groups.items())),
        "kick_leg_counts": dict(sorted(labels.items())),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "motions": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
