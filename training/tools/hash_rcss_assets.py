#!/usr/bin/env python3
"""Create a provenance manifest for the installed RCSSServerMJ T1 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path(
    "/home/win98/.local/pipx/venvs/rcsssmj/lib/python3.10/site-packages/rcsssmj/resources"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_manifest(root: Path) -> dict[str, object]:
    selected = [root / "robots" / "T1", root / "environments" / "soccer"]
    files = []
    for selected_root in selected:
        if not selected_root.is_dir():
            raise FileNotFoundError(
                f"required asset directory missing: {selected_root}"
            )
        for path in sorted(item for item in selected_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    return {
        "schema_version": 1,
        "source_root": str(root),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_manifest(args.root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
