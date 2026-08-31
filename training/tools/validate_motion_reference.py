#!/usr/bin/env python3
"""Validate a retargeted T1 reference before motion-prior training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from my3d_rl.motion_reference import validate_motion_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_motion_reference(args.reference)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
