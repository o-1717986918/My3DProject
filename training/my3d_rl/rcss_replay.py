"""Read authoritative ball poses from RCSSServerMJ RSMP replay logs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TypeAlias

import numpy as np


SExpression: TypeAlias = list["SExpression | str"]
_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def _parse_sexpression(line: str) -> SExpression:
    tokens = _TOKEN.findall(line)
    cursor = 0

    def parse() -> SExpression:
        nonlocal cursor
        if cursor >= len(tokens) or tokens[cursor] != "(":
            raise ValueError("invalid RSMP s-expression")
        cursor += 1
        result: SExpression = []
        while cursor < len(tokens) and tokens[cursor] != ")":
            if tokens[cursor] == "(":
                result.append(parse())
            else:
                result.append(tokens[cursor])
                cursor += 1
        if cursor >= len(tokens):
            raise ValueError("unterminated RSMP s-expression")
        cursor += 1
        return result

    result = parse()
    if cursor != len(tokens):
        raise ValueError("trailing RSMP tokens")
    return result


def _node_children(node: SExpression) -> list[SExpression]:
    return [
        item
        for item in node[1:]
        if isinstance(item, list) and item and item[0] == "nd"
    ]


def _slt_position(node: SExpression) -> np.ndarray | None:
    for item in node[1:]:
        if isinstance(item, list) and item and item[0] == "SLT":
            values = np.asarray(item[1:], dtype=np.float64)
            if values.shape != (16,) or not np.isfinite(values).all():
                raise ValueError("invalid ball SLT in RSMP replay")
            return values[12:15]
    return None


def load_ball_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return every replay time and the last authoritative ball position.

    RSMP diff frames preserve the full scene-graph shape while omitting an SLT
    when a node moved less than the server threshold. The ball is the first
    child of the scene root in both full and diff frames, so its cached pose can
    be updated without interpreting any robot mesh nodes.
    """

    times: list[float] = []
    positions: list[np.ndarray] = []
    current_position: np.ndarray | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            packet = _parse_sexpression(line)
            if (
                len(packet) < 3
                or not isinstance(packet[1], list)
                or not isinstance(packet[2], list)
                or packet[1][0] != ["gt", "1", "0"]
                or packet[2][0] != ["sg", "1", "0"]
            ):
                raise ValueError(f"invalid RSMP replay frame {line_number}")
            time_s = float(packet[1][1])
            scene = packet[2]
            if len(scene) < 3 or scene[1] not in {"full", "diff"}:
                raise ValueError(f"missing scene graph in replay frame {line_number}")
            root = scene[2]
            if not isinstance(root, list) or not root or root[0] != "nd":
                raise ValueError(f"invalid scene root in replay frame {line_number}")
            root_children = _node_children(root)
            if not root_children:
                raise ValueError(f"missing ball node in replay frame {line_number}")
            ball_description = root_children[0]
            ball_children = _node_children(ball_description)
            if not ball_children:
                raise ValueError(f"missing ball transform in replay frame {line_number}")
            update = _slt_position(ball_children[0])
            if update is not None:
                current_position = update
            if current_position is None:
                raise ValueError("RSMP replay starts without a full ball pose")
            if times and time_s <= times[-1]:
                raise ValueError("RSMP replay time must increase")
            times.append(time_s)
            positions.append(current_position.copy())
    if not times:
        raise ValueError("RSMP replay contains no frames")
    return np.asarray(times, dtype=np.float64), np.stack(positions)
