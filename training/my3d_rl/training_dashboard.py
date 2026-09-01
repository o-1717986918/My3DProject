"""TensorBoard-compatible scalar logging for reproducible training runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any

import numpy as np


def scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return finite scalar metrics while preserving their hierarchical names."""
    result: dict[str, float] = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if array.shape != ():
            continue
        number = float(array)
        if np.isfinite(number):
            result[str(name)] = number
    return result


class TrainingDashboard:
    """Write live scalars without coupling the trainer to a hosted service."""

    def __init__(self, log_dir: Path, *, writer: Any | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if writer is None:
            try:
                from tensorboardX import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "tensorboardX is required; recreate the my3d-rl environment"
                ) from error
            writer = SummaryWriter(logdir=str(self.log_dir), flush_secs=5)
        self._writer = writer
        self._started = time.monotonic()

    def write(
        self,
        step: int,
        metrics: Mapping[str, Any],
        *,
        wall_time_unix: float | None = None,
    ) -> None:
        if step < 0:
            raise ValueError("dashboard step must be non-negative")
        for name, value in scalar_metrics(metrics).items():
            self._writer.add_scalar(name, value, global_step=step)
        self._writer.add_scalar(
            "system/elapsed_seconds", time.monotonic() - self._started, step
        )
        self._writer.add_scalar(
            "system/wall_time_unix",
            time.time() if wall_time_unix is None else wall_time_unix,
            step,
        )
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
