from __future__ import annotations

import numpy as np
import pytest

from my3d_rl.training_dashboard import TrainingDashboard, scalar_metrics


class _Writer:
    def __init__(self):
        self.scalars = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, name, value, global_step):
        self.scalars.append((name, value, global_step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def test_scalar_metrics_keeps_only_finite_scalars():
    assert scalar_metrics(
        {"reward/x": np.array(2.5), "vector": [1, 2], "bad": np.nan}
    ) == {"reward/x": 2.5}


def test_dashboard_writes_metrics_and_system_time(tmp_path):
    writer = _Writer()
    dashboard = TrainingDashboard(tmp_path / "events", writer=writer)
    dashboard.write(7, {"eval/survival": 0.75}, wall_time_unix=123.0)
    dashboard.close()

    names = {name for name, _, _ in writer.scalars}
    assert names == {
        "eval/survival",
        "system/elapsed_seconds",
        "system/wall_time_unix",
    }
    assert all(step == 7 for _, _, step in writer.scalars)
    assert writer.flushed and writer.closed
    with pytest.raises(ValueError):
        dashboard.write(-1, {})
