from __future__ import annotations

import numpy as np
import pytest

from tools.optimize_server_kick_teacher import robust_objective


def test_robust_objective_rewards_success_without_ignoring_weakest_case() -> None:
    baseline = robust_objective(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    successful = robust_objective(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
    weak = robust_objective(np.array([2.0, -2.0]), np.array([0.0, 0.0]))

    assert successful > baseline > weak
    with pytest.raises(ValueError, match="aligned"):
        robust_objective(np.array([1.0]), np.array([1.0, 0.0]))
