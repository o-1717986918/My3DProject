import numpy as np
import pytest

from tools.evaluate_transition_kick_onnx import held_out_rows


def test_held_out_rows_rejects_rollout_leakage() -> None:
    with pytest.raises(ValueError, match="leaks rollout IDs"):
        held_out_rows(np.array([0, 1, 1]), np.array([3, 3, 4]))


def test_held_out_rows_returns_the_complete_validation_partition() -> None:
    rows = held_out_rows(np.array([0, 0, 1, 1]), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(rows, [2, 3])
