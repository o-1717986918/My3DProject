import numpy as np
import pytest

from tools.evaluate_frozen_kick_expert_gate import paired_choice_counts


def test_paired_choice_counts_treats_each_release_as_one_trial() -> None:
    result = paired_choice_counts(
        np.asarray([1, 1, 0, 0], dtype=np.uint8),
        np.asarray([1, 0, 1, 0], dtype=np.uint8),
    )
    assert result == {
        "entries": 4,
        "both_succeed": 1,
        "gate_only_succeeds": 1,
        "baseline_only_succeeds": 1,
        "neither_succeeds": 1,
        "net_success_advantage": 0,
        "one_sided_paired_sign_test_p": 0.75,
    }


def test_paired_choice_counts_requires_repeated_discordant_wins() -> None:
    result = paired_choice_counts(np.ones(4), np.zeros(4))
    assert result["net_success_advantage"] == 4
    assert result["one_sided_paired_sign_test_p"] == 0.0625


def test_paired_choice_counts_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_choice_counts(np.ones(2), np.ones(3))
