import math

import pytest

from tools.compare_soccer_motion_policy_evaluations import (
    _one_sided_exact_p,
    _record_key,
)


def test_one_sided_exact_p_matches_small_exact_tail():
    assert _one_sided_exact_p(2, 0) == pytest.approx(0.25)
    assert _one_sided_exact_p(1, 1) == pytest.approx(0.75)
    assert _one_sided_exact_p(0, 0) == 1.0


def test_one_sided_exact_p_is_stable_beyond_float_power_range():
    result = _one_sided_exact_p(800, 400)

    assert math.isfinite(result)
    assert 0.0 <= result < 0.05


def test_one_sided_exact_p_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        _one_sided_exact_p(-1, 2)


def test_record_key_pairs_the_same_reset_perturbation_only():
    record = {
        "relative_path": "motion.t1.npz",
        "start_frame": 7,
        "length": 100,
        "perturbation_seed": 123,
    }

    assert _record_key(record) == ("motion.t1.npz", 7, 100, 123)
    assert _record_key({**record, "perturbation_seed": 124}) != _record_key(record)


def test_record_key_keeps_legacy_unperturbed_reports_compatible():
    assert _record_key(
        {"relative_path": "motion.t1.npz", "start_frame": 7, "length": 100}
    ) == ("motion.t1.npz", 7, 100, None)
