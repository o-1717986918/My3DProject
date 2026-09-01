from tools.compare_soccer_motion_policy_evaluations import _record_key


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
