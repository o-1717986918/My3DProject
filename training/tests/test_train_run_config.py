from tools.train_run import STAGES


def test_motion_reference_initialization_is_scoped_to_motion_stages():
    assert "reference_init_probability" not in STAGES["balance"]
    assert STAGES["motion_track"]["reference_init_probability"] == 0.20
    assert STAGES["motion_straight"]["reference_init_probability"] == 0.10
