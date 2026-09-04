from tools.train_run import STAGES


def test_motion_reference_initialization_is_scoped_to_motion_stages():
    assert "reference_init_probability" not in STAGES["balance"]
    assert STAGES["motion_track"]["reference_init_probability"] == 0.20
    assert STAGES["motion_straight"]["reference_init_probability"] == 0.10
    assert STAGES["reference_residual"]["reference_init_probability"] == 1.0
    assert STAGES["reference_residual"]["fixed_command"] == [1.8, 0.0, 0.0]


def test_soccer_omni_stage_trains_switches_and_full_planar_commands():
    stage = STAGES["soccer_omni"]

    assert stage["lin_vel_x"][0] < 0.0
    assert stage["lin_vel_y"][0] < 0.0 < stage["lin_vel_y"][1]
    assert stage["ang_vel_yaw"][0] < 0.0 < stage["ang_vel_yaw"][1]
    assert stage["command_resample_steps"] < 100
    assert stage["stand_probability"] > 0.0
    assert stage["push_enable"] is True
    assert stage["action_delay_max_steps"] == 1
    assert stage["reward.lateral_tracking"] < 0.0
    assert stage["reward.yaw_rate_error"] < 0.0


def test_axis_aligned_soccer_stage_matches_frozen_command_suite():
    stage = STAGES["soccer_omni_axis"]

    assert stage["axis_aligned_command_probability"] == 0.5
    assert stage["command_resample_steps"] == 100
    assert stage["reward.tracking_yaw"] > STAGES["soccer_omni"][
        "reward.tracking_yaw"
    ]
