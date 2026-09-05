from tools.train_run import STAGES, _effective_timesteps


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


def test_stable_motion_curriculum_decomposes_turn_forward_then_lateral():
    turn = STAGES["rapid_turn"]
    forward = STAGES["stable_forward"]
    lateral = STAGES["soccer_lateral"]

    assert turn["lin_vel_x"] == [0.0, 0.0]
    assert turn["lin_vel_y"] == [0.0, 0.0]
    assert turn["ang_vel_yaw"][0] < -0.5 < 0.5 < turn["ang_vel_yaw"][1]
    assert forward["lin_vel_x"][1] >= 1.5
    assert forward["lin_vel_y"] == [0.0, 0.0]
    assert forward["ang_vel_yaw"] == [0.0, 0.0]
    assert lateral["axis_aligned_command_probability"] >= 0.5
    assert lateral["lin_vel_y"][0] < 0.0 < lateral["lin_vel_y"][1]
    assert turn["reward.foot_slip"] < 0.0
    assert forward["reward.foot_slip"] < 0.0
    assert lateral["reward.foot_slip"] < 0.0


def test_fast_walk_recovery_trains_forward_and_both_turns_without_lateral():
    stage = STAGES["fast_walk_recovery"]

    assert stage["lin_vel_x"][1] >= 1.5
    assert stage["lin_vel_y"] == [0.0, 0.0]
    assert stage["ang_vel_yaw"][0] < 0.0 < stage["ang_vel_yaw"][1]
    assert stage["axis_aligned_command_probability"] == 1.0
    assert stage["axis_command_weights"][1] == 0.0
    assert stage["yaw_negative_probability"] > 0.5
    assert stage["reward.fall"] <= -120.0


def test_effective_timesteps_rounds_to_complete_ppo_epochs():
    assert _effective_timesteps(196_608, 196_608) == 196_608
    assert _effective_timesteps(1_048_576, 196_608) == 1_179_648


def test_right_turn_expert_is_scoped_to_negative_yaw_teacher_data():
    stage = STAGES["right_turn_expert"]

    assert stage["lin_vel_x"] == [0.0, 0.0]
    assert stage["lin_vel_y"] == [0.0, 0.0]
    assert stage["ang_vel_yaw"][1] < 0.0
    assert stage["command_resample_steps"] <= 100
    assert stage["push_enable"] is True
    assert stage["reward.fall"] < STAGES["rapid_turn"]["reward.fall"]


def test_fast_walk_transition_recovery_trains_non_nominal_handoffs():
    stage = STAGES["fast_walk_transition_recovery"]

    assert stage["lin_vel_y"] == [0.0, 0.0]
    assert stage["ang_vel_yaw"] == [0.0, 0.0]
    assert stage["stand_probability"] > 0.0
    assert stage["command_resample_steps"] < 100
    assert stage["reset_policy_action_noise"] > 0.0
    assert stage["reset_joint_velocity_noise"] > 0.0
    assert stage["action_delay_max_steps"] == 1
    assert stage["reward.fall"] <= -150.0


def test_lateral_left_expert_is_pure_and_transition_robust():
    stage = STAGES["lateral_left_expert"]

    assert stage["lin_vel_x"] == [0.0, 0.0]
    assert stage["lin_vel_y"][0] > 0.0
    assert stage["lin_vel_y"][1] >= 0.45
    assert stage["ang_vel_yaw"] == [0.0, 0.0]
    assert stage["stand_probability"] > 0.0
    assert stage["reset_policy_action_noise"] > 0.0
    assert stage["reset_joint_velocity_noise"] > 0.0
    assert stage["command_resample_steps"] < 100
    assert stage["reward.lateral_tracking"] <= -10.0
    assert stage["reward.fall"] <= -150.0


def test_lateral_speed_continuation_increases_demand_without_coupled_axes():
    stage = STAGES["lateral_left_speed"]
    robust = STAGES["lateral_left_expert"]

    assert stage["lin_vel_x"] == [0.0, 0.0]
    assert stage["ang_vel_yaw"] == [0.0, 0.0]
    assert stage["lin_vel_y"][0] >= 0.30
    assert stage["lin_vel_y"][1] > robust["lin_vel_y"][1]
    assert stage["reward.lateral_tracking"] < robust["reward.lateral_tracking"]
    assert stage["reset_joint_velocity_noise"] < robust["reset_joint_velocity_noise"]
    assert stage["reset_policy_action_noise"] < robust["reset_policy_action_noise"]
    assert stage["reward.fall"] <= -150.0


def test_lateral_yaw_lock_targets_observed_drift_before_more_speed():
    stage = STAGES["lateral_left_yaw_lock"]
    baseline = STAGES["lateral_left_expert"]

    assert stage["lin_vel_x"] == [0.0, 0.0]
    assert stage["lin_vel_y"] == baseline["lin_vel_y"]
    assert stage["ang_vel_yaw"] == [0.0, 0.0]
    assert stage["reward.tracking_yaw"] > baseline["reward.tracking_yaw"]
    assert stage["reward.yaw_rate_error"] < baseline["reward.yaw_rate_error"]
    assert stage["reset_policy_action_noise"] > 0.0
