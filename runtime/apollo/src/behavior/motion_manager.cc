// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/behavior/motion_manager.h"

#include "src/decision/role_manager.h"

#include <algorithm>
#include <exception>
#include <variant>

namespace behavior {

MotionManager::MotionManager(const app::RuntimeConfig& config)
    : walk_runner_(config.resolve_asset_path("networks/walk/policy.onnx")),
      neutral_runner_(config.resolve_asset_path("keyframes/neutral.yaml")),
      getup_runner_(config.resolve_asset_path("networks/getup/policy.onnx")),
      parameterized_kick_enabled_(config.enable_parameterized_kick) {
    if (parameterized_kick_enabled_) {
        try {
            kick_residual_runner_.emplace(
                config.resolve_asset_path("keyframes/kick_residual_table.yaml"));
        } catch (const std::exception&) {
            parameterized_kick_enabled_ = false;
            kick_residual_runner_.reset();
        }
    }
}

MotionStepResult MotionManager::step(
    const world::WorldSnapshot& snapshot,
    const decision::HighLevelCommand& command,
    bool reset) {
    if (std::holds_alternative<decision::BeamCommand>(command)) {
        reset_get_up_state();
        return {false, "BeamBypass", {}};
    }

    if (get_up_phase_ != GetUpPhase::Idle) {
        return step_get_up(snapshot, false);
    }

    if (const auto* walk = std::get_if<decision::WalkCommand>(&command)) {
        reset_get_up_state();
        const auto result = walk_runner_.step(snapshot, *walk, reset, walk->role_id);
        return {true, "Walk", result.joint_targets};
    }

    if (const auto* kick = std::get_if<decision::KickCommand>(&command)) {
        reset_get_up_state();
        return step_kick(snapshot, *kick, reset);
    }

    if (std::holds_alternative<decision::NeutralCommand>(command)) {
        reset_get_up_state();
        const auto result = neutral_runner_.step(reset, snapshot.server_time);
        return {true, "Neutral", result.joint_targets};
    }

    if (std::holds_alternative<decision::GetUpCommand>(command)) {
        return step_get_up(snapshot, reset);
    }

    return {false, "Idle", {}};
}

MotionStepResult MotionManager::step_kick(
    const world::WorldSnapshot& snapshot,
    const decision::KickCommand& command,
    bool reset) {
    if (reset) {
        kick_start_time_ = snapshot.server_time;
        kick_profile_ = make_kick_execution_profile(
            snapshot, command, parameterized_kick_enabled_);
        kick_residual_active_ = parameterized_kick_enabled_ &&
            kick_residual_runner_.has_value() &&
            kick_residual_runner_->begin(snapshot, kick_profile_);
        if (parameterized_kick_enabled_ && !kick_residual_active_) {
            kick_profile_ = make_kick_execution_profile(snapshot, command, false);
        }
    }

    const double elapsed = std::max(0.0, snapshot.server_time - kick_start_time_);
    decision::WalkCommand walk_command;
    walk_command.target_absolute = false;
    walk_command.orientation_deg = 0.0;
    walk_command.orientation_absolute = false;
    walk_command.role_id = decision::RoleManager::ROLE_AP;

    const double drive_duration_s =
        kick_residual_active_ ? 0.65 : kick_profile_.drive_duration_s;
    const double total_duration_s =
        kick_residual_active_ ? 1.20 : kick_profile_.total_duration_s;
    const bool drive_forward = elapsed < drive_duration_s;
    const bool macro_complete = elapsed >= total_duration_s;
    walk_command.target_2d_m = drive_forward
        ? (kick_residual_active_
            ? std::array<double, 2>{0.50, -0.04}
            : kick_profile_.local_drive_target_m)
        : std::array<double, 2>{0.0, 0.0};

    auto result = walk_runner_.step(
        snapshot,
        walk_command,
        reset,
        kick_residual_active_ ? std::nullopt : walk_command.role_id);
    if (kick_residual_active_) {
        kick_residual_runner_->apply(elapsed, result.joint_targets);
    }
    const bool parameterized =
        kick_profile_.kind == KickProfileKind::ParameterizedContact;
    return {
        true,
        macro_complete
            ? (kick_residual_active_
                ? "ParameterizedResidualKickHold"
                : (parameterized ? "ParameterizedKickHold" : "KickHold"))
            : (drive_forward
                ? (kick_residual_active_
                    ? "ParameterizedResidualKickForward"
                    : (parameterized ? "ParameterizedKickForward" : "KickForward"))
                : (kick_residual_active_
                    ? "ParameterizedResidualKickStabilize"
                    : (parameterized ? "ParameterizedKickStabilize" : "KickStabilize"))),
        result.joint_targets,
    };
}

MotionStepResult MotionManager::step_get_up(
    const world::WorldSnapshot& snapshot,
    bool reset) {
    if (reset || get_up_phase_ == GetUpPhase::Idle) {
        enter_get_up_phase(GetUpPhase::Active, snapshot.server_time);
    }

    auto consume_phase_reset = [&]() {
        const bool phase_reset = get_up_phase_reset_pending_;
        get_up_phase_reset_pending_ = false;
        return phase_reset;
    };

    const auto result = getup_runner_.step(snapshot, consume_phase_reset());
    const bool timed_out =
        snapshot.server_time - get_up_start_time_ >= kGetUpTimeoutS;
    if (result.upright || timed_out) {
        reset_get_up_state();
    }
    return {true, "GetUpRL", result.joint_targets};
}

void MotionManager::enter_get_up_phase(
    GetUpPhase phase,
    double server_time) {
    get_up_phase_ = phase;
    get_up_phase_reset_pending_ = true;
    if (phase == GetUpPhase::Active) {
        get_up_start_time_ = server_time;
    }
}

void MotionManager::reset_get_up_state() {
    get_up_phase_ = GetUpPhase::Idle;
    get_up_phase_reset_pending_ = false;
    get_up_start_time_ = 0.0;
}

}  // namespace behavior
