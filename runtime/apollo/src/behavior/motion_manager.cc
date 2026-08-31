// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/behavior/motion_manager.h"

#include "src/decision/role_manager.h"

#include <algorithm>
#include <variant>

namespace behavior {

MotionManager::MotionManager(const app::RuntimeConfig& config)
    : walk_runner_(config.resolve_asset_path("networks/walk/policy.onnx")),
      neutral_runner_(config.resolve_asset_path("keyframes/neutral.yaml")),
      getup_runner_(config.resolve_asset_path("networks/getup/policy.onnx")) {}

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

    if (std::holds_alternative<decision::KickCommand>(command)) {
        reset_get_up_state();
        return step_kick(snapshot, reset);
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
    bool reset) {
    if (reset) {
        kick_start_time_ = snapshot.server_time;
    }

    const double elapsed = std::max(0.0, snapshot.server_time - kick_start_time_);
    decision::WalkCommand walk_command;
    walk_command.target_absolute = false;
    walk_command.orientation_deg = 0.0;
    walk_command.orientation_absolute = false;
    walk_command.role_id = decision::RoleManager::ROLE_AP;

    const bool drive_forward = elapsed < kKickForwardDurationS;
    const bool macro_complete = elapsed >= kKickTotalDurationS;
    walk_command.target_2d_m = drive_forward
        ? std::array<double, 2>{0.50, -0.04}
        : std::array<double, 2>{0.0, 0.0};

    const auto result = walk_runner_.step(
        snapshot, walk_command, reset, walk_command.role_id);
    return {
        true,
        macro_complete ? "KickHold" : (drive_forward ? "KickForward" : "KickStabilize"),
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
