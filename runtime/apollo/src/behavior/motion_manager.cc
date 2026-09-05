// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/behavior/motion_manager.h"

#include "src/decision/role_manager.h"

#include <algorithm>
#include <exception>
#include <variant>

namespace behavior {

MotionManager::MotionManager(const app::RuntimeConfig& config)
    : walk_runner_(
          config.resolve_asset_path("networks/walk/policy.onnx"),
          config.enable_fast_walk
              ? std::optional<std::filesystem::path>{config.fast_walk_model}
              : std::nullopt,
          config.enable_rapid_turn
              ? std::optional<std::filesystem::path>{config.rapid_turn_model}
              : std::nullopt),
      neutral_runner_(config.resolve_asset_path("keyframes/neutral.yaml")),
      getup_runner_(config.resolve_asset_path("networks/getup/policy.onnx")),
      parameterized_kick_enabled_(config.enable_parameterized_kick),
      learned_kick_enabled_(config.enable_learned_kick),
      learned_kick_shadow_(config.shadow_learned_kick) {
    if (parameterized_kick_enabled_) {
        try {
            kick_residual_runner_.emplace(
                config.resolve_asset_path("keyframes/kick_residual_table.yaml"));
        } catch (const std::exception&) {
            kick_residual_runner_.reset();
        }
        try {
            procedural_kick_runner_.emplace(
                config.resolve_asset_path("keyframes/procedural_kick.yaml"));
        } catch (const std::exception&) {
            procedural_kick_runner_.reset();
        }
        if (learned_kick_enabled_ || learned_kick_shadow_) {
            // An explicitly requested model must satisfy the tensor contract;
            // unlike optional bundled fallbacks, load failure is configuration
            // failure and must not be silently hidden.
            learned_kick_runner_.emplace(config.learned_kick_model);
        }
    }
}

MotionStepResult MotionManager::step(
    const world::WorldSnapshot& snapshot,
    const decision::HighLevelCommand& command,
    bool reset) {
    if (!std::holds_alternative<decision::KickCommand>(command)) {
        kick_residual_active_ = false;
        procedural_kick_active_ = false;
        learned_kick_active_ = false;
        learned_kick_shadow_valid_ = false;
        learned_kick_maximum_absolute_action_ = 0.0F;
    }
    if (std::holds_alternative<decision::BeamCommand>(command)) {
        reset_get_up_state();
        return {
            false, "BeamBypass", {}, SkillExecutionStatus::Completed,
            decision::MotionRequestKind::Unknown};
    }

    if (get_up_phase_ != GetUpPhase::Idle) {
        return step_get_up(snapshot, false);
    }

    if (const auto* walk = std::get_if<decision::WalkCommand>(&command)) {
        reset_get_up_state();
        const auto result = walk_runner_.step(snapshot, *walk, reset, walk->role_id);
        return {
            true,
            result.rapid_turn_active
                ? result.rapid_turn_mirrored
                    ? "RapidTurnV1RightMirror"
                    : "RapidTurnV1Left"
                : result.fast_walk_active ? "FastWalkV2" : "Walk",
            result.joint_targets,
            SkillExecutionStatus::Running,
            decision::MotionRequestKind::Walk};
    }

    if (const auto* kick = std::get_if<decision::KickCommand>(&command)) {
        reset_get_up_state();
        return step_kick(snapshot, *kick, reset);
    }

    if (std::holds_alternative<decision::NeutralCommand>(command)) {
        reset_get_up_state();
        const auto result = neutral_runner_.step(reset, snapshot.server_time);
        return {
            true, "Neutral", result.joint_targets,
            SkillExecutionStatus::Running,
            decision::MotionRequestKind::Neutral};
    }

    if (std::holds_alternative<decision::GetUpCommand>(command)) {
        return step_get_up(snapshot, reset);
    }

    return {
        false, "Idle", {}, SkillExecutionStatus::Rejected,
        decision::MotionRequestKind::Unknown};
}

MotionStepResult MotionManager::step_kick(
    const world::WorldSnapshot& snapshot,
    const decision::KickCommand& command,
    bool reset) {
    if (reset) {
        kick_start_time_ = snapshot.server_time;
        kick_profile_ = make_kick_execution_profile(
            snapshot, command,
            parameterized_kick_enabled_);
        kick_residual_active_ = parameterized_kick_enabled_ &&
            kick_residual_runner_.has_value() &&
            kick_residual_runner_->begin(snapshot, kick_profile_);
        procedural_kick_active_ = !kick_residual_active_ &&
            parameterized_kick_enabled_ &&
            procedural_kick_runner_.has_value() &&
            procedural_kick_runner_->begin(snapshot, kick_profile_);
        learned_kick_active_ = learned_kick_runner_.has_value() &&
            learned_kick_runner_->begin(snapshot, kick_profile_);
        learned_kick_shadow_valid_ = false;
        learned_kick_maximum_absolute_action_ = 0.0F;
    }

    const bool target_aware = command.mode != decision::KickMode::ForwardContact;
    const bool specialized_executor_active =
        kick_residual_active_ || procedural_kick_active_ ||
        (learned_kick_enabled_ && learned_kick_active_);
    // A decision-layer timeout may deliberately issue the original
    // ForwardContact mode while setting the fallback bit. Preserve that
    // provenance in the motion name even though no target-aware executor is
    // involved; otherwise match telemetry reports ordinary KickForward and
    // silently loses the reason this contact was selected.
    const bool use_forward_contact_fallback =
        command.allow_forward_contact_fallback &&
        (!target_aware || !specialized_executor_active);
    if (target_aware &&
        (!parameterized_kick_enabled_ ||
         !specialized_executor_active) &&
        !use_forward_contact_fallback) {
        kick_residual_active_ = false;
        procedural_kick_active_ = false;
        learned_kick_active_ = false;
        const auto hold = neutral_runner_.step(reset, snapshot.server_time);
        return {
            true,
            "RejectedTargetedKickHold",
            hold.joint_targets,
            SkillExecutionStatus::Rejected,
            decision::MotionRequestKind::Kick};
    }

    const double elapsed = std::max(0.0, snapshot.server_time - kick_start_time_);
    if (procedural_kick_active_ && procedural_kick_runner_.has_value()) {
        const auto result = procedural_kick_runner_->step(elapsed);
        return {
            true,
            result.finished ? "ProceduralKickHold" : "ProceduralKickExecute",
            result.joint_targets,
            result.finished
                ? SkillExecutionStatus::Completed
                : SkillExecutionStatus::Running,
            decision::MotionRequestKind::Kick,
        };
    }

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
    if (learned_kick_active_ && learned_kick_runner_.has_value()) {
        const auto learned = learned_kick_runner_->step(
            snapshot, kick_profile_, elapsed, result.joint_targets);
        learned_kick_shadow_valid_ = learned.valid;
        learned_kick_maximum_absolute_action_ =
            learned.maximum_absolute_action;
        if (learned_kick_enabled_ && learned.valid) {
            return {
                true,
                learned.finished ? "LearnedKickHold" : "LearnedKickExecute",
                learned.joint_targets,
                learned.finished
                    ? SkillExecutionStatus::Completed
                    : SkillExecutionStatus::Running,
                decision::MotionRequestKind::Kick,
            };
        }
        if (!learned.valid) {
            learned_kick_active_ = false;
        }
    }
    if (kick_residual_active_) {
        kick_residual_runner_->apply(elapsed, result.joint_targets);
    }
    const bool parameterized =
        kick_profile_.kind == KickProfileKind::ParameterizedContact;
    return {
        true,
        macro_complete
            ? (use_forward_contact_fallback
                ? "FallbackKickHold"
                : kick_residual_active_
                ? "ParameterizedResidualKickHold"
                : (parameterized ? "ParameterizedKickHold" : "KickHold"))
            : (drive_forward
                ? (use_forward_contact_fallback
                    ? "FallbackKickForward"
                    : kick_residual_active_
                    ? "ParameterizedResidualKickForward"
                    : (parameterized ? "ParameterizedKickForward" : "KickForward"))
                : (use_forward_contact_fallback
                    ? "FallbackKickStabilize"
                    : kick_residual_active_
                    ? "ParameterizedResidualKickStabilize"
                    : (parameterized ? "ParameterizedKickStabilize" : "KickStabilize"))),
        result.joint_targets,
        macro_complete
            ? SkillExecutionStatus::Completed
            : SkillExecutionStatus::Running,
        decision::MotionRequestKind::Kick,
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
    const SkillExecutionStatus status = timed_out
        ? SkillExecutionStatus::TimedOut
        : (result.upright
            ? SkillExecutionStatus::Completed
            : SkillExecutionStatus::Running);
    if (result.upright || timed_out) {
        reset_get_up_state();
    }
    return {
        true, "GetUpRL", result.joint_targets, status,
        decision::MotionRequestKind::GetUp};
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
