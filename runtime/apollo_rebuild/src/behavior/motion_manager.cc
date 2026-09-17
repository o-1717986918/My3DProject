// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/behavior/motion_manager.h"

#include "src/decision/role_manager.h"

#include <algorithm>
#include <memory>
#include <exception>
#include <string>
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
    if (config.enable_dynamic_pass) {
        dynamic_pass_runner_ = std::make_unique<DynamicPassRunner>(
            config.resolve_asset_path("networks/dynamic_pass/selector.onnx"),
            config.dynamic_pass_forced_rollout_id);
    }
    if (parameterized_kick_enabled_) {
        try {
            kick_residual_runner_.emplace(
                config.resolve_asset_path("keyframes/kick_residual_table.yaml"));
        } catch (const std::exception&) {
            kick_residual_runner_.reset();
        }
        procedural_kick_runner_.emplace(
            config.resolve_asset_path("keyframes/procedural_kick.yaml"));
        if (learned_kick_enabled_ || learned_kick_shadow_) {
            learned_kick_runner_.emplace(config.learned_kick_model);
        }
    }
}

MotionStepResult MotionManager::step(
    const world::WorldSnapshot& snapshot,
    const decision::HighLevelCommand& command,
    bool reset) {
    if (std::holds_alternative<decision::BeamCommand>(command)) {
        reset_get_up_state();
        reset_kick_state();
        reset_dynamic_pass_state();
        return {false, "BeamBypass", {}};
    }

    if (get_up_phase_ != GetUpPhase::Idle) {
        reset_kick_state();
        reset_dynamic_pass_state();
        return step_get_up(snapshot, false);
    }

    if (const auto* walk = std::get_if<decision::WalkCommand>(&command)) {
        reset_get_up_state();
        reset_kick_state();
        if (dynamic_pass_runner_) {
            const bool eligible =
                walk->role_id == decision::RoleManager::ROLE_AP &&
                snapshot.play_mode == world::PlayMode::PlayOn;
            const auto activation = dynamic_pass_runner_->consider(snapshot, eligible);
            if (dynamic_pass_runner_->active()) {
                const double elapsed = dynamic_pass_runner_->elapsed_s(snapshot.server_time);
                decision::WalkCommand pass_base;
                pass_base.target_absolute = false;
                pass_base.target_2d_m = elapsed < 0.65
                    ? std::array<double, 2>{0.50, -0.04}
                    : std::array<double, 2>{0.0, 0.0};
                pass_base.orientation_deg = 0.0;
                pass_base.orientation_absolute = false;
                pass_base.role_id = decision::RoleManager::ROLE_AP;
                const auto stable = walk_runner_.step(
                    snapshot, pass_base,
                    activation.started || walk_reset_pending_,
                    pass_base.role_id);
                walk_reset_pending_ = false;
                const auto pass = dynamic_pass_runner_->step(
                    snapshot, stable.joint_targets);
                if (pass.valid) {
                    if (pass.finished) walk_reset_pending_ = true;
                    return {
                        true,
                        "DynamicPass-r" + std::to_string(pass.prototype_rollout_id),
                        pass.joint_targets,
                    };
                }
                walk_reset_pending_ = true;
            }
        }
        const auto result = walk_runner_.step(
            snapshot, *walk, reset || walk_reset_pending_, walk->role_id);
        walk_reset_pending_ = false;
        return {
            true,
            result.rapid_turn_active
                ? result.rapid_turn_mirrored
                    ? "RapidTurnV1RightMirror"
                    : "RapidTurnV1Left"
                : result.fast_walk_active ? "FastWalkV2" : "Walk",
            result.joint_targets,
        };
    }

    if (const auto* kick = std::get_if<decision::KickCommand>(&command)) {
        reset_get_up_state();
        if (dynamic_pass_runner_ && dynamic_pass_runner_->active()) {
            const double elapsed =
                dynamic_pass_runner_->elapsed_s(snapshot.server_time);
            decision::WalkCommand pass_base;
            pass_base.target_absolute = false;
            pass_base.target_2d_m = elapsed < 0.65
                ? std::array<double, 2>{0.50, -0.04}
                : std::array<double, 2>{0.0, 0.0};
            pass_base.orientation_deg = 0.0;
            pass_base.orientation_absolute = false;
            pass_base.role_id = decision::RoleManager::ROLE_AP;
            const auto stable = walk_runner_.step(
                snapshot, pass_base, false, pass_base.role_id);
            const auto pass = dynamic_pass_runner_->step(
                snapshot, stable.joint_targets);
            suppress_kick_until_variant_change_ = true;
            if (pass.valid) {
                return {
                    true,
                    "DynamicPass-r" +
                        std::to_string(pass.prototype_rollout_id),
                    pass.joint_targets,
                };
            }
        }
        if (suppress_kick_until_variant_change_) {
            const auto hold = neutral_runner_.step(false, snapshot.server_time);
            return {true, "SuppressedKickAfterDynamicPass", hold.joint_targets};
        }
        reset_dynamic_pass_state();
        return step_kick(snapshot, *kick, reset);
    }

    if (std::holds_alternative<decision::NeutralCommand>(command)) {
        reset_get_up_state();
        reset_kick_state();
        reset_dynamic_pass_state();
        const auto result = neutral_runner_.step(reset, snapshot.server_time);
        return {true, "Neutral", result.joint_targets};
    }

    if (std::holds_alternative<decision::GetUpCommand>(command)) {
        reset_kick_state();
        reset_dynamic_pass_state();
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
            snapshot,
            command,
            parameterized_kick_enabled_,
            learned_kick_enabled_,
            learned_kick_shadow_);
        kick_residual_active_ = parameterized_kick_enabled_ &&
            kick_profile_.static_executor_eligible &&
            kick_residual_runner_.has_value() &&
            kick_residual_runner_->begin(snapshot, kick_profile_);
        procedural_kick_active_ = !kick_residual_active_ &&
            parameterized_kick_enabled_ &&
            kick_profile_.static_executor_eligible &&
            procedural_kick_runner_.has_value() &&
            procedural_kick_runner_->begin(snapshot, kick_profile_);
        learned_kick_active_ = kick_profile_.learned_transition_eligible &&
            learned_kick_runner_.has_value() &&
            learned_kick_runner_->begin(snapshot, kick_profile_);
    }

    const bool specialized_active =
        kick_residual_active_ || procedural_kick_active_ ||
        (learned_kick_enabled_ && learned_kick_active_);
    const bool target_aware = command.mode != decision::KickMode::ForwardContact;
    const bool use_forward_fallback =
        command.allow_forward_contact_fallback && !specialized_active;
    if (target_aware && !specialized_active && !use_forward_fallback) {
        const auto hold = neutral_runner_.step(reset, snapshot.server_time);
        return {true, "RejectedTargetedKickHold", hold.joint_targets};
    }

    const double elapsed = std::max(0.0, snapshot.server_time - kick_start_time_);
    if (procedural_kick_active_ && procedural_kick_runner_.has_value()) {
        const auto result = procedural_kick_runner_->step(elapsed);
        return {
            true,
            result.finished ? "ProceduralKickHold" : "ProceduralKickExecute",
            result.joint_targets,
        };
    }

    decision::WalkCommand walk_command;
    walk_command.target_absolute = false;
    walk_command.target_2d_m = elapsed < 0.65
        ? std::array<double, 2>{0.50, -0.04}
        : std::array<double, 2>{0.0, 0.0};
    walk_command.orientation_deg = 0.0;
    walk_command.orientation_absolute = false;
    walk_command.role_id = decision::RoleManager::ROLE_AP;
    auto walk = walk_runner_.step(
        snapshot, walk_command, reset, walk_command.role_id);

    if (learned_kick_active_ && learned_kick_runner_.has_value()) {
        const auto learned = learned_kick_runner_->step(
            snapshot, kick_profile_, elapsed, walk.joint_targets);
        if (learned_kick_enabled_ && learned.valid) {
            return {
                true,
                learned.finished ? "LearnedKickHold" : "LearnedKickExecute",
                learned.joint_targets,
            };
        }
        if (!learned.valid) learned_kick_active_ = false;
    }
    if (kick_residual_active_ && kick_residual_runner_.has_value()) {
        kick_residual_runner_->apply(elapsed, walk.joint_targets);
    }
    const bool complete = elapsed >= 1.20;
    return {
        true,
        complete
            ? (use_forward_fallback ? "FallbackKickHold" : "ParameterizedKickHold")
            : (use_forward_fallback ? "FallbackKickForward" : "ParameterizedKickExecute"),
        walk.joint_targets,
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

void MotionManager::reset_kick_state() {
    kick_start_time_ = 0.0;
    kick_residual_active_ = false;
    procedural_kick_active_ = false;
    learned_kick_active_ = false;
    suppress_kick_until_variant_change_ = false;
}

void MotionManager::reset_dynamic_pass_state() {
    if (dynamic_pass_runner_) dynamic_pass_runner_->reset();
    walk_reset_pending_ = true;
}

bool MotionManager::dynamic_pass_release_candidate() const {
    return dynamic_pass_runner_ && dynamic_pass_runner_->release_candidate();
}

float MotionManager::dynamic_pass_max_probability() const {
    return dynamic_pass_runner_ ? dynamic_pass_runner_->max_probability() : -1.0F;
}

int MotionManager::dynamic_pass_max_confirmation_streak() const {
    return dynamic_pass_runner_ ? dynamic_pass_runner_->max_confirmation_streak() : 0;
}

int MotionManager::dynamic_pass_best_prototype_rollout_id() const {
    return dynamic_pass_runner_
        ? dynamic_pass_runner_->best_prototype_rollout_id()
        : -1;
}

}  // namespace behavior
