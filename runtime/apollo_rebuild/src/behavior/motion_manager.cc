// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/behavior/motion_manager.h"

#include "src/decision/role_manager.h"

#include <memory>
#include <string>
#include <variant>

namespace behavior {

MotionManager::MotionManager(const app::RuntimeConfig& config)
    : walk_runner_(config.resolve_asset_path("networks/walk/policy.onnx")),
      neutral_runner_(config.resolve_asset_path("keyframes/neutral.yaml")),
      getup_runner_(config.resolve_asset_path("networks/getup/policy.onnx")) {
    if (config.enable_dynamic_pass) {
        dynamic_pass_runner_ = std::make_unique<DynamicPassRunner>(
            config.resolve_asset_path("networks/dynamic_pass/selector.onnx"),
            config.dynamic_pass_forced_rollout_id);
    }
}

MotionStepResult MotionManager::step(
    const world::WorldSnapshot& snapshot,
    const decision::HighLevelCommand& command,
    bool reset) {
    if (std::holds_alternative<decision::BeamCommand>(command)) {
        reset_get_up_state();
        reset_dynamic_pass_state();
        return {false, "BeamBypass", {}};
    }

    if (get_up_phase_ != GetUpPhase::Idle) {
        reset_dynamic_pass_state();
        return step_get_up(snapshot, false);
    }

    if (const auto* walk = std::get_if<decision::WalkCommand>(&command)) {
        reset_get_up_state();
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
        return {true, "Walk", result.joint_targets};
    }

    if (std::holds_alternative<decision::NeutralCommand>(command)) {
        reset_get_up_state();
        reset_dynamic_pass_state();
        const auto result = neutral_runner_.step(reset, snapshot.server_time);
        return {true, "Neutral", result.joint_targets};
    }

    if (std::holds_alternative<decision::GetUpCommand>(command)) {
        reset_dynamic_pass_state();
        return step_get_up(snapshot, reset);
    }

    return {false, "Idle", {}};
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
