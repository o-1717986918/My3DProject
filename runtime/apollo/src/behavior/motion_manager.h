// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/app/runtime_config.h"
#include "src/decision/high_level_command.h"
#include "src/behavior/getup_runner.h"
#include "src/behavior/kick_execution_profile.h"
#include "src/behavior/kick_residual_runner.h"
#include "src/behavior/keyframe_runner.h"
#include "src/behavior/walk_runner.h"
#include "src/robot/joint_targets.h"
#include "src/world/world_snapshot.h"

#include <cstdint>
#include <optional>
#include <string>

namespace behavior {

/// Low-level targets selected for the current high-level command.
struct MotionStepResult {
    bool handled{false};
    std::string active_motion;
    robot::JointTargets joint_targets;
};

/// Selects and coordinates walk, neutral, and get-up motion runners.
class MotionManager {
public:
    explicit MotionManager(const app::RuntimeConfig& config);

    /// Executes one cycle; `reset` signals a newly selected command variant.
    MotionStepResult step(
        const world::WorldSnapshot& snapshot,
        const decision::HighLevelCommand& command,
        bool reset);

    int active_kick_condition_index() const {
        return kick_residual_active_ && kick_residual_runner_.has_value()
            ? kick_residual_runner_->selected_condition_index()
            : -1;
    }

private:
    enum class GetUpPhase : std::uint8_t {
        Idle,
        Active,
    };

    // Maximum getup duration (seconds) before forcing recovery.
    static constexpr double kGetUpTimeoutS = 6.0;
    WalkRunner walk_runner_;
    std::optional<KickResidualRunner> kick_residual_runner_;
    KeyframeRunner neutral_runner_;
    GetupRunner getup_runner_;
    GetUpPhase get_up_phase_{GetUpPhase::Idle};
    bool get_up_phase_reset_pending_{false};
    double get_up_start_time_{0.0};
    double kick_start_time_{0.0};
    bool parameterized_kick_enabled_{false};
    KickExecutionProfile kick_profile_;
    bool kick_residual_active_{false};

    MotionStepResult step_get_up(
        const world::WorldSnapshot& snapshot,
        bool reset);
    MotionStepResult step_kick(
        const world::WorldSnapshot& snapshot,
        const decision::KickCommand& command,
        bool reset);
    void enter_get_up_phase(
        GetUpPhase phase,
        double server_time);
    void reset_get_up_state();
};

}  // namespace behavior
