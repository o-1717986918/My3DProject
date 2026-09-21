// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/app/runtime_config.h"
#include "src/behavior/dynamic_pass_runner.h"
#include "src/decision/high_level_command.h"
#include "src/behavior/getup_runner.h"
#include "src/behavior/keyframe_runner.h"
#include "src/behavior/kick_execution_profile.h"
#include "src/behavior/kick_residual_runner.h"
#include "src/behavior/learned_kick_runner.h"
#include "src/behavior/procedural_kick_runner.h"
#include "src/behavior/walk_runner.h"
#include "src/robot/joint_targets.h"
#include "src/world/world_snapshot.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace behavior {

/// Low-level targets selected for the current high-level command.
struct MotionStepResult {
    MotionStepResult() = default;
    MotionStepResult(
        bool handled_value,
        std::string active_motion_value,
        robot::JointTargets joint_targets_value,
        robot::JointTargets reference_joint_targets_value = {},
        double motion_elapsed_value = -1.0)
        : handled(handled_value),
          active_motion(std::move(active_motion_value)),
          joint_targets(std::move(joint_targets_value)),
          reference_joint_targets(std::move(reference_joint_targets_value)),
          motion_elapsed_s(motion_elapsed_value) {}

    bool handled{false};
    std::string active_motion;
    robot::JointTargets joint_targets;
    // Optional unmodified base targets used before a learned/procedural
    // overlay. Training telemetry uses this to reproduce the live composition.
    robot::JointTargets reference_joint_targets;
    double motion_elapsed_s{-1.0};
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

    bool dynamic_pass_release_candidate() const;
    float dynamic_pass_max_probability() const;
    int dynamic_pass_max_confirmation_streak() const;
    int dynamic_pass_best_prototype_rollout_id() const;

private:
    enum class GetUpPhase : std::uint8_t {
        Idle,
        Active,
    };

    // Maximum getup duration (seconds) before forcing recovery.
    static constexpr double kGetUpTimeoutS = 6.0;

    WalkRunner walk_runner_;
    std::optional<KickResidualRunner> kick_residual_runner_;
    std::optional<ProceduralKickRunner> procedural_kick_runner_;
    std::optional<LearnedKickRunner> learned_kick_runner_;
    KeyframeRunner neutral_runner_;
    GetupRunner getup_runner_;
    std::unique_ptr<DynamicPassRunner> dynamic_pass_runner_;
    GetUpPhase get_up_phase_{GetUpPhase::Idle};
    bool walk_reset_pending_{false};
    bool get_up_phase_reset_pending_{false};
    double get_up_start_time_{0.0};
    double kick_start_time_{0.0};
    bool parameterized_kick_enabled_{false};
    bool learned_kick_enabled_{false};
    bool learned_kick_shadow_{false};
    KickExecutionProfile kick_profile_;
    bool kick_residual_active_{false};
    bool procedural_kick_active_{false};
    bool learned_kick_active_{false};
    bool suppress_kick_until_variant_change_{false};

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
    void reset_kick_state();
    void reset_dynamic_pass_state();
};

}  // namespace behavior
