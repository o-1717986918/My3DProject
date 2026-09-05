// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/decision/high_level_command.h"
#include "src/behavior/onnx_session.h"
#include "src/robot/joint_targets.h"
#include "src/robot/t1_robot_model.h"
#include "src/world/world_snapshot.h"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace behavior {

/// Policy inputs, outputs, and decoded targets from one walking step.
struct WalkStepResult {
    std::vector<float> observation;
    std::vector<float> action;
    robot::JointTargets joint_targets;
    bool fast_walk_active{false};
    bool rapid_turn_active{false};
    bool rapid_turn_mirrored{false};
};

/// Executes the learned walking policy and its observation history.
class WalkRunner {
public:
    struct HeadTrackerState {
        std::optional<std::array<double, 2>> last_head_target_deg;
        std::optional<double> last_ball_seen_time;
    };

    explicit WalkRunner(
        const std::filesystem::path& model_path,
        std::optional<std::filesystem::path> fast_walk_model_path = std::nullopt,
        std::optional<std::filesystem::path> rapid_turn_model_path = std::nullopt);

    /// Evaluates one policy step; `reset` reinitializes temporal observations.
    WalkStepResult step(
        const world::WorldSnapshot& snapshot,
        const decision::WalkCommand& command,
        bool reset,
        std::optional<int> role_id = std::nullopt);

private:
    static constexpr float kActionScale = 0.25F;
    // The policy input is an angular-velocity request in rad/s. A 0.2 gain
    // left a persistent ~15 degree dead-zone during close ball alignment;
    // unit gain preserves the physical meaning and reaches the 2 degree kick
    // release gate without exceeding the existing +/-0.5 rad/s clamp.
    static constexpr float kOrientationToAngVelScale = 1.0F;

    OnnxSession session_;
    std::optional<OnnxSession> fast_walk_session_;
    std::optional<OnnxSession> rapid_turn_session_;
    robot::T1RobotModel robot_model_;
    std::vector<float> previous_action_;
    std::vector<float> observation_;
    std::vector<float> step_obs_buffer_;
    int history_length_{1};
    int step_obs_dim_{0};
    HeadTrackerState head_tracker_state_;
    std::vector<float> fast_previous_action_;
    double fast_gait_phase_{0.0};
    bool fast_walk_disabled_{false};
    bool fast_walk_active_{false};
    double fast_walk_cooldown_until_s_{0.0};
    mutable int last_fast_walk_gate_{-1};
    std::vector<float> rapid_turn_previous_action_;
    double rapid_turn_gait_phase_{0.0};
    bool rapid_turn_disabled_{false};
    bool rapid_turn_active_{false};
    double rapid_turn_cooldown_until_s_{0.0};

    std::array<float, 3> compute_velocity_command(
        const world::WorldSnapshot& snapshot,
        const decision::WalkCommand& command) const;
    void build_observation(
        const world::WorldSnapshot& snapshot,
        const std::array<float, 3>& velocity_command);
    robot::JointTargets decode_action(
        const world::WorldSnapshot& snapshot,
        const std::vector<float>& action,
        std::optional<int> role_id);
    bool fast_walk_supported(
        const world::WorldSnapshot& snapshot,
        const decision::WalkCommand& command,
        const std::array<float, 3>& stable_velocity_command);
    std::optional<robot::JointTargets> step_fast_walk(
        const world::WorldSnapshot& snapshot,
        const decision::WalkCommand& command,
        const std::array<float, 3>& stable_velocity_command,
        const robot::JointTargets& stable_targets,
        bool reset);
    std::vector<float> build_run_policy_observation(
        const world::WorldSnapshot& snapshot,
        const std::array<float, 3>& velocity_command,
        const std::vector<float>& previous_action,
        double gait_phase) const;
    bool rapid_turn_supported(
        const world::WorldSnapshot& snapshot,
        const std::array<float, 3>& stable_velocity_command);
    std::optional<robot::JointTargets> step_rapid_turn(
        const world::WorldSnapshot& snapshot,
        const std::array<float, 3>& stable_velocity_command,
        const robot::JointTargets& stable_targets,
        bool reset);
};

}  // namespace behavior
