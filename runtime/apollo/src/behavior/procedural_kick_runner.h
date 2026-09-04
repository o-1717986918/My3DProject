// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/behavior/kick_execution_profile.h"
#include "src/robot/joint_targets.h"
#include "src/robot/t1_robot_model.h"
#include "src/world/world_snapshot.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace behavior {

struct ProceduralKickStepResult {
    robot::JointTargets joint_targets;
    bool finished{false};
};

/// Standalone deterministic kick trajectory. This runner owns all 23 joint
/// targets and never creates or invokes an ONNX session.
class ProceduralKickRunner {
public:
    explicit ProceduralKickRunner(const std::filesystem::path& asset_path);

    /// Selects a physically gated anchor and captures the measured start pose.
    bool begin(
        const world::WorldSnapshot& snapshot,
        const KickExecutionProfile& profile);

    ProceduralKickStepResult step(double elapsed_s) const;

    std::size_t anchor_count() const { return anchors_.size(); }
    bool active() const { return selected_ != nullptr; }
    std::string active_anchor_name() const;
    double duration_s() const;

private:
    static constexpr std::size_t kJointCount = 23U;
    static constexpr std::size_t kParameterCount = 14U;
    static constexpr std::size_t kKeyCount = 6U;

    struct Anchor {
        std::string name;
        std::string mode;
        double target_distance_m{0.0};
        double target_distance_tolerance_m{0.0};
        double target_angle_deg{0.0};
        double target_angle_tolerance_deg{0.0};
        double requested_speed_mps{0.0};
        double requested_speed_tolerance_mps{0.0};
        double ball_local_x_m{0.0};
        double ball_local_y_m{0.0};
        double ball_x_tolerance_m{0.0};
        double ball_y_tolerance_m{0.0};
        double kp{150.0};
        double kd{1.0};
        std::array<double, kKeyCount> key_times_s{};
        std::array<double, kParameterCount> parameters{};
    };

    robot::T1RobotModel robot_model_;
    std::vector<Anchor> anchors_;
    const Anchor* selected_{nullptr};
    std::array<double, kJointCount> captured_pose_rad_{};

    std::array<double, kJointCount> joint_delta_at(double elapsed_s) const;
};

}  // namespace behavior
