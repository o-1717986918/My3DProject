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

/// Experimental deterministic residual kick selected from an exact-physics
/// parameter table. Unsupported requests fail closed to the walk-only macro.
class KickResidualRunner {
public:
    explicit KickResidualRunner(const std::filesystem::path& table_path);

    /// Selects one validated table node for a newly started kick.
    bool begin(
        const world::WorldSnapshot& snapshot,
        const KickExecutionProfile& profile);

    /// Adds the selected bounded residual to Apollo walk joint targets.
    bool apply(double elapsed_s, robot::JointTargets& joint_targets) const;

    std::size_t node_count() const;
    int selected_condition_index() const;

private:
    static constexpr std::size_t kJointCount = 23U;
    static constexpr std::size_t kParameterCount = 14U;

    struct Node {
        int condition_index{-1};
        double distance_m{0.0};
        double angle_deg{0.0};
        double requested_speed_mps{0.0};
        double ball_x_offset_m{0.0};
        double ball_y_offset_m{0.0};
        std::string mode;
        std::array<double, kParameterCount> parameters{};
    };

    robot::T1RobotModel robot_model_;
    std::vector<Node> nodes_;
    const Node* selected_{nullptr};

    std::array<double, kJointCount> residual_at(double elapsed_s) const;
};

}  // namespace behavior
