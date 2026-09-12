// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/behavior/onnx_session.h"
#include "src/robot/joint_targets.h"
#include "src/robot/t1_robot_model.h"
#include "src/world/world_snapshot.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace behavior {

struct DynamicPassActivation {
    bool started{false};
    int prototype_rollout_id{-1};
    float confidence{0.0F};
};

struct DynamicPassStepResult {
    bool valid{false};
    bool finished{false};
    int prototype_rollout_id{-1};
    float confidence{0.0F};
    robot::JointTargets joint_targets;
};

/// Experimental, opt-in 2 m straight-pass primitive.
///
/// The selector observes live walk-to-ball states. Once one prototype remains
/// above its frozen threshold for two consecutive frames, the selected
/// fourteen-parameter trajectory is overlaid on the live Apollo Walk target.
/// This deliberately does not import the old KickCommand or strategy stack.
class DynamicPassRunner {
public:
    explicit DynamicPassRunner(const std::filesystem::path& selector_path);

    DynamicPassActivation consider(
        const world::WorldSnapshot& snapshot,
        bool eligible);
    DynamicPassStepResult step(
        const world::WorldSnapshot& snapshot,
        const robot::JointTargets& stable_walk_targets);
    void reset();

    bool active() const { return active_; }
    bool release_candidate() const { return release_candidate_; }
    float max_probability() const { return max_probability_; }
    int max_confirmation_streak() const;
    int best_prototype_rollout_id() const { return best_prototype_rollout_id_; }
    double elapsed_s(double server_time) const;

    static bool release_geometry(const world::WorldSnapshot& snapshot);
    static std::vector<float> build_selector_observation(
        const world::WorldSnapshot& snapshot,
        const robot::T1RobotModel& robot_model);
    static std::array<double, 23> prototype_delta_rad(
        std::size_t prototype_index,
        double elapsed_s);
    static robot::JointTargets compose_targets(
        const robot::JointTargets& stable_walk_targets,
        const std::array<double, 23>& delta_rad,
        const robot::T1RobotModel& robot_model);

private:
    static constexpr std::size_t kPrototypeCount = 10U;
    static constexpr double kDurationS = 1.20;

    OnnxSession selector_;
    robot::T1RobotModel robot_model_;
    std::array<int, kPrototypeCount> streak_{};
    bool active_{false};
    bool armed_{true};
    std::size_t selected_prototype_{0U};
    float selected_confidence_{0.0F};
    double start_time_s_{0.0};
    bool release_candidate_{false};
    float max_probability_{-1.0F};
    int best_prototype_rollout_id_{-1};
};

}  // namespace behavior
