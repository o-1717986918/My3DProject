// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/behavior/kick_execution_profile.h"
#include "src/behavior/onnx_session.h"
#include "src/robot/joint_targets.h"
#include "src/robot/t1_robot_model.h"
#include "src/world/world_snapshot.h"

#include <filesystem>
#include <vector>

namespace behavior {

struct LearnedKickStepResult {
    robot::JointTargets joint_targets;
    bool valid{false};
    bool finished{false};
    float maximum_absolute_action{0.0F};
};

/// Executes the deployable kick_policy_v3 contract ([1,98] -> [1,23]).
///
/// The actor is a residual over Apollo's stable walk targets. The caller may
/// run it in shadow mode and discard `joint_targets`, or grant it full-body
/// ownership after explicitly opting in. Any invalid observation or inference
/// result fails closed so MotionManager can use its already-started fallback.
class LearnedKickRunner {
public:
    explicit LearnedKickRunner(const std::filesystem::path& model_path);

    bool begin(
        const world::WorldSnapshot& snapshot,
        const KickExecutionProfile& profile);

    LearnedKickStepResult step(
        const world::WorldSnapshot& snapshot,
        const KickExecutionProfile& profile,
        double elapsed_s,
        const robot::JointTargets& stable_walk_targets);

    bool active() const { return active_; }
    bool failed() const { return failed_; }
    const std::vector<float>& last_observation() const {
        return last_observation_;
    }

    /// Exposed for deterministic contract tests; no ONNX session is needed.
    static std::vector<float> build_observation(
        const world::WorldSnapshot& snapshot,
        const KickExecutionProfile& profile,
        double elapsed_s,
        const std::vector<float>& previous_action,
        const robot::T1RobotModel& robot_model);

    /// Applies the bounded policy residual to stable walk targets.
    static robot::JointTargets compose_targets(
        const robot::JointTargets& stable_walk_targets,
        const std::vector<float>& action,
        const robot::T1RobotModel& robot_model);

private:
    static constexpr double kDurationS = 1.20;

    OnnxSession session_;
    robot::T1RobotModel robot_model_;
    std::vector<float> previous_action_;
    std::vector<float> last_observation_;
    bool active_{false};
    bool failed_{false};
};

}  // namespace behavior
