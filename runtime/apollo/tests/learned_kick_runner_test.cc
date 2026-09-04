// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/learned_kick_runner.h"
#include "src/behavior/policy_common.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

world::WorldSnapshot make_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.self.position_m = {1.0, 2.0, 0.65};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.self.lin_vel_b = {0.10, -0.02, 0.0};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.04;
    snapshot.ball.position_m = {1.30, 2.04, 0.11};
    snapshot.ball.velocity_valid = true;
    snapshot.ball.velocity_mps = {0.20, 0.03, 0.0};
    robot::T1RobotModel robot;
    for (std::size_t i = 0; i < robot.readable_joint_names().size(); ++i) {
        const auto& name = robot.readable_joint_names()[i];
        snapshot.self.joint_positions_deg[name] =
            math::rad_to_deg(behavior::kDefaultPosRad[i]);
        snapshot.self.joint_velocities_deg_s[name] = 0.0;
    }
    return snapshot;
}

behavior::KickExecutionProfile make_profile() {
    behavior::KickExecutionProfile profile;
    profile.kind = behavior::KickProfileKind::ParameterizedContact;
    profile.mode = decision::KickMode::TargetedPass;
    profile.target_distance_m = 2.0;
    profile.relative_target_angle_deg = 0.0;
    profile.requested_speed_mps = 1.43;
    return profile;
}

bool near(double left, double right, double tolerance = 1.0e-6) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main(int argc, char* argv[]) {
    const robot::T1RobotModel robot;
    const auto snapshot = make_snapshot();
    const auto profile = make_profile();
    const std::vector<float> previous_action(23U, 0.0F);
    const auto observation = behavior::LearnedKickRunner::build_observation(
        snapshot, profile, 0.0, previous_action, robot);
    if (observation.size() != 98U ||
        !near(observation[75], 0.30) ||
        !near(observation[76], 0.04) ||
        !near(observation[78], 0.10) ||
        !near(observation[79], 0.05) ||
        !near(observation[81], 1.0) ||
        !near(observation[82], 0.0) ||
        !near(observation[83], 2.0) ||
        !near(observation[84], 1.43) ||
        !near(observation[85], 0.8) ||
        !near(observation[86], 1.0) ||
        !near(observation[89], 0.04) ||
        !near(observation[90], 1.0) ||
        !near(observation[91], 0.0) ||
        !near(observation[92], 1.0) ||
        !near(observation[93], 0.0) ||
        !near(observation[94], 1.0) ||
        !near(observation[95], 0.0) ||
        !near(observation[96], 1.0) ||
        !near(observation[97], 0.0)) {
        std::cerr << "kick_policy_v3 observation layout drifted\n";
        return 1;
    }

    robot::JointTargets baseline;
    for (const auto& name : robot.readable_joint_names()) {
        baseline.push_back({name, 0.0, 0.0, 10.0, 0.1, 0.0});
    }
    std::vector<float> action(23U, 0.0F);
    action[14] = 1.0F;
    const auto composed = behavior::LearnedKickRunner::compose_targets(
        baseline, action, robot);
    if (composed.size() != 23U ||
        !near(composed[14].q_deg, math::rad_to_deg(0.45))) {
        std::cerr << "learned kick residual decoder drifted\n";
        return 1;
    }

    action[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_non_finite = false;
    try {
        static_cast<void>(behavior::LearnedKickRunner::compose_targets(
            baseline, action, robot));
    } catch (const std::runtime_error&) {
        rejected_non_finite = true;
    }
    if (!rejected_non_finite) {
        std::cerr << "non-finite learned kick action was accepted\n";
        return 1;
    }

    if (argc == 2) {
        behavior::LearnedKickRunner runner(argv[1]);
        if (!runner.begin(snapshot, profile)) {
            std::cerr << "valid target pass did not start learned kick\n";
            return 1;
        }
        const auto inference = runner.step(
            snapshot, profile, 0.0, baseline);
        if (!inference.valid || inference.finished ||
            inference.joint_targets.size() != 23U || runner.failed()) {
            std::cerr << "external kick_policy_v3 inference failed\n";
            return 1;
        }
        auto out_of_envelope = profile;
        out_of_envelope.target_distance_m = 3.0;
        if (runner.begin(snapshot, out_of_envelope)) {
            std::cerr << "fixed-2m actor accepted an untrained distance\n";
            return 1;
        }
    }
    return 0;
}
