// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/kick_residual_runner.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

behavior::KickExecutionProfile make_profile() {
    behavior::KickExecutionProfile profile;
    profile.kind = behavior::KickProfileKind::ParameterizedContact;
    profile.target_distance_m = 2.0;
    profile.relative_target_angle_deg = 0.0;
    profile.requested_speed_mps = 1.43;
    profile.mode = decision::KickMode::TargetedPass;
    return profile;
}

world::WorldSnapshot make_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.self.position_m = {0.0, 0.0, 0.65};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {0.32, 0.0, 0.11};
    return snapshot;
}

robot::JointTargets zero_targets() {
    robot::T1RobotModel model;
    robot::JointTargets targets;
    for (const auto& name : model.readable_joint_names()) {
        targets.push_back({name, 0.0, 0.0, 0.0, 0.0, 0.0});
    }
    return targets;
}

double maximum_position(const robot::JointTargets& targets) {
    double result = 0.0;
    for (const auto& target : targets) {
        result = std::max(result, std::abs(target.q_deg));
    }
    return result;
}

double position_for(
    const robot::JointTargets& targets,
    const std::string& joint_name) {
    for (const auto& target : targets) {
        if (target.joint_name == joint_name) {
            return target.q_deg;
        }
    }
    return 0.0;
}

}  // namespace

int main() {
    const auto table = std::filesystem::path(
        APOLLO_CODE_BASE_PROJECT_SOURCE_DIR) /
        "assets/keyframes/kick_residual_table.yaml";
    behavior::KickResidualRunner runner(table);
    if (runner.node_count() < 150U) {
        std::cerr << "unexpected kick residual table size\n";
        return 1;
    }

    auto snapshot = make_snapshot();
    auto profile = make_profile();
    if (!runner.begin(snapshot, profile) ||
        runner.selected_condition_index() < 0) {
        std::cerr << "supported request did not select a table node\n";
        return 1;
    }

    auto start = zero_targets();
    runner.apply(0.0, start);
    if (maximum_position(start) > 1.0e-9) {
        std::cerr << "residual was non-zero at action start\n";
        return 1;
    }
    auto strike = zero_targets();
    runner.apply(0.54, strike);
    if (maximum_position(strike) < 1.0) {
        std::cerr << "strike phase did not alter joint targets\n";
        return 1;
    }
    // The selected x=-0.01/y=0 node requests 0.954 rad at the right knee;
    // the shared decoder clips it to the declared 0.45 rad residual bound.
    const double expected_knee_deg = 0.45 * 180.0 / 3.14159265358979323846;
    if (std::abs(
            position_for(strike, "Right_Knee_Pitch") - expected_knee_deg) >
        1.0e-9) {
        std::cerr << "C++ residual decoder diverged from the training contract\n";
        return 1;
    }
    auto complete = zero_targets();
    runner.apply(1.20, complete);
    if (maximum_position(complete) > 1.0e-9) {
        std::cerr << "residual did not return to zero\n";
        return 1;
    }

    profile.target_distance_m = 2.76;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "out-of-envelope distance selected a node\n";
        return 1;
    }
    profile = make_profile();
    profile.target_distance_m = 2.75;
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "residual distance boundary was rejected\n";
        return 1;
    }
    profile = make_profile();
    snapshot.ball.visible = false;
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "fresh occluded ball did not select a node\n";
        return 1;
    }
    snapshot.ball.position_age_s = 1.50;
    snapshot.ball.near_contact_track = true;
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "bounded near-field ball track was rejected\n";
        return 1;
    }
    snapshot.ball.position_age_s =
        world::kNearContactBallTrackLifetimeS + 0.01;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "expired ball position selected a node\n";
        return 1;
    }
    return 0;
}
