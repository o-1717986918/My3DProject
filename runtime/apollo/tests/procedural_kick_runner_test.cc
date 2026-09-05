// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/procedural_kick_runner.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

world::WorldSnapshot make_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.self.position_m = {0.0, 0.0, 0.65};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {0.32, 0.04, 0.11};
    robot::T1RobotModel robot;
    for (const auto& name : robot.readable_joint_names()) {
        snapshot.self.joint_positions_deg[name] = 0.0;
        snapshot.self.joint_velocities_deg_s[name] = 0.0;
    }
    return snapshot;
}

behavior::KickExecutionProfile make_profile() {
    behavior::KickExecutionProfile profile;
    profile.kind = behavior::KickProfileKind::ProceduralContact;
    profile.target_distance_m = 0.55;
    profile.relative_target_angle_deg = 0.0;
    profile.requested_speed_mps = 0.90;
    profile.mode = decision::KickMode::DribbleTouch;
    return profile;
}

double maximum_position(const robot::JointTargets& targets) {
    double result = 0.0;
    for (const auto& target : targets) {
        result = std::max(result, std::abs(target.q_deg));
    }
    return result;
}

}  // namespace

int main() {
    const auto asset = std::filesystem::path(
        APOLLO_CODE_BASE_PROJECT_SOURCE_DIR) /
        "assets/keyframes/procedural_kick.yaml";
    behavior::ProceduralKickRunner runner(asset);
    if (runner.anchor_count() != 5U) {
        std::cerr << "unexpected procedural anchor count\n";
        return 1;
    }

    auto snapshot = make_snapshot();
    auto profile = make_profile();
    if (!runner.begin(snapshot, profile) ||
        runner.active_anchor_name() != "right_dribble_055m_v1" ||
        std::abs(runner.duration_s() - 1.20) > 1.0e-9) {
        std::cerr << "validated short-touch anchor was not selected\n";
        return 1;
    }

    profile.relative_target_angle_deg = 5.99;
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "relaxed short-touch angle was not accepted\n";
        return 1;
    }
    profile.relative_target_angle_deg = 6.01;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "out-of-envelope short-touch angle was accepted\n";
        return 1;
    }
    profile = make_profile();
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "nominal short-touch could not restart\n";
        return 1;
    }

    snapshot.ball.position_m[1] = 0.0649;
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "validated dispatch-margin short-touch was rejected\n";
        return 1;
    }
    snapshot.ball.position_m[1] = 0.0651;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "short-touch outside the dispatch margin was accepted\n";
        return 1;
    }
    snapshot = make_snapshot();
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "nominal short-touch could not restart after margin test\n";
        return 1;
    }

    const auto start = runner.step(0.0);
    if (start.finished || start.joint_targets.size() != 23U ||
        maximum_position(start.joint_targets) > 1.0e-9) {
        std::cerr << "procedural kick did not capture a neutral start\n";
        return 1;
    }
    for (const auto& target : start.joint_targets) {
        if (target.kp != 150.0 || target.kd != 1.0) {
            std::cerr << "procedural kick lost its independent PD contract\n";
            return 1;
        }
    }

    const auto strike = runner.step(0.54);
    if (strike.finished || maximum_position(strike.joint_targets) < 5.0) {
        std::cerr << "procedural strike phase produced no joint motion\n";
        return 1;
    }
    const auto complete = runner.step(1.20);
    if (!complete.finished || maximum_position(complete.joint_targets) > 1.0e-9) {
        std::cerr << "procedural kick did not return to neutral\n";
        return 1;
    }

    profile.target_distance_m = 4.0;
    profile.requested_speed_mps = 2.50;
    profile.mode = decision::KickMode::Shot;
    if (!runner.begin(snapshot, profile) ||
        runner.active_anchor_name() != "right_shot_4m_v1") {
        std::cerr << "validated 4 m shot anchor was not selected\n";
        return 1;
    }

    profile.target_distance_m = 6.0;
    profile.requested_speed_mps = 3.50;
    profile.mode = decision::KickMode::Clear;
    if (!runner.begin(snapshot, profile) ||
        runner.active_anchor_name() != "right_clear_6m_v1") {
        std::cerr << "validated 6 m clear anchor was not selected\n";
        return 1;
    }

    snapshot = make_snapshot();
    snapshot.ball.position_m = {0.31, -0.04, 0.11};
    profile.kind = behavior::KickProfileKind::ParameterizedContact;
    profile.target_distance_m = 3.5;
    profile.requested_speed_mps = 2.20;
    profile.relative_target_angle_deg = 0.0;
    profile.mode = decision::KickMode::TargetedPass;
    if (!runner.begin(snapshot, profile) ||
        runner.active_anchor_name() != "right_pass_3p5m_experimental_v1") {
        std::cerr << "experimental 3.5 m pass anchor was not selected\n";
        return 1;
    }
    profile.target_distance_m = 5.0;
    profile.requested_speed_mps = 3.00;
    if (!runner.begin(snapshot, profile) ||
        runner.active_anchor_name() != "right_pass_5m_experimental_v1") {
        std::cerr << "experimental 5 m pass anchor was not selected\n";
        return 1;
    }

    snapshot.ball.position_m[1] = 0.0;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "unvalidated ball slot selected a procedural anchor\n";
        return 1;
    }
    snapshot = make_snapshot();
    profile.mode = decision::KickMode::TargetedPass;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "short-touch anchor was misreported as a pass\n";
        return 1;
    }
    profile = make_profile();
    snapshot.self.lin_vel_b[0] = 0.49;
    if (!runner.begin(snapshot, profile)) {
        std::cerr << "bounded moving release was not accepted\n";
        return 1;
    }
    snapshot.self.lin_vel_b[0] = 0.51;
    if (runner.begin(snapshot, profile)) {
        std::cerr << "excessive start speed bypassed the procedural release guard\n";
        return 1;
    }
    snapshot = make_snapshot();
    snapshot.self.joint_positions_deg.erase("Right_Knee_Pitch");
    if (runner.begin(snapshot, profile)) {
        std::cerr << "incomplete joint state bypassed capture validation\n";
        return 1;
    }
    return 0;
}
