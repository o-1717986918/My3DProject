// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/dynamic_pass_runner.h"
#include "src/behavior/policy_common.h"
#include "src/math/math_utils.h"

#include <cassert>
#include <cmath>

namespace {

bool near(double actual, double expected, double tolerance = 1.0e-5) {
    return std::abs(actual - expected) <= tolerance;
}

world::WorldSnapshot make_snapshot(const robot::T1RobotModel& robot_model) {
    world::WorldSnapshot snapshot;
    snapshot.self.position_m = {0.0, 0.0, 0.60};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.self.gyro_deg_s = {
        math::rad_to_deg(1.0), math::rad_to_deg(-2.0), math::rad_to_deg(0.5)};
    snapshot.self.lin_vel_b = {0.20, -0.10, 0.05};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {0.34375, 0.01, 0.11};
    snapshot.ball.velocity_mps = {0.30, 0.20, 0.02};
    snapshot.ball.velocity_valid = true;

    const auto& names = robot_model.readable_joint_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        double offset = 0.0;
        if (i == 17U) offset = 0.10;
        snapshot.self.joint_positions_deg[names[i]] =
            math::rad_to_deg(behavior::kDefaultPosRad[i] + offset);
        snapshot.self.joint_velocities_deg_s[names[i]] = 0.0;
    }
    return snapshot;
}

}  // namespace

int main() {
    robot::T1RobotModel robot_model;
    auto snapshot = make_snapshot(robot_model);
    assert(behavior::DynamicPassRunner::release_geometry(snapshot));
    snapshot.ball.position_m[1] = 0.20;
    assert(!behavior::DynamicPassRunner::release_geometry(snapshot));
    snapshot.ball.position_m[1] = 0.01;
    snapshot.ball.visible = false;
    snapshot.ball.position_age_s = 0.04;
    assert(behavior::DynamicPassRunner::release_geometry(snapshot));
    snapshot.ball.position_age_s = 0.11;
    assert(!behavior::DynamicPassRunner::release_geometry(snapshot));
    snapshot.ball.visible = true;
    snapshot.ball.position_age_s = 0.0;

    const auto observation =
        behavior::DynamicPassRunner::build_selector_observation(snapshot, robot_model);
    assert(observation.size() == 98U);
    assert(near(observation[0], 1.0));
    assert(near(observation[1], -2.0));
    assert(near(observation[2], 0.5));
    assert(near(observation[5], -1.0));
    assert(near(observation[6 + 17], 0.10));
    assert(near(observation[75], 0.34375));
    assert(near(observation[76], 0.01));
    assert(near(observation[77], -0.49));
    assert(near(observation[78], 0.10));
    assert(near(observation[79], 0.30));
    assert(near(observation[80], -0.03));
    assert(near(observation[81], 1.0));
    assert(near(observation[82], 0.0));
    assert(near(observation[83], 2.0));
    assert(near(observation[84], 1.43));
    assert(near(observation[85], 0.8));
    assert(near(observation[89], 0.0));
    assert(near(observation[90], 1.0));
    assert(near(observation[91], 0.0));
    assert(near(observation[92], 1.0));
    assert(near(observation[93], 1.0));
    assert(near(observation[94], 0.0));
    assert(near(observation[95], 1.0));
    assert(near(observation[96], 0.0));
    assert(near(observation[97], 0.0));

    const auto start = behavior::DynamicPassRunner::prototype_delta_rad(8U, 0.0);
    const auto contact = behavior::DynamicPassRunner::prototype_delta_rad(8U, 0.54);
    const auto finish = behavior::DynamicPassRunner::prototype_delta_rad(8U, 1.20);
    for (const double value : start) assert(near(value, 0.0));
    for (const double value : finish) assert(near(value, 0.0));
    assert(near(contact[17], -0.35));
    assert(near(contact[20], 0.45));
    assert(near(contact[21], 0.25));
    assert(near(contact[18], 0.20));

    robot::JointTargets base;
    const auto& names = robot_model.readable_joint_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        base.push_back({
            names[i], math::rad_to_deg(behavior::kDefaultPosRad[i]), 0.0,
            robot_model.joint_kp(names[i]), robot_model.joint_kd(names[i]), 0.0});
    }
    const auto composed = behavior::DynamicPassRunner::compose_targets(
        base, contact, robot_model);
    assert(near(math::deg_to_rad(composed[17].q_deg), -0.55));
    assert(near(math::deg_to_rad(composed[20].q_deg), 0.85));
    return 0;
}
