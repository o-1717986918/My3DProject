// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/kick_execution_profile.h"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

world::WorldSnapshot make_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.self.position_m = {1.0, 2.0, 0.8};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    return snapshot;
}

decision::KickCommand make_targeted(double speed, double angle_deg) {
    decision::KickCommand command;
    const double angle_rad = angle_deg * 3.14159265358979323846 / 180.0;
    command.target_point_m = std::array<double, 2>{
        1.0 + 4.0 * std::cos(angle_rad),
        2.0 + 4.0 * std::sin(angle_rad),
    };
    command.requested_ball_speed_mps = speed;
    command.mode = decision::KickMode::TargetedPass;
    return command;
}

bool near(double lhs, double rhs, double tolerance = 1.0e-9) {
    return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    const world::WorldSnapshot snapshot = make_snapshot();

    const auto disabled = behavior::make_kick_execution_profile(
        snapshot, make_targeted(1.43, 0.0), false);
    if (disabled.kind != behavior::KickProfileKind::StableFallback ||
        !near(disabled.local_drive_target_m[0], 0.50) ||
        !near(disabled.drive_duration_s, 0.65) ||
        !near(disabled.total_duration_s, 1.0)) {
        std::cerr << "disabled profile did not preserve the accepted fallback\n";
        return 1;
    }

    const auto slow = behavior::make_kick_execution_profile(
        snapshot, make_targeted(0.8, 0.0), true);
    const auto fast = behavior::make_kick_execution_profile(
        snapshot, make_targeted(3.0, 0.0), true);
    if (slow.kind != behavior::KickProfileKind::ParameterizedContact ||
        fast.kind != behavior::KickProfileKind::ParameterizedContact ||
        !near(slow.local_drive_target_m[0], 0.50) ||
        !near(fast.local_drive_target_m[0], 0.85) ||
        !near(slow.target_distance_m, 4.0) ||
        slow.mode != decision::KickMode::TargetedPass ||
        !(fast.drive_duration_s > slow.drive_duration_s)) {
        std::cerr << "speed request was not mapped monotonically and boundedly\n";
        return 1;
    }

    const auto left = behavior::make_kick_execution_profile(
        snapshot, make_targeted(10.0 / 9.0, 10.0), true);
    const auto right = behavior::make_kick_execution_profile(
        snapshot, make_targeted(10.0 / 9.0, -10.0), true);
    if (!(left.local_drive_target_m[1] > -0.04) ||
        !(right.local_drive_target_m[1] < -0.04) ||
        !near(left.relative_target_angle_deg, 10.0, 1.0e-8) ||
        !near(right.relative_target_angle_deg, -10.0, 1.0e-8)) {
        std::cerr << "target direction did not produce the expected lateral bias\n";
        return 1;
    }

    auto invalid = make_targeted(1.43, 0.0);
    invalid.requested_ball_speed_mps =
        std::numeric_limits<double>::quiet_NaN();
    const auto invalid_profile = behavior::make_kick_execution_profile(
        snapshot, invalid, true);
    const auto outside_angle = behavior::make_kick_execution_profile(
        snapshot, make_targeted(1.43, 16.0), true);
    const auto outside_speed = behavior::make_kick_execution_profile(
        snapshot, make_targeted(3.01, 0.0), true);
    if (invalid_profile.kind != behavior::KickProfileKind::StableFallback ||
        outside_angle.kind != behavior::KickProfileKind::StableFallback ||
        outside_speed.kind != behavior::KickProfileKind::StableFallback) {
        std::cerr << "unsupported requests did not use the stable fallback\n";
        return 1;
    }

    return 0;
}
