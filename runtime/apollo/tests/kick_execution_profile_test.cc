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
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {1.0, 2.0, 0.11};
    return snapshot;
}

decision::KickCommand make_targeted(double speed, double angle_deg) {
    decision::KickCommand command;
    const double angle_rad = angle_deg * 3.14159265358979323846 / 180.0;
    command.target_point_m = std::array<double, 2>{
        1.0 + 2.0 * std::cos(angle_rad),
        2.0 + 2.0 * std::sin(angle_rad),
    };
    command.requested_ball_speed_mps = speed;
    command.mode = decision::KickMode::TargetedPass;
    return command;
}

decision::KickCommand make_dribble(double angle_deg = 0.0) {
    decision::KickCommand command;
    const double angle_rad = angle_deg * 3.14159265358979323846 / 180.0;
    command.target_point_m = std::array<double, 2>{
        1.0 + 0.55 * std::cos(angle_rad),
        2.0 + 0.55 * std::sin(angle_rad),
    };
    command.requested_ball_speed_mps = 0.90;
    command.mode = decision::KickMode::DribbleTouch;
    return command;
}

decision::KickCommand make_shot(double distance_m = 4.0, double angle_deg = 0.0) {
    decision::KickCommand command;
    const double angle_rad = angle_deg * 3.14159265358979323846 / 180.0;
    command.target_point_m = std::array<double, 2>{
        1.0 + distance_m * std::cos(angle_rad),
        2.0 + distance_m * std::sin(angle_rad),
    };
    command.requested_ball_speed_mps = 2.50;
    command.mode = decision::KickMode::Shot;
    return command;
}

decision::KickCommand make_clear(double distance_m = 6.0, double angle_deg = 0.0) {
    decision::KickCommand command;
    const double angle_rad = angle_deg * 3.14159265358979323846 / 180.0;
    command.target_point_m = std::array<double, 2>{
        1.0 + distance_m * std::cos(angle_rad),
        2.0 + distance_m * std::sin(angle_rad),
    };
    command.requested_ball_speed_mps = 3.50;
    command.mode = decision::KickMode::Clear;
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

    const auto pass = behavior::make_kick_execution_profile(
        snapshot, make_targeted(1.43, 0.0), true);
    if (pass.kind != behavior::KickProfileKind::ParameterizedContact ||
        !near(pass.target_distance_m, 2.0) ||
        pass.mode != decision::KickMode::TargetedPass) {
        std::cerr << "validated pass profile was not accepted\n";
        return 1;
    }

    const auto dribble = behavior::make_kick_execution_profile(
        snapshot, make_dribble(), true);
    const auto angled_dribble = behavior::make_kick_execution_profile(
        snapshot, make_dribble(2.50), true);
    if (dribble.kind != behavior::KickProfileKind::ProceduralContact ||
        angled_dribble.kind != behavior::KickProfileKind::ProceduralContact ||
        !near(dribble.target_distance_m, 0.55) ||
        !near(dribble.requested_speed_mps, 0.90) ||
        dribble.mode != decision::KickMode::DribbleTouch) {
        std::cerr << "validated procedural short-touch profile was not accepted\n";
        return 1;
    }
    const auto shot = behavior::make_kick_execution_profile(
        snapshot, make_shot(), true);
    if (shot.kind != behavior::KickProfileKind::ProceduralContact ||
        !near(shot.target_distance_m, 4.0) ||
        !near(shot.requested_speed_mps, 2.50) ||
        shot.mode != decision::KickMode::Shot) {
        std::cerr << "validated procedural shot profile was not accepted\n";
        return 1;
    }
    const auto clear = behavior::make_kick_execution_profile(
        snapshot, make_clear(), true);
    if (clear.kind != behavior::KickProfileKind::ProceduralContact ||
        !near(clear.target_distance_m, 6.0) ||
        !near(clear.requested_speed_mps, 3.50) ||
        clear.mode != decision::KickMode::Clear) {
        std::cerr << "validated procedural clear profile was not accepted\n";
        return 1;
    }

    const auto left = behavior::make_kick_execution_profile(
        snapshot, make_targeted(1.43, 1.5), true);
    const auto right = behavior::make_kick_execution_profile(
        snapshot, make_targeted(1.43, -1.5), true);
    if (!(left.local_drive_target_m[1] > -0.04) ||
        !(right.local_drive_target_m[1] < -0.04) ||
        !near(left.relative_target_angle_deg, 1.5, 1.0e-8) ||
        !near(right.relative_target_angle_deg, -1.5, 1.0e-8)) {
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
    const auto unsupported_pass_speed = behavior::make_kick_execution_profile(
        snapshot, make_targeted(1.42, 0.0), true);
    const auto outside_dribble_angle = behavior::make_kick_execution_profile(
        snapshot, make_dribble(3.01), true);
    const auto outside_shot_angle = behavior::make_kick_execution_profile(
        snapshot, make_shot(4.0, 1.01), true);
    const auto outside_shot_distance = behavior::make_kick_execution_profile(
        snapshot, make_shot(4.51, 0.0), true);
    const auto outside_clear_angle = behavior::make_kick_execution_profile(
        snapshot, make_clear(6.0, 1.01), true);
    const auto outside_clear_distance = behavior::make_kick_execution_profile(
        snapshot, make_clear(6.51, 0.0), true);
    if (invalid_profile.kind != behavior::KickProfileKind::StableFallback ||
        outside_angle.kind != behavior::KickProfileKind::StableFallback ||
        outside_speed.kind != behavior::KickProfileKind::StableFallback ||
        unsupported_pass_speed.kind != behavior::KickProfileKind::StableFallback ||
        outside_dribble_angle.kind != behavior::KickProfileKind::StableFallback ||
        outside_shot_angle.kind != behavior::KickProfileKind::StableFallback ||
        outside_shot_distance.kind != behavior::KickProfileKind::StableFallback ||
        outside_clear_angle.kind != behavior::KickProfileKind::StableFallback ||
        outside_clear_distance.kind != behavior::KickProfileKind::StableFallback) {
        std::cerr << "unsupported requests did not use the stable fallback\n";
        return 1;
    }

    return 0;
}
