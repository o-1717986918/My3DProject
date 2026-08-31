// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/kick_execution_profile.h"

#include "src/math/math_utils.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <cmath>

namespace behavior {

namespace {

constexpr double kMinimumTargetDistanceM = 0.25;
constexpr double kMaximumTargetAngleDeg = 15.0;
constexpr double kMinimumRequestedSpeedMps = 0.8;
constexpr double kMaximumRequestedSpeedMps = 3.0;
constexpr double kMinimumDriveTargetM = 0.50;
constexpr double kMaximumDriveTargetM = 0.85;
constexpr double kBaseLateralTargetM = -0.04;
constexpr double kMaximumLateralCorrectionM = 0.08;
constexpr double kMinimumDriveDurationS = 0.65;
constexpr double kMaximumDriveDurationS = 0.82;
constexpr double kStabilizationDurationS = 0.35;

bool finite_point(const std::array<double, 2>& point) {
    return std::isfinite(point[0]) && std::isfinite(point[1]);
}

}  // namespace

KickExecutionProfile make_kick_execution_profile(
    const world::WorldSnapshot& snapshot,
    const decision::KickCommand& command,
    bool parameterized_enabled) {
    KickExecutionProfile profile;
    if (!parameterized_enabled ||
        command.mode == decision::KickMode::ForwardContact ||
        !command.target_point_m.has_value() ||
        !std::isfinite(command.requested_ball_speed_mps) ||
        command.requested_ball_speed_mps < kMinimumRequestedSpeedMps ||
        command.requested_ball_speed_mps > kMaximumRequestedSpeedMps ||
        !finite_point(*command.target_point_m) ||
        !finite_point({
            snapshot.self.position_m[0], snapshot.self.position_m[1]})) {
        return profile;
    }

    const std::array<double, 2> target_delta{
        (*command.target_point_m)[0] - snapshot.self.position_m[0],
        (*command.target_point_m)[1] - snapshot.self.position_m[1],
    };
    const double target_distance_m = math::norm2(target_delta);
    if (target_distance_m < kMinimumTargetDistanceM) {
        return profile;
    }

    const double target_heading_deg = math::vector_angle_deg(target_delta);
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double relative_angle_deg =
        math::normalize_deg(target_heading_deg - self_yaw_deg);
    if (!std::isfinite(relative_angle_deg) ||
        std::abs(relative_angle_deg) > kMaximumTargetAngleDeg) {
        return profile;
    }

    const double speed_fraction =
        (command.requested_ball_speed_mps - kMinimumRequestedSpeedMps) /
        (kMaximumRequestedSpeedMps - kMinimumRequestedSpeedMps);
    const double drive_target =
        kMinimumDriveTargetM +
        speed_fraction * (kMaximumDriveTargetM - kMinimumDriveTargetM);
    const double drive_duration =
        kMinimumDriveDurationS +
        speed_fraction * (kMaximumDriveDurationS - kMinimumDriveDurationS);
    const double lateral_correction = std::clamp(
        std::tan(math::deg_to_rad(relative_angle_deg)) * 0.30,
        -kMaximumLateralCorrectionM,
        kMaximumLateralCorrectionM);

    profile.kind = KickProfileKind::ParameterizedContact;
    profile.local_drive_target_m = {
        drive_target,
        kBaseLateralTargetM + lateral_correction,
    };
    profile.drive_duration_s = drive_duration;
    profile.total_duration_s = drive_duration + kStabilizationDurationS;
    profile.requested_speed_mps = command.requested_ball_speed_mps;
    profile.relative_target_angle_deg = relative_angle_deg;
    profile.target_distance_m = target_distance_m;
    profile.mode = command.mode;
    return profile;
}

}  // namespace behavior
