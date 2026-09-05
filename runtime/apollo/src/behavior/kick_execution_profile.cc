// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/kick_execution_profile.h"

#include "src/decision/kick_contract.h"
#include "src/math/math_utils.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <cmath>

namespace behavior {

namespace {

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
        !snapshot.ball.position_valid ||
        !std::isfinite(command.requested_ball_speed_mps) ||
        command.requested_ball_speed_mps < decision::kick_contract::kMinimumRequestedSpeedMps ||
        command.requested_ball_speed_mps > decision::kick_contract::kMaximumRequestedSpeedMps ||
        !finite_point(*command.target_point_m) ||
        !finite_point({
            snapshot.self.position_m[0], snapshot.self.position_m[1]})) {
        return profile;
    }

    const std::array<double, 2> target_delta{
        (*command.target_point_m)[0] - snapshot.ball.position_m[0],
        (*command.target_point_m)[1] - snapshot.ball.position_m[1],
    };
    const double target_distance_m = math::norm2(target_delta);

    const double target_heading_deg = math::vector_angle_deg(target_delta);
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double relative_angle_deg =
        math::normalize_deg(target_heading_deg - self_yaw_deg);
    if (!std::isfinite(relative_angle_deg)) {
        return profile;
    }

    if (command.mode == decision::KickMode::DribbleTouch ||
        command.mode == decision::KickMode::Shot ||
        command.mode == decision::KickMode::Clear) {
        double minimum_distance_m = 0.0;
        double maximum_distance_m = 0.0;
        double maximum_angle_deg = 0.0;
        double requested_speed_mps = 0.0;
        if (command.mode == decision::KickMode::DribbleTouch) {
            minimum_distance_m =
                decision::kick_contract::kProceduralDribbleMinimumTargetDistanceM;
            maximum_distance_m =
                decision::kick_contract::kProceduralDribbleMaximumTargetDistanceM;
            maximum_angle_deg =
                decision::kick_contract::kProceduralDribbleMaximumTargetAngleDeg;
            requested_speed_mps =
                decision::kick_contract::kProceduralDribbleRequestedSpeedMps;
        } else if (command.mode == decision::KickMode::Shot) {
            minimum_distance_m =
                decision::kick_contract::kProceduralShotMinimumTargetDistanceM;
            maximum_distance_m =
                decision::kick_contract::kProceduralShotMaximumTargetDistanceM;
            maximum_angle_deg =
                decision::kick_contract::kProceduralShotMaximumTargetAngleDeg;
            requested_speed_mps =
                decision::kick_contract::kProceduralShotRequestedSpeedMps;
        } else {
            minimum_distance_m =
                decision::kick_contract::kProceduralClearMinimumTargetDistanceM;
            maximum_distance_m =
                decision::kick_contract::kProceduralClearMaximumTargetDistanceM;
            maximum_angle_deg =
                decision::kick_contract::kProceduralClearMaximumTargetAngleDeg;
            requested_speed_mps =
                decision::kick_contract::kProceduralClearRequestedSpeedMps;
        }
        if (target_distance_m < minimum_distance_m ||
            target_distance_m > maximum_distance_m ||
            std::abs(relative_angle_deg) > maximum_angle_deg ||
            std::abs(command.requested_ball_speed_mps - requested_speed_mps) >
                1.0e-9) {
            return profile;
        }
        profile.kind = KickProfileKind::ProceduralContact;
        profile.requested_speed_mps = command.requested_ball_speed_mps;
        profile.relative_target_angle_deg = relative_angle_deg;
        profile.target_distance_m = target_distance_m;
        profile.mode = command.mode;
        profile.total_duration_s = 1.20;
        return profile;
    }

    if (command.mode != decision::KickMode::TargetedPass ||
        !decision::kick_contract::parameterized_pass_request_supported(
            target_distance_m, command.requested_ball_speed_mps) ||
        std::abs(relative_angle_deg) >
            decision::kick_contract::kParameterizedPassMaximumTargetAngleDeg) {
        return profile;
    }

    const double speed_fraction =
        (command.requested_ball_speed_mps - decision::kick_contract::kMinimumRequestedSpeedMps) /
        (decision::kick_contract::kMaximumRequestedSpeedMps -
         decision::kick_contract::kMinimumRequestedSpeedMps);
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
