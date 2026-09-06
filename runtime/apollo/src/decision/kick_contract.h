// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>

namespace decision::kick_contract {

inline constexpr double kMinimumTargetDistanceM = 0.25;
inline constexpr double kMaximumTargetDistanceM = 8.0;
inline constexpr double kMaximumTargetAngleDeg = 15.0;
inline constexpr double kMinimumRequestedSpeedMps = 0.8;
inline constexpr double kMaximumRequestedSpeedMps = 3.5;

// Permissive match fallback envelope for every static-base procedural
// trajectory.  These values deliberately admit normal gait phases instead of
// waiting for an almost motionless neutral pose.  The runner still requires a
// finite complete joint state, an upright robot and a usable ball track.  A
// phase-conditioned learned transition should eventually replace this broad
// capture rather than tightening the match trigger again.
inline constexpr double kProceduralMaximumStartPlanarSpeedMps = 1.20;
inline constexpr double kProceduralMaximumStartTiltRateDegS = 90.0;
inline constexpr double kProceduralMaximumStartLegPositionDeg = 100.0;
inline constexpr double kProceduralMaximumStartLegVelocityDegS = 360.0;

// Shared release slot for the model-independent forward-contact macro.  The
// restart coordinator, setup controller, and final release gate must agree on
// this pose; otherwise coordination can wait forever at a pose the action
// layer already considers executable (or authorize a pose it will reject).
inline constexpr double kForwardContactBallLocalXM = 0.34;
inline constexpr double kForwardContactBallLocalXToleranceM = 0.14;
inline constexpr double kForwardContactBallLocalYM = 0.0;
inline constexpr double kForwardContactBallLocalYToleranceM = 0.18;
inline constexpr double kForwardContactMaximumTargetAngleDeg = 25.0;

struct ParameterizedPassAnchorContract {
    double target_distance_m;
    double maximum_distance_error_m;
    double requested_speed_mps;
    double maximum_speed_error_mps;
};

// The 2 m anchor is backed by the dense residual table. The 3.5 m and 5 m
// anchors remain experimental. The broad overlap below is a match-availability
// policy, not evidence that one trajectory continuously scales across it.
inline constexpr std::array<ParameterizedPassAnchorContract, 3>
kParameterizedPassAnchors{{
    {2.0, 1.00, 1.43, 0.35},
    {3.5, 1.00, 2.20, 0.35},
    {5.0, 1.00, 3.00, 0.35},
}};
inline constexpr double kParameterizedPassMinimumTargetDistanceM = 1.00;
inline constexpr double kParameterizedPassMaximumTargetDistanceM = 6.00;
inline constexpr double kParameterizedPassMaximumTargetAngleDeg = 15.0;
inline constexpr double kParameterizedPassRequestedSpeedMps = 1.43;
inline constexpr double kParameterizedPassMinimumRequestedSpeedMps = 1.43;
inline constexpr double kParameterizedPassMaximumRequestedSpeedMps = 3.00;

// The currently deployable learned transition actor was trained only on the
// fixed-2 m corpus.  Its policy consumes the live gait phase and was exposed
// to a materially wider approach-yaw and ball-pose distribution than the
// static residual/trajectory bank.  Keep this contract separate: widening the
// static bank to these limits would make its deterministic contact unsafe,
// while applying the static limits to this actor discards states it learned to
// recover from.
inline constexpr double kLearnedTransitionMinimumTargetDistanceM = 1.90;
inline constexpr double kLearnedTransitionMaximumTargetDistanceM = 2.10;
inline constexpr double kLearnedTransitionMaximumTargetAngleDeg = 12.0;
inline constexpr double kLearnedTransitionMinimumRequestedSpeedMps = 1.23;
inline constexpr double kLearnedTransitionMaximumRequestedSpeedMps = 1.63;
inline constexpr double kLearnedTransitionMinimumBallLocalXM = 0.30;
inline constexpr double kLearnedTransitionMaximumBallLocalXM = 0.39;
inline constexpr double kLearnedTransitionMinimumBallLocalYM = -0.03;
inline constexpr double kLearnedTransitionMaximumBallLocalYM = 0.05;
// Version this independently even while it equals the current static value.
// The retained transition corpus contains only two of 460 accepted release
// states above 0.50 m/s, so its 0.57 m/s maximum is not enough evidence to
// broaden live control. Future gait-entry training may change this limit
// without weakening the procedural trajectory's stationary-start contract.
inline constexpr double kLearnedTransitionMaximumStartPlanarSpeedMps = 0.50;

inline bool learned_transition_pass_request_supported(
    double distance_m,
    double requested_speed_mps) {
    return std::isfinite(distance_m) &&
        std::isfinite(requested_speed_mps) &&
        distance_m >= kLearnedTransitionMinimumTargetDistanceM &&
        distance_m <= kLearnedTransitionMaximumTargetDistanceM &&
        requested_speed_mps >= kLearnedTransitionMinimumRequestedSpeedMps &&
        requested_speed_mps <= kLearnedTransitionMaximumRequestedSpeedMps;
}

inline double parameterized_pass_requested_speed_mps(double distance_m) {
    const ParameterizedPassAnchorContract* nearest =
        &kParameterizedPassAnchors.front();
    double nearest_error = std::abs(distance_m - nearest->target_distance_m);
    for (const auto& anchor : kParameterizedPassAnchors) {
        const double error = std::abs(distance_m - anchor.target_distance_m);
        if (error < nearest_error) {
            nearest = &anchor;
            nearest_error = error;
        }
    }
    return nearest->requested_speed_mps;
}

inline bool parameterized_pass_request_supported(
    double distance_m,
    double requested_speed_mps) {
    if (!std::isfinite(distance_m) ||
        !std::isfinite(requested_speed_mps) ||
        distance_m < kParameterizedPassMinimumTargetDistanceM ||
        distance_m > kParameterizedPassMaximumTargetDistanceM) {
        return false;
    }
    for (const auto& anchor : kParameterizedPassAnchors) {
        if (std::abs(distance_m - anchor.target_distance_m) <=
                anchor.maximum_distance_error_m &&
            std::abs(requested_speed_mps - anchor.requested_speed_mps) <=
                anchor.maximum_speed_error_mps) {
            return true;
        }
    }
    return false;
}

// First model-independent anchor. Its trajectory remains a fixed short touch;
// the widened request and release envelope is an availability-first fallback.
inline constexpr double kProceduralDribbleMinimumTargetDistanceM = 0.25;
inline constexpr double kProceduralDribbleMaximumTargetDistanceM = 0.90;
inline constexpr double kProceduralDribbleMaximumTargetAngleDeg = 15.0;
inline constexpr double kProceduralDribbleRequestedSpeedMps = 0.90;
// Decision and motion share this deliberately broad contact corridor so a
// decision-layer release is not rejected one cycle later.
inline constexpr double kProceduralDribbleBallLocalXM = 0.32;
inline constexpr double kProceduralDribbleBallLocalXToleranceM = 0.12;
inline constexpr double kProceduralDribbleBallLocalYM = 0.04;
inline constexpr double kProceduralDribbleBallLocalYToleranceM = 0.15;
inline constexpr double kProceduralDribbleMinimumBallLocalYM = -0.11;
inline constexpr double kProceduralDribbleMaximumBallLocalYM = 0.19;

// Exact-physics 4 m shot teacher. Its evidence remains attached to the asset;
// the wider match envelope below intentionally exceeds that validated slice.
inline constexpr double kProceduralShotMinimumTargetDistanceM = 3.00;
inline constexpr double kProceduralShotMaximumTargetDistanceM = 5.00;
inline constexpr double kProceduralShotMaximumTargetAngleDeg = 15.0;
inline constexpr double kProceduralShotRequestedSpeedMps = 2.50;
inline constexpr double kProceduralShotBallLocalXM = 0.3260;
inline constexpr double kProceduralShotBallLocalXRangeM = 0.1200;
inline constexpr double kProceduralShotBallLocalYM = 0.0400;
inline constexpr double kProceduralShotBallLocalYRangeM = 0.1500;

// Safety-clearance teacher. Success means at least 4.5 m of forward progress
// inside a 1.5 m half-corridor while the robot remains controllable; unlike a
// pass or shot, exact landing range and arrival speed are not claimed.
inline constexpr double kProceduralClearMinimumTargetDistanceM = 5.00;
inline constexpr double kProceduralClearMaximumTargetDistanceM = 7.00;
inline constexpr double kProceduralClearMaximumTargetAngleDeg = 15.0;
inline constexpr double kProceduralClearRequestedSpeedMps = 3.50;
inline constexpr double kProceduralClearBallLocalXM = 0.3260;
inline constexpr double kProceduralClearBallLocalXRangeM = 0.1200;
inline constexpr double kProceduralClearBallLocalYM = 0.0400;
inline constexpr double kProceduralClearBallLocalYRangeM = 0.1500;

}  // namespace decision::kick_contract
