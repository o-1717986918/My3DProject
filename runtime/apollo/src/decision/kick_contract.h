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

struct ParameterizedPassAnchorContract {
    double target_distance_m;
    double maximum_distance_error_m;
    double requested_speed_mps;
    double maximum_speed_error_mps;
};

// The 2 m anchor is backed by the dense residual table. The 3.5 m and 5 m
// anchors are narrow deterministic trajectories and remain experimental; all
// three are selected discretely instead of pretending that one trajectory can
// continuously scale across the complete range.
inline constexpr std::array<ParameterizedPassAnchorContract, 3>
kParameterizedPassAnchors{{
    {2.0, 0.75, 1.43, 0.20},
    {3.5, 0.75, 2.20, 0.20},
    {5.0, 0.75, 3.00, 0.20},
}};
inline constexpr double kParameterizedPassMinimumTargetDistanceM = 1.45;
inline constexpr double kParameterizedPassMaximumTargetDistanceM = 5.75;
inline constexpr double kParameterizedPassMaximumTargetAngleDeg = 2.0;
inline constexpr double kParameterizedPassRequestedSpeedMps = 1.43;
inline constexpr double kParameterizedPassMinimumRequestedSpeedMps = 1.43;
inline constexpr double kParameterizedPassMaximumRequestedSpeedMps = 3.00;

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

// First model-independent anchor. This is intentionally a narrow short-touch
// contract: widening it to pass or shot distances requires new physical
// anchors and the same held-out/server promotion gates.
inline constexpr double kProceduralDribbleMinimumTargetDistanceM = 0.45;
inline constexpr double kProceduralDribbleMaximumTargetDistanceM = 0.65;
// At 0.55 m, a six-degree body/target mismatch is about 5.8 cm laterally: it
// is acceptable for a recovery dribble touch, but not for a pass or shot.
// Keeping this a dribble-only contract avoids starving contact while the
// stricter directional actions retain their measured release envelopes.
inline constexpr double kProceduralDribbleMaximumTargetAngleDeg = 6.0;
inline constexpr double kProceduralDribbleRequestedSpeedMps = 0.90;
inline constexpr double kProceduralDribbleMinimumBallLocalYM = 0.02;
inline constexpr double kProceduralDribbleMaximumBallLocalYM = 0.06;

// Exact-physics 4 m shot teacher. The trajectory passed 100/100 independently
// seeded held-out trials only inside this narrow release slot; strategy and
// motion must not widen the distance, angle, speed, or ball-position contract
// without new evidence.
inline constexpr double kProceduralShotMinimumTargetDistanceM = 3.50;
inline constexpr double kProceduralShotMaximumTargetDistanceM = 4.50;
inline constexpr double kProceduralShotMaximumTargetAngleDeg = 1.0;
inline constexpr double kProceduralShotRequestedSpeedMps = 2.50;
inline constexpr double kProceduralShotBallLocalXM = 0.3260;
inline constexpr double kProceduralShotBallLocalXRangeM = 0.0140;
inline constexpr double kProceduralShotBallLocalYM = 0.0400;
inline constexpr double kProceduralShotBallLocalYRangeM = 0.0120;

// Safety-clearance teacher. Success means at least 4.5 m of forward progress
// inside a 1.5 m half-corridor while the robot remains controllable; unlike a
// pass or shot, exact landing range and arrival speed are not claimed.
inline constexpr double kProceduralClearMinimumTargetDistanceM = 5.50;
inline constexpr double kProceduralClearMaximumTargetDistanceM = 6.50;
inline constexpr double kProceduralClearMaximumTargetAngleDeg = 1.0;
inline constexpr double kProceduralClearRequestedSpeedMps = 3.50;
inline constexpr double kProceduralClearBallLocalXM = 0.3260;
inline constexpr double kProceduralClearBallLocalXRangeM = 0.0140;
inline constexpr double kProceduralClearBallLocalYM = 0.0400;
inline constexpr double kProceduralClearBallLocalYRangeM = 0.0120;

}  // namespace decision::kick_contract
