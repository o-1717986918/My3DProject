// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace decision::kick_contract {

inline constexpr double kMinimumTargetDistanceM = 0.25;
inline constexpr double kMaximumTargetDistanceM = 8.0;
inline constexpr double kMaximumTargetAngleDeg = 15.0;
inline constexpr double kMinimumRequestedSpeedMps = 0.8;
inline constexpr double kMaximumRequestedSpeedMps = 3.5;

// The currently shipped residual table is calibrated at 1.43 m/s, 0 deg and
// a 2.0 m target. Its selection tolerances are part of the physical contract
// until a wider table is promoted; keeping these limits here prevents the
// planner from committing targets that MotionManager must reject later.
inline constexpr double kParameterizedPassMinimumTargetDistanceM = 1.45;
inline constexpr double kParameterizedPassMaximumTargetDistanceM = 2.55;
inline constexpr double kParameterizedPassMaximumTargetAngleDeg = 2.0;
inline constexpr double kParameterizedPassRequestedSpeedMps = 1.43;

// First model-independent anchor. This is intentionally a narrow short-touch
// contract: widening it to pass or shot distances requires new physical
// anchors and the same held-out/server promotion gates.
inline constexpr double kProceduralDribbleMinimumTargetDistanceM = 0.45;
inline constexpr double kProceduralDribbleMaximumTargetDistanceM = 0.65;
// At 0.55 m, a three-degree body/target mismatch is under 3 cm laterally and
// remains inside the short-touch control corridor. The former one-degree
// boundary rejected a natural release after decision-time alignment because
// localization yaw moved by 1.79 degrees before motion dispatch.
inline constexpr double kProceduralDribbleMaximumTargetAngleDeg = 3.0;
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
