// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace decision::kick_contract {

inline constexpr double kMinimumTargetDistanceM = 0.25;
inline constexpr double kMaximumTargetDistanceM = 8.0;
inline constexpr double kMaximumTargetAngleDeg = 15.0;
inline constexpr double kMinimumRequestedSpeedMps = 0.8;
inline constexpr double kMaximumRequestedSpeedMps = 3.0;

// The currently shipped residual table is calibrated at 1.43 m/s, 0 deg and
// a 2.0 m target. Its selection tolerances are part of the physical contract
// until a wider table is promoted; keeping these limits here prevents the
// planner from committing targets that MotionManager must reject later.
inline constexpr double kParameterizedPassMinimumTargetDistanceM = 1.45;
inline constexpr double kParameterizedPassMaximumTargetDistanceM = 2.55;
inline constexpr double kParameterizedPassMaximumTargetAngleDeg = 2.0;
inline constexpr double kParameterizedPassRequestedSpeedMps = 1.43;

}  // namespace decision::kick_contract
