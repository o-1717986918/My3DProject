// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/decision/high_level_command.h"
#include "src/world/world_snapshot.h"

#include <array>

namespace behavior {

enum class KickProfileKind {
    StableFallback,
    ParameterizedContact,
};

/// Bounded parameters consumed by the current walk-backed contact executor.
/// This is deliberately separate from a learned kick policy: it makes the
/// strategy-to-motion contract executable and measurable while preserving the
/// accepted fixed-contact profile whenever a request is unsupported.
struct KickExecutionProfile {
    KickProfileKind kind{KickProfileKind::StableFallback};
    std::array<double, 2> local_drive_target_m{0.50, -0.04};
    double drive_duration_s{0.65};
    double total_duration_s{1.0};
    double requested_speed_mps{0.0};
    double relative_target_angle_deg{0.0};
};

/// Converts a target-aware high-level kick into a conservative, bounded
/// contact profile. Unsupported, non-finite, or out-of-envelope requests use
/// the exact previously validated stable fallback.
KickExecutionProfile make_kick_execution_profile(
    const world::WorldSnapshot& snapshot,
    const decision::KickCommand& command,
    bool parameterized_enabled);

}  // namespace behavior
