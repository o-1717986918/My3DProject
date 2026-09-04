// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/world/world_snapshot.h"

#include <string_view>

namespace strategy {

enum class PossessionOwner {
    Unknown,
    Ours,
    Theirs,
    Contested,
};

enum class TacticalPhase {
    Unknown,
    Attack,
    Defend,
    Transition,
    SetPlay,
};

/// Risk preference derived only when a valid late-match clock is available.
/// The threshold is explicit so a simulator with an unknown match duration
/// stays Balanced instead of guessing from a hard-coded game length.
enum class TacticalRiskMode {
    Balanced,
    ProtectLead,
    ChaseGoal,
};

struct TacticalRiskParameters {
    double late_match_threshold_s{300.0};
};

struct TacticalState {
    PossessionOwner possession{PossessionOwner::Unknown};
    TacticalPhase phase{TacticalPhase::Unknown};
    double possession_confidence{0.0};
    double nearest_teammate_ball_distance_m{0.0};
    double nearest_opponent_ball_distance_m{0.0};
    double nearest_teammate_ball_time_s{0.0};
    double nearest_opponent_ball_time_s{0.0};
    int score_difference{0};
    double match_time_s{0.0};
    TacticalRiskMode risk_mode{TacticalRiskMode::Balanced};
};

TacticalState build_tactical_state(const world::WorldSnapshot& snapshot);
TacticalState build_tactical_state(
    const world::WorldSnapshot& snapshot,
    TacticalRiskParameters parameters);
std::string_view to_string(PossessionOwner owner);
std::string_view to_string(TacticalPhase phase);
std::string_view to_string(TacticalRiskMode mode);

}  // namespace strategy
