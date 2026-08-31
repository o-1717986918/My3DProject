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

struct TacticalState {
    PossessionOwner possession{PossessionOwner::Unknown};
    TacticalPhase phase{TacticalPhase::Unknown};
    double possession_confidence{0.0};
    double nearest_teammate_ball_distance_m{0.0};
    double nearest_opponent_ball_distance_m{0.0};
    int score_difference{0};
    double match_time_s{0.0};
};

TacticalState build_tactical_state(const world::WorldSnapshot& snapshot);
std::string_view to_string(PossessionOwner owner);
std::string_view to_string(TacticalPhase phase);

}  // namespace strategy
