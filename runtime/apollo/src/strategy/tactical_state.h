// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/world/world_snapshot.h"

#include <optional>
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
/// The default is the final minute of RCSSServerMJ's five-minute match.  Tests
/// and alternate runtimes may still provide an explicit threshold.
enum class TacticalRiskMode {
    Balanced,
    ProtectLead,
    ChaseGoal,
};

struct TacticalRiskParameters {
    double late_match_threshold_s{240.0};
};

struct TacticalState {
    PossessionOwner possession{PossessionOwner::Unknown};
    TacticalPhase phase{TacticalPhase::Unknown};
    double possession_confidence{0.0};
    double nearest_teammate_ball_distance_m{0.0};
    double nearest_opponent_ball_distance_m{0.0};
    double nearest_teammate_ball_time_s{0.0};
    double nearest_opponent_ball_time_s{0.0};
    int ball_owner_player_number{0};
    bool ball_owner_is_teammate{false};
    int score_difference{0};
    double match_time_s{0.0};
    TacticalRiskMode risk_mode{TacticalRiskMode::Balanced};
};

TacticalState build_tactical_state(const world::WorldSnapshot& snapshot);
TacticalState build_tactical_state(
    const world::WorldSnapshot& snapshot,
    TacticalRiskParameters parameters);

/// Stabilizes noisy reach-time possession estimates without hiding decisive
/// turnovers.  The tracker is deliberately small and deterministic so every
/// agent given the same world snapshots derives the same tactical phase.
class TacticalStateTracker {
public:
    struct Parameters {
        double possession_confirmation_s{0.40};
        double strong_evidence_confidence{0.85};
        double counter_press_window_s{1.20};
        double maximum_ball_age_s{0.75};
    };

    TacticalStateTracker();
    explicit TacticalStateTracker(Parameters parameters);

    TacticalState update(const world::WorldSnapshot& snapshot);
    void reset();

private:
    Parameters parameters_;
    bool initialized_{false};
    PossessionOwner stable_possession_{PossessionOwner::Unknown};
    int stable_owner_player_number_{0};
    bool stable_owner_is_teammate_{false};
    std::optional<PossessionOwner> pending_possession_;
    double pending_since_s_{0.0};
    double counter_press_until_s_{-1.0};
    double last_server_time_s_{-1.0};
};

std::string_view to_string(PossessionOwner owner);
std::string_view to_string(TacticalPhase phase);
std::string_view to_string(TacticalRiskMode mode);

}  // namespace strategy
