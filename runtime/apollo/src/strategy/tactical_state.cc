// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/tactical_state.h"

#include "src/math/math_utils.h"
#include "src/strategy/reach_time_model.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strategy {

namespace {

double nearest_distance(
    const std::vector<world::PlayerObservation>& players,
    const std::array<double, 2>& ball,
    int self_player_number,
    const world::WorldSnapshot& snapshot,
    bool include_self) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& player : players) {
        if (player.player_number <= 0 || player.fallen) {
            continue;
        }
        if (player.player_number == self_player_number && include_self) {
            best = std::min(best, math::planar_dist(
                {snapshot.self.position_m[0], snapshot.self.position_m[1]}, ball));
            continue;
        }
        if (!player.seen &&
            (player.last_seen_time < 0.0 ||
             snapshot.server_time - player.last_seen_time > 2.0)) {
            continue;
        }
        best = std::min(best, math::planar_dist(
            {player.position_m[0], player.position_m[1]}, ball));
    }
    return best;
}

double nearest_reach_time(
    const std::vector<world::PlayerObservation>& players,
    const std::array<double, 2>& ball,
    int self_player_number,
    const world::WorldSnapshot& snapshot,
    bool include_self,
    const ReachTimeModel& model) {
    double best = std::numeric_limits<double>::infinity();
    if (include_self &&
        snapshot.self.position_m[2] >= world::kFallenHeightThresholdM) {
        const std::array<double, 2> self{
            snapshot.self.position_m[0], snapshot.self.position_m[1]};
        const double yaw_deg =
            world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                snapshot.self.orientation_wxyz);
        best = model.estimate_s(self, ball, yaw_deg);
    }
    for (const auto& player : players) {
        if (player.player_number <= 0 || player.fallen) continue;
        if (player.player_number == self_player_number && include_self) continue;
        if (!player.seen &&
            (player.last_seen_time < 0.0 ||
             snapshot.server_time - player.last_seen_time > 2.0)) {
            continue;
        }
        best = std::min(
            best,
            model.estimate_s(
                {player.position_m[0], player.position_m[1]}, ball));
    }
    return best;
}

}  // namespace

TacticalState build_tactical_state(const world::WorldSnapshot& snapshot) {
    return build_tactical_state(snapshot, TacticalRiskParameters{});
}

TacticalState build_tactical_state(
    const world::WorldSnapshot& snapshot,
    TacticalRiskParameters parameters) {
    TacticalState state;
    state.score_difference = snapshot.own_score - snapshot.opponent_score;
    state.match_time_s = snapshot.match_time_s;
    const double late_match_threshold_s =
        std::isfinite(parameters.late_match_threshold_s) &&
            parameters.late_match_threshold_s > 0.0
        ? parameters.late_match_threshold_s
        : TacticalRiskParameters{}.late_match_threshold_s;
    const bool late_match = std::isfinite(snapshot.match_time_s) &&
        snapshot.match_time_s >= late_match_threshold_s;
    if (late_match && state.score_difference > 0) {
        state.risk_mode = TacticalRiskMode::ProtectLead;
    } else if (late_match && state.score_difference < 0) {
        state.risk_mode = TacticalRiskMode::ChaseGoal;
    }

    if (snapshot.play_mode_group == world::PlayModeGroup::OurKick ||
        snapshot.play_mode_group == world::PlayModeGroup::TheirKick) {
        state.phase = TacticalPhase::SetPlay;
    }

    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    state.nearest_teammate_ball_distance_m = nearest_distance(
        snapshot.teammates, ball, snapshot.player_number, snapshot, true);
    state.nearest_opponent_ball_distance_m = nearest_distance(
        snapshot.opponents, ball, snapshot.player_number, snapshot, false);
    for (const auto& opponent : snapshot.shared_opponents) {
        if (!opponent.seen) continue;
        state.nearest_opponent_ball_distance_m = std::min(
            state.nearest_opponent_ball_distance_m,
            math::planar_dist(
                {opponent.position_m[0], opponent.position_m[1]}, ball));
    }

    const ReachTimeModel teammate_reach(
        ReachTimeModel::Parameters{0.90, 120.0, 0.25, 0.55, 0.20});
    const ReachTimeModel opponent_reach(
        ReachTimeModel::Parameters{1.35, 180.0, 0.15, 0.55, 0.10});
    state.nearest_teammate_ball_time_s = nearest_reach_time(
        snapshot.teammates, ball, snapshot.player_number, snapshot, true,
        teammate_reach);
    state.nearest_opponent_ball_time_s = nearest_reach_time(
        snapshot.opponents, ball, snapshot.player_number, snapshot, false,
        opponent_reach);
    state.nearest_opponent_ball_time_s = std::min(
        state.nearest_opponent_ball_time_s,
        nearest_reach_time(
            snapshot.shared_opponents, ball, snapshot.player_number, snapshot,
            false, opponent_reach));

    const bool our_finite = std::isfinite(state.nearest_teammate_ball_time_s);
    const bool their_finite = std::isfinite(state.nearest_opponent_ball_time_s);
    if (!our_finite && !their_finite) {
        state.possession = PossessionOwner::Unknown;
        state.possession_confidence = 0.0;
    } else if (!their_finite) {
        state.possession = PossessionOwner::Ours;
        state.possession_confidence = 1.0;
    } else if (!our_finite) {
        state.possession = PossessionOwner::Theirs;
        state.possession_confidence = 1.0;
    } else {
        const double advantage_s =
            state.nearest_opponent_ball_time_s -
            state.nearest_teammate_ball_time_s;
        state.possession_confidence = std::clamp(
            std::abs(advantage_s) / 1.0, 0.0, 1.0);
        if (advantage_s > 0.25) {
            state.possession = PossessionOwner::Ours;
        } else if (advantage_s < -0.25) {
            state.possession = PossessionOwner::Theirs;
        } else {
            state.possession = PossessionOwner::Contested;
        }
    }

    if (state.phase != TacticalPhase::SetPlay) {
        switch (state.possession) {
            case PossessionOwner::Ours: state.phase = TacticalPhase::Attack; break;
            case PossessionOwner::Theirs: state.phase = TacticalPhase::Defend; break;
            case PossessionOwner::Contested: state.phase = TacticalPhase::Transition; break;
            case PossessionOwner::Unknown: state.phase = TacticalPhase::Unknown; break;
        }
    }
    return state;
}

std::string_view to_string(PossessionOwner owner) {
    switch (owner) {
        case PossessionOwner::Unknown: return "Unknown";
        case PossessionOwner::Ours: return "Ours";
        case PossessionOwner::Theirs: return "Theirs";
        case PossessionOwner::Contested: return "Contested";
    }
    return "Unknown";
}

std::string_view to_string(TacticalPhase phase) {
    switch (phase) {
        case TacticalPhase::Unknown: return "Unknown";
        case TacticalPhase::Attack: return "Attack";
        case TacticalPhase::Defend: return "Defend";
        case TacticalPhase::Transition: return "Transition";
        case TacticalPhase::SetPlay: return "SetPlay";
    }
    return "Unknown";
}

std::string_view to_string(TacticalRiskMode mode) {
    switch (mode) {
        case TacticalRiskMode::Balanced: return "Balanced";
        case TacticalRiskMode::ProtectLead: return "ProtectLead";
        case TacticalRiskMode::ChaseGoal: return "ChaseGoal";
    }
    return "Balanced";
}

}  // namespace strategy
