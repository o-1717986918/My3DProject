// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/tactical_state.h"

#include "src/math/math_utils.h"
#include "src/strategy/reach_time_model.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

struct ReachCandidate {
    double time_s{std::numeric_limits<double>::infinity()};
    int player_number{0};
};

ReachCandidate nearest_reach_candidate(
    const std::vector<world::PlayerObservation>& players,
    const std::array<double, 2>& ball,
    int self_player_number,
    const world::WorldSnapshot& snapshot,
    bool include_self,
    const ReachTimeModel& model) {
    ReachCandidate best;
    if (include_self &&
        snapshot.self.position_m[2] >= world::kFallenHeightThresholdM) {
        const std::array<double, 2> self{
            snapshot.self.position_m[0], snapshot.self.position_m[1]};
        const double yaw_deg =
            world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                snapshot.self.orientation_wxyz);
        best = {model.estimate_s(self, ball, yaw_deg), self_player_number};
    }
    for (const auto& player : players) {
        if (player.player_number <= 0 || player.fallen) continue;
        if (player.player_number == self_player_number && include_self) continue;
        if (!player.seen &&
            (player.last_seen_time < 0.0 ||
             snapshot.server_time - player.last_seen_time > 2.0)) {
            continue;
        }
        const double time_s = model.estimate_s(
            {player.position_m[0], player.position_m[1]}, ball);
        if (time_s < best.time_s - 1.0e-9 ||
            (std::abs(time_s - best.time_s) <= 1.0e-9 &&
             (best.player_number == 0 ||
              player.player_number < best.player_number))) {
            best = {time_s, player.player_number};
        }
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

    const bool usable_ball = snapshot.ball.position_valid &&
        (snapshot.ball.visible || snapshot.ball.position_age_s <= 0.75);
    if (!usable_ball) {
        state.nearest_teammate_ball_distance_m =
            std::numeric_limits<double>::infinity();
        state.nearest_opponent_ball_distance_m =
            std::numeric_limits<double>::infinity();
        state.nearest_teammate_ball_time_s =
            std::numeric_limits<double>::infinity();
        state.nearest_opponent_ball_time_s =
            std::numeric_limits<double>::infinity();
        state.possession = PossessionOwner::Unknown;
        state.possession_confidence = 0.0;
        if (state.phase != TacticalPhase::SetPlay) {
            state.phase = TacticalPhase::Unknown;
        }
        return state;
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
    const ReachCandidate teammate_candidate = nearest_reach_candidate(
        snapshot.teammates, ball, snapshot.player_number, snapshot, true,
        teammate_reach);
    ReachCandidate opponent_candidate = nearest_reach_candidate(
        snapshot.opponents, ball, snapshot.player_number, snapshot, false,
        opponent_reach);
    const ReachCandidate shared_opponent_candidate = nearest_reach_candidate(
        snapshot.shared_opponents, ball, snapshot.player_number, snapshot,
        false, opponent_reach);
    if (shared_opponent_candidate.time_s < opponent_candidate.time_s - 1.0e-9 ||
        (std::abs(shared_opponent_candidate.time_s -
                  opponent_candidate.time_s) <= 1.0e-9 &&
         shared_opponent_candidate.player_number > 0 &&
         (opponent_candidate.player_number == 0 ||
          shared_opponent_candidate.player_number <
              opponent_candidate.player_number))) {
        opponent_candidate = shared_opponent_candidate;
    }
    state.nearest_teammate_ball_time_s = teammate_candidate.time_s;
    state.nearest_opponent_ball_time_s = opponent_candidate.time_s;

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
    if (state.possession == PossessionOwner::Ours) {
        state.ball_owner_player_number = teammate_candidate.player_number;
        state.ball_owner_is_teammate = true;
    } else if (state.possession == PossessionOwner::Theirs) {
        state.ball_owner_player_number = opponent_candidate.player_number;
        state.ball_owner_is_teammate = false;
    }
    return state;
}

namespace {

bool valid_tracker_parameters(const TacticalStateTracker::Parameters& parameters) {
    return std::isfinite(parameters.possession_confirmation_s) &&
        parameters.possession_confirmation_s >= 0.0 &&
        std::isfinite(parameters.strong_evidence_confidence) &&
        parameters.strong_evidence_confidence >= 0.0 &&
        parameters.strong_evidence_confidence <= 1.0 &&
        std::isfinite(parameters.counter_press_window_s) &&
        parameters.counter_press_window_s >= 0.0 &&
        std::isfinite(parameters.maximum_ball_age_s) &&
        parameters.maximum_ball_age_s > 0.0;
}

bool firm_possession(PossessionOwner owner) {
    return owner == PossessionOwner::Ours || owner == PossessionOwner::Theirs;
}

TacticalPhase phase_for(PossessionOwner owner) {
    switch (owner) {
        case PossessionOwner::Ours: return TacticalPhase::Attack;
        case PossessionOwner::Theirs: return TacticalPhase::Defend;
        case PossessionOwner::Contested: return TacticalPhase::Transition;
        case PossessionOwner::Unknown: return TacticalPhase::Unknown;
    }
    return TacticalPhase::Unknown;
}

}  // namespace

TacticalStateTracker::TacticalStateTracker()
    : TacticalStateTracker(Parameters{}) {}

TacticalStateTracker::TacticalStateTracker(Parameters parameters)
    : parameters_(parameters) {
    if (!valid_tracker_parameters(parameters_)) {
        throw std::invalid_argument("invalid tactical state tracker parameters");
    }
}

void TacticalStateTracker::reset() {
    initialized_ = false;
    stable_possession_ = PossessionOwner::Unknown;
    stable_owner_player_number_ = 0;
    stable_owner_is_teammate_ = false;
    pending_possession_.reset();
    pending_since_s_ = 0.0;
    counter_press_until_s_ = -1.0;
    last_server_time_s_ = -1.0;
}

TacticalState TacticalStateTracker::update(
    const world::WorldSnapshot& snapshot) {
    TacticalState state = build_tactical_state(snapshot);
    const bool time_regressed = last_server_time_s_ >= 0.0 &&
        snapshot.server_time + 1.0e-9 < last_server_time_s_;
    const bool usable_open_play = snapshot.play_mode == world::PlayMode::PlayOn &&
        snapshot.ball.position_valid &&
        (snapshot.ball.visible ||
         snapshot.ball.position_age_s <= parameters_.maximum_ball_age_s);
    if (time_regressed || !usable_open_play) {
        reset();
        last_server_time_s_ = snapshot.server_time;
        return state;
    }

    last_server_time_s_ = snapshot.server_time;
    if (!initialized_) {
        initialized_ = true;
        stable_possession_ = state.possession;
        stable_owner_player_number_ = state.ball_owner_player_number;
        stable_owner_is_teammate_ = state.ball_owner_is_teammate;
        pending_possession_.reset();
        return state;
    }

    const PossessionOwner observed = state.possession;
    if (!firm_possession(observed)) {
        pending_possession_.reset();
        state.possession = observed;
        state.phase = phase_for(observed);
        return state;
    }

    if (!firm_possession(stable_possession_)) {
        stable_possession_ = observed;
        stable_owner_player_number_ = state.ball_owner_player_number;
        stable_owner_is_teammate_ = state.ball_owner_is_teammate;
        pending_possession_.reset();
    } else if (observed == stable_possession_) {
        stable_owner_player_number_ = state.ball_owner_player_number;
        stable_owner_is_teammate_ = state.ball_owner_is_teammate;
        pending_possession_.reset();
    } else {
        bool accept_turnover =
            state.possession_confidence >= parameters_.strong_evidence_confidence;
        if (!accept_turnover) {
            if (!pending_possession_.has_value() ||
                *pending_possession_ != observed) {
                pending_possession_ = observed;
                pending_since_s_ = snapshot.server_time;
            } else if (snapshot.server_time - pending_since_s_ >=
                       parameters_.possession_confirmation_s) {
                accept_turnover = true;
            }
        }
        if (accept_turnover) {
            const bool lost_possession =
                stable_possession_ == PossessionOwner::Ours &&
                observed == PossessionOwner::Theirs;
            stable_possession_ = observed;
            stable_owner_player_number_ = state.ball_owner_player_number;
            stable_owner_is_teammate_ = state.ball_owner_is_teammate;
            pending_possession_.reset();
            if (lost_possession) {
                counter_press_until_s_ = snapshot.server_time +
                    parameters_.counter_press_window_s;
            }
        }
    }

    state.possession = stable_possession_;
    state.ball_owner_player_number = stable_owner_player_number_;
    state.ball_owner_is_teammate = stable_owner_is_teammate_;
    state.phase = phase_for(stable_possession_);
    if (stable_possession_ == PossessionOwner::Theirs &&
        snapshot.server_time < counter_press_until_s_) {
        state.phase = TacticalPhase::Transition;
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
