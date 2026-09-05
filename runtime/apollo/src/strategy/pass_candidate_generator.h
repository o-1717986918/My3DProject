// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/decision/kick_contract.h"
#include "src/strategy/ball_trajectory_model.h"
#include "src/strategy/cooperative_action.h"
#include "src/strategy/reach_time_model.h"
#include "src/world/world_snapshot.h"

#include <vector>

namespace strategy {

struct CandidateGenerationResult {
    std::vector<CooperativeAction> candidates;
    std::vector<RejectedCandidate> rejections;
};

class PassCandidateGenerator {
public:
    struct Parameters {
        double minimum_pass_distance_m{1.5};
        double maximum_pass_distance_m{
            decision::kick_contract::kParameterizedPassMaximumTargetDistanceM};
        double field_margin_m{1.0};
        double minimum_receiver_lead_s{0.10};
        double minimum_interception_margin_s{0.35};
        double leading_offset_m{1.0};
        double dangerous_backpass_x_m{-20.0};
        double maximum_observation_age_s{1.0};
        bool enable_leading_pass{true};
    };

    PassCandidateGenerator();
    explicit PassCandidateGenerator(Parameters parameters);

    CandidateGenerationResult generate(const world::WorldSnapshot& snapshot) const;

private:
    Parameters parameters_;
    BallTrajectoryModel ball_model_;
    ReachTimeModel receiver_model_;
    ReachTimeModel opponent_model_;

    RejectionReason evaluate_candidate(
        const world::WorldSnapshot& snapshot,
        int receiver_player_number,
        const Position2& receiver_position_m,
        bool receiver_fallen,
        PassType pass_type,
        const Position2& target_point_m,
        CooperativeAction* output) const;
    double earliest_opponent_time_s(
        const world::WorldSnapshot& snapshot,
        const Position2& ball_position_m,
        const Position2& target_point_m,
        double initial_ball_speed_mps,
        double* minimum_margin_s) const;
};

}  // namespace strategy
