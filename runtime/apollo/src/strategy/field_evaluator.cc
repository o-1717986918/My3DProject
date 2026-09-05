// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/field_evaluator.h"

#include "src/math/math_utils.h"
#include "src/server/server_constants.h"

#include <algorithm>
#include <cmath>

namespace strategy {

FieldEvaluator::FieldEvaluator() = default;

FieldEvaluator::FieldEvaluator(Weights weights)
    : weights_(weights) {}

double FieldEvaluator::evaluate(
    const CooperativeAction& action,
    const world::WorldSnapshot& snapshot,
    const TacticalState& tactical_state) const {
    const Position2 ball{snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const Position2 self{
        snapshot.self.position_m[0], snapshot.self.position_m[1]};
    if (action.category == ActionCategory::Hold) {
        return weights_.hold_bias;
    }
    if (action.category == ActionCategory::Move) {
        return weights_.move_bias + weights_.move_ball_distance *
            std::clamp(math::planar_dist(self, ball), 0.0, 10.0);
    }
    const Position2 goal{server_constants::kFieldHalfLengthM, 0.0};
    const double forward = action.target_point_m[0] - ball[0];
    const double distance = math::planar_dist(ball, action.target_point_m);
    const double old_goal_distance = math::planar_dist(ball, goal);
    const double new_goal_distance = math::planar_dist(action.target_point_m, goal);
    const double goal_gain = old_goal_distance - new_goal_distance;
    const double edge_distance = std::min(
        server_constants::kFieldHalfLengthM - std::abs(action.target_point_m[0]),
        server_constants::kFieldHalfWidthM - std::abs(action.target_point_m[1]));
    const double boundary_risk = std::clamp((1.5 - edge_distance) / 1.5, 0.0, 1.0);
    const double pressure_bonus =
        std::isfinite(tactical_state.nearest_opponent_ball_distance_m) &&
        tactical_state.nearest_opponent_ball_distance_m < 2.5
            ? 1.0
            : 0.0;

    double utility =
        weights_.forward_progress * forward +
        weights_.interception_margin *
            std::clamp(action.interception_margin_s, -2.0, 3.0) +
        weights_.goal_proximity * goal_gain +
        weights_.possession_confidence * tactical_state.possession_confidence +
        weights_.pass_distance * distance +
        weights_.pressure_release_bonus * pressure_bonus +
        weights_.boundary_risk * boundary_risk;

    if (action.pass_type == PassType::Leading) {
        utility += weights_.leading_pass_cost;
    }
    if (forward < -0.5) {
        utility += weights_.back_pass_cost * std::abs(forward);
    }
    switch (action.category) {
        case ActionCategory::Dribble:
            utility += weights_.dribble_bias;
            utility += weights_.dribble_pressure_cost * pressure_bonus;
            break;
        case ActionCategory::Shoot:
            utility += weights_.shot_bias;
            if (tactical_state.risk_mode == TacticalRiskMode::ChaseGoal) {
                utility += weights_.chase_goal_shot_bonus;
            }
            break;
        case ActionCategory::Clear: {
            const double defensive_line =
                -server_constants::kFieldHalfLengthM + 10.0;
            const double defensive_urgency = std::clamp(
                (defensive_line - ball[0]) / 10.0, 0.0, 1.0);
            utility += weights_.clear_bias +
                weights_.defensive_clear_urgency * defensive_urgency;
            if (tactical_state.risk_mode == TacticalRiskMode::ProtectLead) {
                utility += weights_.protect_lead_clear_bonus;
            }
            break;
        }
        case ActionCategory::Pass:
        case ActionCategory::Hold:
        case ActionCategory::Move:
        case ActionCategory::NoAction:
            break;
    }
    return utility;
}

}  // namespace strategy
