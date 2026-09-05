// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/strategy/cooperative_action.h"
#include "src/strategy/tactical_state.h"
#include "src/world/world_snapshot.h"

namespace strategy {

class FieldEvaluator {
public:
    struct Weights {
        double forward_progress{1.8};
        double interception_margin{2.0};
        double goal_proximity{0.20};
        double possession_confidence{0.6};
        double pass_distance{-0.08};
        double leading_pass_cost{-0.35};
        double pressure_release_bonus{1.0};
        double boundary_risk{-1.2};
        double back_pass_cost{-1.0};
        double dribble_bias{0.45};
        double dribble_pressure_cost{-1.5};
        double shot_bias{10.0};
        double clear_bias{8.0};
        double defensive_clear_urgency{8.0};
        double protect_lead_clear_bonus{2.0};
        double chase_goal_shot_bonus{2.0};
        double hold_bias{1.0};
        double move_bias{0.4};
        double move_ball_distance{0.75};
    };

    FieldEvaluator();
    explicit FieldEvaluator(Weights weights);

    double evaluate(
        const CooperativeAction& action,
        const world::WorldSnapshot& snapshot,
        const TacticalState& tactical_state) const;

private:
    Weights weights_;
};

}  // namespace strategy
