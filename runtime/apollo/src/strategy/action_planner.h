// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/strategy/cooperative_action.h"
#include "src/strategy/field_evaluator.h"
#include "src/strategy/pass_candidate_generator.h"
#include "src/strategy/tactical_state.h"

#include <optional>
#include <vector>

namespace strategy {

struct PlanningResult {
    TacticalState tactical_state;
    std::vector<CooperativeAction> candidates;
    std::vector<RejectedCandidate> rejections;
    std::optional<CooperativeAction> selected;
};

class ActionPlanner {
public:
    struct Parameters {
        double minimum_pass_utility{1.0};
    };

    ActionPlanner();
    explicit ActionPlanner(Parameters parameters);

    PlanningResult plan(const world::WorldSnapshot& snapshot) const;

private:
    Parameters parameters_;
    PassCandidateGenerator pass_generator_;
    FieldEvaluator field_evaluator_;
};

}  // namespace strategy
