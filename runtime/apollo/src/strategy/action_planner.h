// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/strategy/action_capability.h"
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
        double minimum_action_utility{1.0};
    };

    ActionPlanner();
    explicit ActionPlanner(Parameters parameters);

    /// Convenience entry point used by offline tools: all currently deployed
    /// parameterized capabilities are considered available.
    PlanningResult plan(const world::WorldSnapshot& snapshot) const;
    /// Generate and compare every supported open-play ball action. A selected
    /// action may still require bounded approach/alignment before it becomes
    /// immediately executable at the motion layer.
    PlanningResult plan(
        const world::WorldSnapshot& snapshot,
        const ActionCapabilityRegistry& capabilities,
        bool enable_passes) const;
    PlanningResult plan(
        const world::WorldSnapshot& snapshot,
        const ActionCapabilityRegistry& capabilities,
        bool enable_passes,
        const TacticalState& tactical_state) const;

private:
    Parameters parameters_;
    PassCandidateGenerator pass_generator_;
    FieldEvaluator field_evaluator_;
};

}  // namespace strategy
