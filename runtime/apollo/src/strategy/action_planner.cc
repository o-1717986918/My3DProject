// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/action_planner.h"

#include <algorithm>
#include <cmath>

namespace strategy {

ActionPlanner::ActionPlanner() = default;

ActionPlanner::ActionPlanner(Parameters parameters)
    : parameters_(parameters) {}

PlanningResult ActionPlanner::plan(const world::WorldSnapshot& snapshot) const {
    PlanningResult result;
    result.tactical_state = build_tactical_state(snapshot);
    CandidateGenerationResult generated = pass_generator_.generate(snapshot);
    result.rejections = std::move(generated.rejections);
    result.candidates = std::move(generated.candidates);

    for (auto& candidate : result.candidates) {
        candidate.utility = field_evaluator_.evaluate(
            candidate, snapshot, result.tactical_state);
    }
    std::sort(
        result.candidates.begin(), result.candidates.end(),
        [](const CooperativeAction& lhs, const CooperativeAction& rhs) {
            if (std::abs(lhs.utility - rhs.utility) > 1.0e-9) {
                return lhs.utility > rhs.utility;
            }
            if (lhs.target_player_number != rhs.target_player_number) {
                return lhs.target_player_number < rhs.target_player_number;
            }
            if (lhs.pass_type != rhs.pass_type) {
                return lhs.pass_type < rhs.pass_type;
            }
            return lhs.action_id < rhs.action_id;
        });

    if (!result.candidates.empty() &&
        result.candidates.front().utility >= parameters_.minimum_pass_utility) {
        result.selected = result.candidates.front();
    } else if (!result.candidates.empty()) {
        const auto& best = result.candidates.front();
        result.rejections.push_back({
            best.pass_type, best.target_player_number, best.target_point_m,
            RejectionReason::BelowUtilityFloor});
    }
    return result;
}

}  // namespace strategy
