// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/decision_manager.h"

#include "src/decision/role_behaviors.h"

namespace decision {

DecisionManager::DecisionManager(
    bool enable_goalkeeper_intercept,
    bool enable_discrete_ball_action,
    bool enable_turn_first)
    : behavior_tree_(enable_goalkeeper_intercept) {
    configure_candidate_action_features(
        enable_discrete_ball_action, enable_turn_first,
        enable_goalkeeper_intercept);
}

HighLevelCommand DecisionManager::decide(const world::WorldSnapshot& snapshot) {
    return behavior_tree_.evaluate(snapshot, blackboard_, role_manager_);
}

const Blackboard& DecisionManager::blackboard() const {
    return blackboard_;
}

}  // namespace decision
