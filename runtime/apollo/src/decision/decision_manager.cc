// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/decision_manager.h"

namespace decision {

DecisionManager::DecisionManager(bool enable_pass_strategy)
    : enable_pass_strategy_(enable_pass_strategy) {}

HighLevelCommand DecisionManager::decide(const world::WorldSnapshot& snapshot) {
    return behavior_tree_.evaluate(
        snapshot, blackboard_, role_manager_, enable_pass_strategy_);
}

const strategy::PlanningResult* DecisionManager::strategy_plan() const {
    if (!blackboard_.exists(Blackboard::kKeyStrategyPlan)) return nullptr;
    return &blackboard_.get<strategy::PlanningResult>(Blackboard::kKeyStrategyPlan);
}

const strategy::CooperativeAction* DecisionManager::selected_cooperative_action() const {
    if (!blackboard_.exists(Blackboard::kKeySelectedCooperativeAction)) return nullptr;
    return &blackboard_.get<strategy::CooperativeAction>(
        Blackboard::kKeySelectedCooperativeAction);
}

const Blackboard& DecisionManager::blackboard() const {
    return blackboard_;
}

}  // namespace decision
