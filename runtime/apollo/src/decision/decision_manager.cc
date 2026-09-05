// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/decision_manager.h"

namespace decision {

DecisionManager::DecisionManager(
    bool enable_pass_strategy,
    bool enable_targeted_kick,
    bool enable_team_tactics)
    : enable_pass_strategy_(enable_pass_strategy),
      enable_targeted_kick_(enable_targeted_kick),
      enable_team_tactics_(enable_team_tactics) {}

HighLevelCommand DecisionManager::decide(
    const world::WorldSnapshot& snapshot,
    const std::optional<ExecutionFeedback>& execution_feedback) {
    return behavior_tree_.evaluate(
        snapshot, blackboard_, role_manager_, enable_pass_strategy_,
        enable_targeted_kick_, execution_feedback, enable_team_tactics_);
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

const comm::OutgoingPassIntent* DecisionManager::outgoing_pass_intent() const {
    if (!blackboard_.exists(Blackboard::kKeyOutgoingPassIntent)) return nullptr;
    return &blackboard_.get<comm::OutgoingPassIntent>(
        Blackboard::kKeyOutgoingPassIntent);
}

const Blackboard& DecisionManager::blackboard() const {
    return blackboard_;
}

}  // namespace decision
