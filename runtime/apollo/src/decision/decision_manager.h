// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/comm/team_comm_types.h"
#include "src/decision/behavior_tree.h"
#include "src/decision/blackboard.h"
#include "src/decision/execution_feedback.h"
#include "src/decision/high_level_command.h"
#include "src/decision/role_manager.h"
#include "src/strategy/action_planner.h"
#include "src/world/world_snapshot.h"

#include <optional>

namespace decision {

/// Owns persistent decision state and evaluates one command per world snapshot.
class DecisionManager {
public:
    explicit DecisionManager(
        bool enable_pass_strategy = true,
        bool enable_targeted_kick = false);

    HighLevelCommand decide(
        const world::WorldSnapshot& snapshot,
        const std::optional<ExecutionFeedback>& execution_feedback =
            std::nullopt);
    const Blackboard& blackboard() const;
    const strategy::PlanningResult* strategy_plan() const;
    const strategy::CooperativeAction* selected_cooperative_action() const;
    const comm::OutgoingPassIntent* outgoing_pass_intent() const;

private:
    Blackboard blackboard_;
    RoleManager role_manager_;
    BehaviorTree behavior_tree_;
    bool enable_pass_strategy_{true};
    bool enable_targeted_kick_{false};
};

}  // namespace decision
