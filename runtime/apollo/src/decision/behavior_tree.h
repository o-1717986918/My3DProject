// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/decision/blackboard.h"
#include "src/decision/execution_feedback.h"
#include "src/decision/high_level_command.h"
#include "src/decision/restart_coordinator.h"
#include "src/decision/role_behaviors.h"
#include "src/decision/role_manager.h"
#include "src/decision/team_tactics.h"
#include "src/world/world_snapshot.h"

#include <optional>

namespace decision {

/// Evaluates match state, set plays, and role behavior into one command.
class BehaviorTree {
public:
    HighLevelCommand evaluate(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard,
        RoleManager& role_manager,
        bool enable_pass_strategy,
        bool enable_targeted_kick = false,
        const std::optional<ExecutionFeedback>& execution_feedback =
            std::nullopt,
        bool enable_team_tactics = true) const;

private:
    mutable TeamTactics team_tactics_;
    mutable RestartCoordinator restart_coordinator_;
    mutable RoleBehaviorSet role_behaviors_;
    mutable world::PlayMode previous_play_mode_{
        world::PlayMode::NotInitialized};
    mutable double kickoff_hold_until_s_{-1.0};
};

}  // namespace decision
