// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/decision/blackboard.h"
#include "src/decision/execution_feedback.h"
#include "src/decision/high_level_command.h"
#include "src/decision/pass_lifecycle.h"
#include "src/decision/restart_coordinator.h"
#include "src/decision/role_manager.h"
#include "src/decision/team_tactics.h"
#include "src/strategy/action_planner.h"
#include "src/world/world_snapshot.h"

#include <optional>

namespace decision {

/// Interface for role-specific command generation.
class RoleBehavior {
public:
    virtual ~RoleBehavior() = default;
    virtual bool matches(const Blackboard& blackboard) const = 0;
    virtual HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard) const = 0;
};

/// Persistent attacker state carried between decision cycles.
struct APState {
    bool dribble_ready{false};
    bool set_play_released{false};
    double previous_ball_distance{0.0};
    double kick_active_until_s{0.0};
    double next_kick_allowed_s{0.0};
    double kick_setup_stable_since_s{0.0};
    double kick_setup_started_s{0.0};
    double kick_setup_last_update_s{0.0};
    double kick_setup_direction_deg{0.0};
    double pass_commit_until_s{0.0};
    double pass_retry_after_s{0.0};
    int last_kick_setup_gate{-1};
    std::uint8_t next_pass_sequence_id{0U};
    std::optional<strategy::CooperativeAction> committed_pass;
    std::optional<KickCommand> active_kick_command;
    PassLifecycle pass_lifecycle;
};

/// Generates the active-player command and set-play handoff state.
class APBehavior final {
public:
    bool matches(const Blackboard& blackboard) const;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard,
        RoleManager& role_manager,
        bool enable_pass_strategy,
        bool enable_targeted_kick = false) const;
    void apply_execution_feedback(const ExecutionFeedback& feedback) const;
    void reset_state() const { state_ = {}; }
private:
    mutable APState state_;
    strategy::ActionPlanner action_planner_;
};

/// Walk-to-formation behavior shared by CBM, ST, CBL, CBR, and CDM.
///
/// The defensive variant clips the planner against opponents already past the
/// current ball line.
class SimpleRoleBehavior final : public RoleBehavior {
public:
    SimpleRoleBehavior(int role_id, bool defensive_opponent_clip)
        : role_id_(role_id), defensive_opponent_clip_(defensive_opponent_clip) {}
    bool matches(const Blackboard& blackboard) const override;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard) const override;
    void reset_state() const;
private:
    int role_id_;
    bool defensive_opponent_clip_;
    mutable APState relay_state_;
    mutable std::optional<comm::PassIntentRecord> receive_intent_;
    mutable double receive_intent_until_s_{0.0};
};

/// Generates goalkeeper positioning and goal-kick commands.
class GKBehavior final : public RoleBehavior {
public:
    bool matches(const Blackboard& blackboard) const override;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard) const override;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard,
        bool enable_targeted_kick) const;
    void apply_execution_feedback(const ExecutionFeedback& feedback) const;
    void reset_state() const { clearance_state_ = {}; }
private:
    mutable APState clearance_state_;
    strategy::ActionPlanner action_planner_;
};

/// Per-agent owner for every persistent role behavior. Keeping this object in
/// DecisionManager prevents cooldown, pass, and restart state from leaking
/// between agents or independent replay tests in one process.
class RoleBehaviorSet {
public:
    std::optional<HighLevelCommand> select(
        const world::WorldSnapshot& snapshot,
        Blackboard& blackboard,
        RoleManager& role_manager,
        bool enable_pass_strategy,
        bool enable_targeted_kick = false) const;
    void reset() const;
    void apply_execution_feedback(const ExecutionFeedback& feedback) const;

private:
    APBehavior ap_;
    SimpleRoleBehavior cbm_{RoleManager::ROLE_CBM, false};
    SimpleRoleBehavior st_{RoleManager::ROLE_ST, false};
    SimpleRoleBehavior cbl_{RoleManager::ROLE_CBL, true};
    SimpleRoleBehavior cbr_{RoleManager::ROLE_CBR, true};
    SimpleRoleBehavior cdm_{RoleManager::ROLE_CDM, false};
    GKBehavior gk_;
};

int current_role_from_blackboard(const Blackboard& blackboard);

}  // namespace decision
