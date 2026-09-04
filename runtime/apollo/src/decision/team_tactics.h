// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/decision/field_geometry.h"
#include "src/decision/role_manager.h"
#include "src/world/world_snapshot.h"

#include <array>
#include <optional>
#include <string_view>
#include <vector>

namespace decision {

enum class TacticalDuty {
    Formation,
    Support,
    Unmark,
    Outlet,
    Pressure,
    Cover,
    Mark,
    BlockLane,
    Intercept,
    GoalkeeperHold,
    GoalkeeperIntercept,
    Receive,
};

struct TacticalTarget {
    TacticalDuty duty{TacticalDuty::Formation};
    field_geometry::Position2 position_m{0.0, 0.0};
    std::optional<field_geometry::Position2> face_point_m;
    int marked_opponent_player_number{0};
    double confidence{0.0};
};

struct TeamTacticalAssignment {
    int player_number{0};
    int role_id{-1};
    TacticalTarget target;
};

/// One deterministic tactical view for the complete seven-player assignment.
/// Every robot builds this from the same role and world inputs, then consumes
/// only its own entry.
struct TeamPlan {
    std::vector<TeamTacticalAssignment> assignments;

    const TeamTacticalAssignment* for_player(int player_number) const;
    const TeamTacticalAssignment* for_role(int role_id) const;
};

/// Converts shared world state and a stable formation slot into an executable
/// off-ball target. It owns no motion primitives: every result is consumed by
/// the existing bounded walk/turn path.
class TeamTactics {
public:
    TeamPlan plan_all(
        const world::WorldSnapshot& snapshot,
        const std::vector<RoleAssignment>& role_assignments) const;

    /// Compatibility entry point for isolated tests and tools. Runtime code
    /// should use plan_all() so ownership decisions remain team-consistent.
    TacticalTarget plan(
        const world::WorldSnapshot& snapshot,
        int role_id,
        const field_geometry::Position2& formation_target_m) const;

private:
    struct SupportLatch {
        std::optional<TacticalTarget> target;
        double until_s{-1.0};
        int role_id{-1};
    };
    mutable std::array<SupportLatch, RoleManager::kPreviousRoleSlots>
        support_latches_{};
};

std::string_view to_string(TacticalDuty duty);

}  // namespace decision
