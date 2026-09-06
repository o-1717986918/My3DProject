// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/decision/field_geometry.h"
#include "src/decision/role_manager.h"
#include "src/strategy/tactical_state.h"
#include "src/world/world_snapshot.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace decision {

enum class TacticalDuty {
    Formation,
    SearchBall,
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
    GoalkeeperSmother,
    Receive,
};

// The world model deliberately expires ordinary ball coordinates after 0.20 s
// for contact, pass, and shot decisions.  Movement may use the last observed
// point for a little longer: this preserves pressure across a short occlusion
// without promoting the estimate back to an executable ball state.  The world
// state retains a genuine last-known point for longer, so this bound remains a
// deliberately short ordinary-search policy rather than a storage limit.
inline constexpr double kLostBallSearchLifetimeS = 1.5;
// When the last genuine observation was inside our final eight metres, losing
// all vision is itself defensive evidence. Keep exactly one field player at
// the legal goalkeeper-area boundary while uncertainty grows; no stale-ball
// action is authorized by this longer movement-only lifetime.
inline constexpr double kDefensiveLostBallSearchLifetimeS =
    world::kLostBallPositionMemoryLifetimeS;

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
    strategy::TacticalState tactical_state;
    double source_server_time_s{0.0};
    std::uint64_t revision{0U};
    bool fresh{false};

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

    void reset() const;

private:
    struct SupportLatch {
        std::optional<TacticalTarget> target;
        double until_s{-1.0};
        int role_id{-1};
    };
    mutable std::array<SupportLatch, RoleManager::kPreviousRoleSlots>
        support_latches_{};
    mutable strategy::TacticalStateTracker tactical_state_tracker_;
    mutable double last_plan_server_time_s_{-1.0};
};

std::string_view to_string(TacticalDuty duty);

}  // namespace decision
