// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/behavior_tree.h"
#include "src/decision/role_manager.h"
#include "src/math/math_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <variant>

namespace {

world::PlayerObservation teammate(int number, double x, double y) {
    world::PlayerObservation player;
    player.player_number = number;
    player.is_teammate = true;
    player.seen = true;
    player.last_seen_time = 10.0;
    player.position_m = {x, y, 0.8};
    return player;
}

world::PlayerObservation opponent(int number, double x, double y) {
    world::PlayerObservation player;
    player.player_number = number;
    player.is_teammate = false;
    player.seen = true;
    player.last_seen_time = 10.0;
    player.position_m = {x, y, 0.8};
    return player;
}

world::WorldSnapshot full_team_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.player_number = 7;
    snapshot.server_time = 10.0;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.has_beamed = true;
    snapshot.self.position_m = {5.0, 0.0, 0.8};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_m = {0.0, 0.0, 0.11};
    snapshot.teammates = {
        teammate(1, -27.0, 0.0),
        teammate(2, -12.0, 5.0),
        teammate(3, 1.5, 0.0),
        teammate(4, -8.0, 0.0),
        teammate(5, -2.0, 5.0),
        teammate(6, 3.0, -3.0),
        teammate(7, 5.0, 0.0),
    };
    return snapshot;
}

int player_for_role(
    const std::vector<decision::RoleAssignment>& assignments,
    int role) {
    for (const auto& assignment : assignments) {
        if (assignment.role_id == role) return assignment.player_number;
    }
    return 0;
}

bool roles_unique(const std::vector<decision::RoleAssignment>& assignments) {
    std::set<int> roles;
    for (const auto& assignment : assignments) {
        if (assignment.role_id < 0) continue;
        if (!roles.insert(assignment.role_id).second) return false;
    }
    return true;
}

}  // namespace

int main() {
    world::WorldSnapshot snapshot = full_team_snapshot();
    decision::RoleManager role_manager;
    const auto first = role_manager.assign(snapshot);
    const auto second = role_manager.assign(snapshot);
    if (!roles_unique(first) || first.size() != 7U || second.size() != 7U ||
        player_for_role(first, decision::RoleManager::ROLE_GK) != 1 ||
        player_for_role(first, decision::RoleManager::ROLE_AP) != 3 ||
        player_for_role(first, decision::RoleManager::ROLE_AP) !=
            player_for_role(second, decision::RoleManager::ROLE_AP)) {
        std::cerr << "role assignment was not unique and deterministic\n";
        return 1;
    }

    world::WorldSnapshot reordered_snapshot = snapshot;
    std::reverse(
        reordered_snapshot.teammates.begin(),
        reordered_snapshot.teammates.end());
    const auto reordered = decision::RoleManager{}.assign(reordered_snapshot);
    for (int role = decision::RoleManager::ROLE_GK;
         role <= decision::RoleManager::ROLE_AP; ++role) {
        if (player_for_role(first, role) != player_for_role(reordered, role)) {
            std::cerr << "role assignment depended on observation order\n";
            return 1;
        }
    }

    decision::Formation::RolePositions tie_positions{};
    tie_positions[decision::RoleManager::ROLE_CBL] = {-1.0, 0.0};
    tie_positions[decision::RoleManager::ROLE_CBR] = {1.0, 0.0};
    decision::RoleManager::PreviousRoleByPlayer no_previous;
    no_previous.fill(-1);
    const auto comm_tiebreak = decision::assign_remaining_players(
        tie_positions,
        {
            {2, {0.0, 0.0}, false, decision::RoleManager::ROLE_CBR},
            {3, {0.0, 0.0}, false, decision::RoleManager::ROLE_CBL},
        },
        {decision::RoleManager::ROLE_CBL, decision::RoleManager::ROLE_CBR},
        no_previous);
    if (player_for_role(comm_tiebreak, decision::RoleManager::ROLE_CBL) != 3 ||
        player_for_role(comm_tiebreak, decision::RoleManager::ROLE_CBR) != 2) {
        std::cerr << "communicated roles did not stabilize an ambiguous match\n";
        return 1;
    }

    world::WorldSnapshot fallen_goalkeeper = snapshot;
    fallen_goalkeeper.teammates[0].fallen = true;
    const auto replacement = decision::RoleManager{}.assign(fallen_goalkeeper);
    if (player_for_role(replacement, decision::RoleManager::ROLE_GK) != 2 ||
        !roles_unique(replacement)) {
        std::cerr << "fallen goalkeeper did not receive a unique replacement\n";
        return 1;
    }

    world::WorldSnapshot stale = snapshot;
    stale.teammates[1].seen = false;
    stale.teammates[1].last_seen_time = 1.0;
    stale.teammates[1].position_m = {0.0, 0.0, 0.8};
    const auto stale_assignment = decision::RoleManager{}.assign(stale);
    if (player_for_role(stale_assignment, decision::RoleManager::ROLE_AP) == 2) {
        std::cerr << "stale teammate position was used indefinitely for AP\n";
        return 1;
    }

    // A precision ball action gets a rolling AP lease only inside the same
    // near-equal ball race. A clearly closer teammate must pre-empt it rather
    // than walking back to formation beside the ball. The lease also expires
    // quickly and never overrides a fall.
    world::WorldSnapshot leased = snapshot;
    leased.server_time = 10.0;
    leased.teammates[2].position_m = {0.95, 0.0, 0.8};
    leased.teammates[6].position_m = {1.05, 0.0, 0.8};
    leased.self.position_m = leased.teammates[6].position_m;
    decision::RoleManager leased_roles;
    leased_roles.retain_self_as_ap_for_action(7, leased, 0.35);
    const auto during_lease = leased_roles.assign(leased);
    if (player_for_role(during_lease, decision::RoleManager::ROLE_AP) != 7) {
        std::cerr << "committed action did not retain its AP actor\n";
        return 1;
    }
    world::WorldSnapshot clear_challenger = leased;
    clear_challenger.server_time = 10.10;
    clear_challenger.teammates[2].position_m = {0.25, 0.0, 0.8};
    const auto preempted_lease = leased_roles.assign(clear_challenger);
    if (player_for_role(preempted_lease, decision::RoleManager::ROLE_AP) != 3) {
        std::cerr << "action lease blocked a clearly closer ball challenger\n";
        return 1;
    }
    leased.server_time = 10.36;
    const auto after_lease = leased_roles.assign(leased);
    if (player_for_role(after_lease, decision::RoleManager::ROLE_AP) != 3) {
        std::cerr << "expired action lease retained a stale AP actor\n";
        return 1;
    }
    leased.server_time = 11.0;
    leased.teammates[6].fallen = true;
    leased_roles.retain_self_as_ap_for_action(7, leased, 0.35);
    const auto fallen_lease = leased_roles.assign(leased);
    if (player_for_role(fallen_lease, decision::RoleManager::ROLE_AP) == 7) {
        std::cerr << "fallen action actor retained AP through its lease\n";
        return 1;
    }

    // A sideways AP 0.8 m from the ball used to select a side-relocation
    // waypoint equal to its current position, then stand still facing the
    // goal while an opponent arrived. An urgent contest must instead turn on
    // the shortest through-ball line, and the tactics-off baseline must still
    // report the live AP as Pressure rather than Formation.
    world::WorldSnapshot urgent = full_team_snapshot();
    urgent.player_number = 3;
    urgent.self.position_m = {0.0, 0.8, 0.8};
    urgent.teammates[2].position_m = urgent.self.position_m;
    urgent.opponents = {opponent(1, 0.4, 0.0)};
    decision::BehaviorTree urgent_tree;
    decision::Blackboard urgent_blackboard;
    decision::RoleManager urgent_roles;
    const auto urgent_command = urgent_tree.evaluate(
        urgent, urgent_blackboard, urgent_roles,
        true, false, std::nullopt, false);
    const auto* urgent_walk = std::get_if<decision::WalkCommand>(
        &urgent_command);
    const auto& urgent_target = urgent_blackboard.get<
        decision::TacticalTarget>(
            decision::Blackboard::kKeyTacticalTarget);
    if (urgent_blackboard.get<int>(
            decision::Blackboard::kKeyCurrentRole) !=
            decision::RoleManager::ROLE_AP ||
        urgent_target.duty != decision::TacticalDuty::Pressure ||
        urgent_walk == nullptr ||
        !urgent_walk->orientation_deg.has_value() ||
        std::abs(math::normalize_deg(
            *urgent_walk->orientation_deg + 90.0)) > 5.0) {
        std::cerr << "urgent ball-side AP did not take the shortest contest line\n";
        return 1;
    }

    // Disabling experimental field-player duties must not also disable the
    // goalkeeper's tested safety planner. That all-or-nothing coupling left
    // the v43 keeper on Formation during both goal sequences.
    world::WorldSnapshot keeper_safety = full_team_snapshot();
    keeper_safety.player_number = 1;
    keeper_safety.self.position_m = {-26.5, 0.0, 0.8};
    keeper_safety.teammates[0].position_m = keeper_safety.self.position_m;
    keeper_safety.ball.position_m = {-24.5, 0.2, 0.11};
    keeper_safety.ball.velocity_valid = true;
    keeper_safety.ball.velocity_mps = {-1.0, 0.0, 0.0};
    decision::BehaviorTree keeper_safety_tree;
    decision::Blackboard keeper_safety_blackboard;
    decision::RoleManager keeper_safety_roles;
    keeper_safety_tree.evaluate(
        keeper_safety, keeper_safety_blackboard, keeper_safety_roles,
        true, true, std::nullopt, false);
    const auto keeper_safety_duty = keeper_safety_blackboard.get<
        decision::TacticalTarget>(
            decision::Blackboard::kKeyTacticalTarget).duty;
    if (keeper_safety_duty != decision::TacticalDuty::GoalkeeperHold &&
        keeper_safety_duty !=
            decision::TacticalDuty::GoalkeeperIntercept &&
        keeper_safety_duty != decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "stable field tactics disabled goalkeeper safety planning\n";
        return 1;
    }

    snapshot.play_mode = world::PlayMode::GameOver;
    decision::BehaviorTree tree;
    decision::Blackboard blackboard;
    decision::RoleManager game_over_roles;
    const decision::HighLevelCommand stopped = tree.evaluate(
        snapshot, blackboard, game_over_roles, true, true);
    if (!std::holds_alternative<decision::NeutralCommand>(stopped)) {
        std::cerr << "GameOver did not stop team behavior\n";
        return 1;
    }

    // V40 side-swap regression: with dynamic duties disabled, both centre-
    // back formation slots followed a deep ball into the goalkeeper area.
    // The goalkeeper plus both centre-backs can exceed the server's two-player
    // limit. CBL owns the one field-player allowance, so every other role must
    // remain beyond the boundary even during a deep formation shift.
    world::WorldSnapshot deep_defense = full_team_snapshot();
    deep_defense.player_number = 6;
    deep_defense.self.position_m = {-19.7, 2.6, 0.8};
    deep_defense.teammates[5].position_m = deep_defense.self.position_m;
    deep_defense.ball.position_m = {-22.8, -0.7, 0.11};
    decision::BehaviorTree stable_shape_tree;
    decision::Blackboard stable_shape_blackboard;
    decision::RoleManager stable_shape_roles;
    stable_shape_tree.evaluate(
        deep_defense, stable_shape_blackboard, stable_shape_roles,
        true, true, std::nullopt, false);
    const auto& stable_shape_target = stable_shape_blackboard.get<
        decision::TacticalTarget>(
            decision::Blackboard::kKeyTacticalTarget);
    const int stable_shape_role = stable_shape_blackboard.get<int>(
        decision::Blackboard::kKeyCurrentRole);
    const decision::TacticalDuty expected_deep_duty =
        stable_shape_role == decision::RoleManager::ROLE_AP
        ? decision::TacticalDuty::Pressure
        : decision::TacticalDuty::Formation;
    if (stable_shape_role == decision::RoleManager::ROLE_GK ||
        stable_shape_role == decision::RoleManager::ROLE_CBL ||
        stable_shape_target.duty != expected_deep_duty ||
        stable_shape_target.position_m[0] <
            -decision::field_geometry::kActualHalfLengthM +
                decision::field_geometry::kGoalieAreaDepthM +
                decision::field_geometry::kFieldPlayerGoalieAreaClearanceM -
                1.0e-9) {
        std::cerr << "stable-shape field player entered our goalkeeper area"
                  << " role=" << stable_shape_role
                  << " target_x=" << stable_shape_target.position_m[0]
                  << '\n';
        return 1;
    }

    struct RestartFixture {
        world::PlayMode mode;
        std::array<double, 2> ball;
        bool requires_clearance;
    };
    const std::array<RestartFixture, 11> their_restarts{{
        {world::PlayMode::TheirKickOff, {0.0, 0.0}, true},
        {world::PlayMode::TheirThrowIn, {3.0, 18.0}, true},
        {world::PlayMode::TheirThrowIn, {-4.0, -18.0}, true},
        {world::PlayMode::TheirCornerKick, {-27.5, 18.0}, true},
        {world::PlayMode::TheirCornerKick, {-27.5, -18.0}, true},
        {world::PlayMode::TheirGoalKick, {23.0, 0.0}, true},
        {world::PlayMode::TheirOffside, {8.0, -4.0}, true},
        {world::PlayMode::TheirFreeKick, {-6.0, 7.0}, true},
        {world::PlayMode::TheirDirectFreeKick, {-12.0, -5.0}, true},
        {world::PlayMode::TheirPenaltyKick, {-20.0, 0.0}, true},
        {world::PlayMode::TheirPenaltyShoot, {-20.0, 0.0}, true},
    }};
    for (const auto& fixture : their_restarts) {
        world::WorldSnapshot restart = full_team_snapshot();
        restart.play_mode = fixture.mode;
        restart.play_mode_group = world::PlayModeGroup::TheirKick;
        restart.ball.position_m = {
            fixture.ball[0], fixture.ball[1], 0.11};
        decision::BehaviorTree restart_tree;
        decision::Blackboard restart_blackboard;
        decision::RoleManager restart_roles;
        const auto command = restart_tree.evaluate(
            restart, restart_blackboard, restart_roles, true, true);
        if (std::holds_alternative<decision::KickCommand>(command)) {
            std::cerr << "opponent restart emitted a kick command\n";
            return 1;
        }
        if (const auto* walk = std::get_if<decision::WalkCommand>(&command);
            walk != nullptr && walk->target_absolute) {
            if (!std::isfinite(walk->target_2d_m[0]) ||
                !std::isfinite(walk->target_2d_m[1]) ||
                std::abs(walk->target_2d_m[0]) >
                    decision::field_geometry::kActualHalfLengthM ||
                std::abs(walk->target_2d_m[1]) >
                    decision::field_geometry::kActualHalfWidthM ||
                (fixture.requires_clearance &&
                 std::hypot(
                     walk->target_2d_m[0] - fixture.ball[0],
                     walk->target_2d_m[1] - fixture.ball[1]) < 5.5)) {
                std::cerr << "opponent restart target was illegal\n";
                return 1;
            }
        }
    }

    const std::array<RestartFixture, 10> our_restarts{{
        {world::PlayMode::OurKickOff, {0.0, 0.0}, false},
        {world::PlayMode::OurThrowIn, {3.0, 18.0}, false},
        {world::PlayMode::OurCornerKick, {27.5, 18.0}, false},
        {world::PlayMode::OurCornerKick, {27.5, -18.0}, false},
        {world::PlayMode::OurGoalKick, {-23.0, 0.0}, false},
        {world::PlayMode::OurOffside, {-4.0, 3.0}, false},
        {world::PlayMode::OurFreeKick, {5.0, -7.0}, false},
        {world::PlayMode::OurDirectFreeKick, {12.0, 5.0}, false},
        {world::PlayMode::OurPenaltyKick, {20.0, 0.0}, false},
        {world::PlayMode::OurPenaltyShoot, {20.0, 0.0}, false},
    }};
    for (const auto& fixture : our_restarts) {
        world::WorldSnapshot restart = full_team_snapshot();
        restart.player_number = 2;
        restart.self.position_m = {-12.0, 5.0, 0.8};
        restart.teammates[1].position_m = restart.self.position_m;
        restart.play_mode = fixture.mode;
        restart.play_mode_group = world::PlayModeGroup::OurKick;
        restart.ball.position_m = {
            fixture.ball[0], fixture.ball[1], 0.11};
        decision::BehaviorTree restart_tree;
        decision::Blackboard restart_blackboard;
        decision::RoleManager restart_roles;
        const auto command = restart_tree.evaluate(
            restart, restart_blackboard, restart_roles, true, true);
        if (std::holds_alternative<decision::KickCommand>(command)) {
            std::cerr << "non-taker emitted a restart kick\n";
            return 1;
        }
    }

    // Reproduce the v20 goal-kick approach: the ball is mostly lateral to a
    // keeper facing downfield.  Long-range setup must first face the actual
    // approach waypoint, not demand final kick yaw while translating sideways.
    world::WorldSnapshot lateral_goal_kick = full_team_snapshot();
    lateral_goal_kick.player_number = 1;
    lateral_goal_kick.play_mode = world::PlayMode::OurGoalKick;
    lateral_goal_kick.play_mode_group = world::PlayModeGroup::OurKick;
    lateral_goal_kick.ball.position_m = {-27.5043, -3.20417, 0.11};
    lateral_goal_kick.self.position_m = {-27.005, -1.11, 0.8};
    constexpr double kYawMinus86HalfRadians = -0.7504915783575616;
    lateral_goal_kick.self.orientation_wxyz = {
        std::cos(kYawMinus86HalfRadians), 0.0, 0.0,
        std::sin(kYawMinus86HalfRadians)};
    lateral_goal_kick.teammates[0].position_m =
        lateral_goal_kick.self.position_m;
    decision::BehaviorTree lateral_goal_kick_tree;
    decision::Blackboard lateral_goal_kick_blackboard;
    decision::RoleManager lateral_goal_kick_roles;
    const auto lateral_goal_kick_command = lateral_goal_kick_tree.evaluate(
        lateral_goal_kick, lateral_goal_kick_blackboard,
        lateral_goal_kick_roles, true, true);
    const auto* lateral_goal_kick_walk =
        std::get_if<decision::WalkCommand>(&lateral_goal_kick_command);
    if (lateral_goal_kick_walk == nullptr ||
        !lateral_goal_kick_walk->orientation_deg.has_value() ||
        *lateral_goal_kick_walk->orientation_deg > -70.0) {
        std::cerr << "lateral goal-kick approach did not face its travel waypoint\n";
        return 1;
    }

    world::WorldSnapshot goal_kick = full_team_snapshot();
    goal_kick.player_number = 1;
    goal_kick.play_mode = world::PlayMode::OurGoalKick;
    goal_kick.play_mode_group = world::PlayModeGroup::OurKick;
    goal_kick.ball.position_m = {-23.0, 0.0, 0.11};
    goal_kick.self.position_m = {-23.33, 0.0, 0.8};
    goal_kick.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    goal_kick.teammates[0].position_m = goal_kick.self.position_m;
    decision::BehaviorTree goal_kick_tree;
    decision::Blackboard goal_kick_blackboard;
    decision::RoleManager goal_kick_roles;
    const auto stabilizing = goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false);
    if (std::holds_alternative<decision::KickCommand>(stabilizing)) {
        std::cerr << "goal kick skipped the stable setup hold\n";
        return 1;
    }
    const auto& frozen_restart = goal_kick_blackboard.get<
        decision::RestartCoordinationDecision>(
            decision::Blackboard::kKeyRestartDecision);
    if (!frozen_restart.plan.has_value()) {
        std::cerr << "goal kick did not publish a frozen restart plan\n";
        return 1;
    }
    const double goal_kick_direction_rad = math::deg_to_rad(
        frozen_restart.plan->contact_direction_deg);
    goal_kick.self.position_m = {
        goal_kick.ball.position_m[0] -
            0.33 * std::cos(goal_kick_direction_rad),
        goal_kick.ball.position_m[1] -
            0.33 * std::sin(goal_kick_direction_rad),
        0.8};
    goal_kick.self.orientation_wxyz = {
        std::cos(goal_kick_direction_rad * 0.5),
        0.0,
        0.0,
        std::sin(goal_kick_direction_rad * 0.5)};
    goal_kick.teammates[0].position_m = goal_kick.self.position_m;
    goal_kick.server_time += 0.10;
    // A stationary restart ball is commonly hidden by the taker's torso.  The
    // frozen, initially observed anchor remains actionable while the server is
    // still in OurKick; losing live vision must not strand the restart.
    goal_kick.ball.position_valid = false;
    const auto aligned_hold = goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false);
    if (std::holds_alternative<decision::KickCommand>(aligned_hold)) {
        std::cerr << "goal kick skipped the aligned stable hold\n";
        return 1;
    }
    goal_kick.server_time += 0.30;
    const auto clearance = goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false);
    const auto* kick = std::get_if<decision::KickCommand>(&clearance);
    if (kick == nullptr || kick->mode != decision::KickMode::ForwardContact ||
        !kick->restart_epoch.has_value() ||
        !kick->restart_revision.has_value()) {
        std::cerr << "goalkeeper did not execute the available goal-kick contact\n";
        return 1;
    }

    decision::ExecutionFeedback completed;
    completed.request_id = 1U;
    completed.server_time = goal_kick.server_time;
    completed.status = decision::ExecutionStatus::Completed;
    completed.request_kind = decision::MotionRequestKind::Kick;
    completed.restart_epoch = kick->restart_epoch;
    completed.restart_revision = kick->restart_revision;
    goal_kick.server_time += 0.10;
    const auto verifying = goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false,
        completed);
    if (std::holds_alternative<decision::KickCommand>(verifying)) {
        std::cerr << "completed restart contact was repeated during verification\n";
        return 1;
    }
    goal_kick.ball.position_m[0] += 0.5;
    goal_kick.ball.velocity_valid = true;
    goal_kick.ball.velocity_mps = {0.8, 0.0, 0.0};
    goal_kick.server_time += 0.10;
    goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false);
    goal_kick.server_time += 0.10;
    goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false);
    goal_kick.play_mode = world::PlayMode::PlayOn;
    goal_kick.play_mode_group = world::PlayModeGroup::Other;
    goal_kick.server_time += 0.10;
    const auto locked_out = goal_kick_tree.evaluate(
        goal_kick, goal_kick_blackboard, goal_kick_roles, true, false);
    if (std::holds_alternative<decision::KickCommand>(locked_out) ||
        !goal_kick_blackboard.exists(
            decision::Blackboard::kKeyRestartDecision) ||
        !goal_kick_blackboard.get<decision::RestartCoordinationDecision>(
            decision::Blackboard::kKeyRestartDecision).self_locked_out) {
        std::cerr << "restart taker was not locked out after release\n";
        return 1;
    }
    return 0;
}
