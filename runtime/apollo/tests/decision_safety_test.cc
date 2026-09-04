// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/behavior_tree.h"
#include "src/decision/role_manager.h"

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
