// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/role_manager.h"
#include "src/decision/team_tactics.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

world::PlayerObservation player(
    int number,
    double x,
    double y,
    bool teammate) {
    world::PlayerObservation result;
    result.player_number = number;
    result.is_teammate = teammate;
    result.seen = true;
    result.last_seen_time = 10.0;
    result.position_m = {x, y, 0.8};
    return result;
}

world::WorldSnapshot base_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.player_number = 6;
    snapshot.server_time = 10.0;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {2.0, 0.0, 0.8};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_m = {0.0, 0.0, 0.11};
    snapshot.teammates = {
        player(6, 2.0, 0.0, true),
        player(7, 0.3, 0.0, true),
    };
    snapshot.opponents = {player(1, 8.0, 6.0, false)};
    return snapshot;
}

}  // namespace

int main() {
    const decision::TeamTactics tactics;

    world::WorldSnapshot attack = base_snapshot();
    const auto support = tactics.plan(
        attack, decision::RoleManager::ROLE_ST, {4.0, 0.0});
    if (support.duty != decision::TacticalDuty::Support ||
        support.position_m[0] <= attack.ball.position_m[0] ||
        std::abs(support.position_m[1]) < 1.5) {
        std::cerr << "striker did not create a forward lateral support lane\n";
        return 1;
    }

    world::WorldSnapshot chase = attack;
    chase.server_time = 11.0;
    chase.own_score = 0;
    chase.opponent_score = 1;
    chase.match_time_s = 300.0;
    const auto chase_support = tactics.plan(
        chase, decision::RoleManager::ROLE_ST, {4.0, 0.0});
    world::WorldSnapshot protect = attack;
    protect.server_time = 12.0;
    protect.own_score = 1;
    protect.opponent_score = 0;
    protect.match_time_s = 300.0;
    const auto protect_support = tactics.plan(
        protect, decision::RoleManager::ROLE_ST, {4.0, 0.0});
    if (chase_support.position_m[0] <= support.position_m[0] + 0.5 ||
        protect_support.position_m[0] >= support.position_m[0] - 0.5) {
        std::cerr << "score/time risk mode did not change striker depth\n";
        return 1;
    }

    world::WorldSnapshot marked_attack = attack;
    marked_attack.opponents = {player(2, 4.0, 0.0, false)};
    const auto unmark = tactics.plan(
        marked_attack, decision::RoleManager::ROLE_ST, {4.0, 0.0});
    if (unmark.duty != decision::TacticalDuty::Unmark ||
        std::hypot(unmark.position_m[0] - 4.0, unmark.position_m[1]) < 1.0) {
        std::cerr << "marked striker did not leave the occupied formation point\n";
        return 1;
    }

    world::WorldSnapshot defense = base_snapshot();
    defense.self.position_m = {-9.0, 4.0, 0.8};
    defense.teammates = {player(6, -9.0, 4.0, true)};
    defense.opponents = {
        player(1, 0.2, 0.0, false),
        player(2, -8.0, 3.0, false),
    };
    const auto left_mark = tactics.plan(
        defense, decision::RoleManager::ROLE_CBL, {-10.0, 4.0});
    const auto right_cover = tactics.plan(
        defense, decision::RoleManager::ROLE_CBR, {-10.0, -4.0});
    if (left_mark.duty != decision::TacticalDuty::Mark ||
        right_cover.duty != decision::TacticalDuty::Mark ||
        left_mark.marked_opponent_player_number == 0 ||
        right_cover.marked_opponent_player_number == 0 ||
        left_mark.marked_opponent_player_number ==
            right_cover.marked_opponent_player_number ||
        left_mark.position_m[0] >=
            defense.opponents[static_cast<std::size_t>(
                left_mark.marked_opponent_player_number - 1)].position_m[0]) {
        std::cerr << "defenders did not produce unique goal-side responsibilities\n";
        return 1;
    }

    world::WorldSnapshot moving_ball = defense;
    moving_ball.self.position_m = {-4.0, 1.0, 0.8};
    moving_ball.self.orientation_wxyz = {0.0, 0.0, 0.0, 1.0};
    moving_ball.ball.position_m = {0.0, 1.0, 0.11};
    moving_ball.ball.velocity_valid = true;
    moving_ball.ball.velocity_mps = {-2.0, 0.0, 0.0};
    const auto intercept = tactics.plan(
        moving_ball, decision::RoleManager::ROLE_CDM, {-5.0, 0.0});
    const auto non_owner = tactics.plan(
        moving_ball, decision::RoleManager::ROLE_CBM, {-3.0, 2.0});
    if (intercept.duty != decision::TacticalDuty::Intercept ||
        non_owner.duty == decision::TacticalDuty::Intercept) {
        std::cerr << "ball interception ownership was not unique\n";
        return 1;
    }
    moving_ball.self.position_m = {4.0, -5.0, 0.8};
    moving_ball.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    const auto late = tactics.plan(
        moving_ball, decision::RoleManager::ROLE_CDM, {-5.0, 0.0});
    if (late.duty == decision::TacticalDuty::Intercept) {
        std::cerr << "late midfielder claimed an unreachable interception\n";
        return 1;
    }

    world::WorldSnapshot goalkeeper = defense;
    goalkeeper.player_number = 1;
    goalkeeper.self.position_m = {-27.0, 0.0, 0.8};
    goalkeeper.ball.position_m = {-23.0, 0.0, 0.11};
    goalkeeper.ball.velocity_valid = true;
    goalkeeper.ball.velocity_mps = {-1.0, 0.25, 0.0};
    const double reachable_heading_half_rad =
        math::deg_to_rad(114.0 * 0.5);
    goalkeeper.self.orientation_wxyz = {
        std::cos(reachable_heading_half_rad), 0.0, 0.0,
        std::sin(reachable_heading_half_rad)};
    const auto reachable = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (reachable.duty != decision::TacticalDuty::GoalkeeperIntercept ||
        reachable.position_m[1] < 0.9 || reachable.position_m[1] > 1.1) {
        std::cerr << "goalkeeper did not select the reachable goal-line crossing\n";
        return 1;
    }

    goalkeeper.ball.position_m = {-26.0, 0.0, 0.11};
    goalkeeper.ball.velocity_mps = {-2.0, 3.0, 0.0};
    goalkeeper.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    const auto unreachable = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (unreachable.duty != decision::TacticalDuty::GoalkeeperHold) {
        std::cerr << "goalkeeper chased an unreachable crossing\n";
        return 1;
    }

    // Regression from the 2026-09-06 comparison: a mostly longitudinal
    // 3.7 m/s shot crossed the keeper's current depth about 0.6 m to its side.
    // The full goal-line crossing was unreachable, but the former fallback
    // held the current pose instead of contesting the closer trajectory point.
    goalkeeper.ball.position_m = {-23.6848, -1.24576, 0.11};
    goalkeeper.ball.velocity_mps = {-3.68538, 0.509408, 0.0};
    goalkeeper.self.position_m = {-26.242, -0.270, 0.8};
    const double direct_shot_heading_half_rad =
        math::deg_to_rad(-146.553 * 0.5);
    goalkeeper.self.orientation_wxyz = {
        std::cos(direct_shot_heading_half_rad), 0.0, 0.0,
        std::sin(direct_shot_heading_half_rad)};
    const auto best_effort_block = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (best_effort_block.duty !=
            decision::TacticalDuty::GoalkeeperIntercept ||
        best_effort_block.position_m[0] >= -25.8 ||
        best_effort_block.position_m[1] >= -0.65 ||
        std::hypot(
            best_effort_block.position_m[0] - goalkeeper.self.position_m[0],
            best_effort_block.position_m[1] - goalkeeper.self.position_m[1]) >=
            0.9) {
        std::cerr << "goalkeeper did not attempt the closer shot-line block\n";
        return 1;
    }

    goalkeeper.ball.position_m = {-25.2, 1.0, 0.11};
    goalkeeper.ball.velocity_valid = false;
    goalkeeper.self.position_m = {-27.0, 0.0, 0.8};
    goalkeeper.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    goalkeeper.opponents = {player(1, -20.0, 4.0, false)};
    const auto smother = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (smother.duty != decision::TacticalDuty::GoalkeeperSmother ||
        !decision::field_geometry::is_in_our_goalie_area(
            smother.position_m)) {
        std::cerr << "goalkeeper did not claim a safe loose ball in its area\n";
        return 1;
    }
    goalkeeper.opponents = {player(1, -25.1, 1.0, false)};
    const auto contested_smother = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (contested_smother.duty != decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "goalkeeper yielded a central last-area carry\n";
        return 1;
    }
    goalkeeper.ball.position_m = {-24.1, 1.0, 0.11};
    goalkeeper.opponents = {player(1, -24.0, 1.0, false)};
    const auto outside_emergency = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (outside_emergency.duty == decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "goalkeeper ignored an opponent-first race outside the emergency depth\n";
        return 1;
    }

    goalkeeper.ball.position_m = {-27.35, 1.0, 0.11};
    goalkeeper.self.position_m = {-27.0, 0.0, 0.8};
    const double emergency_heading_half_rad =
        math::deg_to_rad(109.3 * 0.5);
    goalkeeper.self.orientation_wxyz = {
        std::cos(emergency_heading_half_rad), 0.0, 0.0,
        std::sin(emergency_heading_half_rad)};
    goalkeeper.opponents = {player(1, -27.2, 1.0, false)};
    const auto emergency_smother = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (emergency_smother.duty !=
        decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "goalkeeper yielded an imminent goal-mouth challenge\n";
        return 1;
    }

    // Regression from the 2026-12-17 comparison: the attacker carried the
    // ball 36 cm outside the post at y=2.19 m, then cut inside before the old
    // emergency corridor armed. Once the keeper has taken its angular cover
    // position this must be treated as an imminent goal-mouth challenge.
    goalkeeper.ball.position_m = {-27.0, 2.19, 0.11};
    goalkeeper.self.position_m = {-27.0, 1.15, 0.8};
    goalkeeper.self.orientation_wxyz = {
        std::cos(math::deg_to_rad(90.0 * 0.5)), 0.0, 0.0,
        std::sin(math::deg_to_rad(90.0 * 0.5))};
    goalkeeper.opponents = {player(1, -27.0, 2.19, false)};
    const auto outside_post_smother = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (outside_post_smother.duty !=
        decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "goalkeeper waited too long on an outside-post carry\n";
        return 1;
    }

    // Regression from the 2026-12-24 comparison: at this exact geometry the
    // old 1.5 m depth gate held the keeper on its cover arc for another 360
    // cycles.  It only challenged one sample before the goal.  The target is
    // still inside the goalkeeper area and the keeper can reach it, so the
    // emergency claim must begin now even when the opponent is ball-side.
    goalkeeper.ball.position_m = {-25.80, 2.21, 0.11};
    goalkeeper.self.position_m = {-26.86, 0.89, 0.8};
    const double near_post_heading_half_rad =
        math::deg_to_rad(51.2 * 0.5);
    goalkeeper.self.orientation_wxyz = {
        std::cos(near_post_heading_half_rad), 0.0, 0.0,
        std::sin(near_post_heading_half_rad)};
    goalkeeper.opponents = {player(3, -25.80, 2.21, false)};
    const auto early_goal_mouth_smother = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (early_goal_mouth_smother.duty !=
        decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "goalkeeper repeated the late 2026-12-24 challenge\n";
        return 1;
    }

    // Do not flip into a turn-away retreat when the perceived ball crosses
    // the old 3.0 m depth edge during a live near-post cutback.
    goalkeeper.ball.position_m = {-24.45, 2.55, 0.11};
    goalkeeper.self.position_m = {-25.80, 1.50, 0.8};
    const double cutback_heading_half_rad =
        math::deg_to_rad(37.9 * 0.5);
    goalkeeper.self.orientation_wxyz = {
        std::cos(cutback_heading_half_rad), 0.0, 0.0,
        std::sin(cutback_heading_half_rad)};
    goalkeeper.opponents = {player(3, -24.45, 2.55, false)};
    const auto near_post_cutback = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (near_post_cutback.duty !=
        decision::TacticalDuty::GoalkeeperSmother) {
        std::cerr << "goalkeeper abandoned a live near-post cutback\n";
        return 1;
    }

    goalkeeper.ball.position_m = {-23.0, 5.0, 0.11};
    goalkeeper.self.position_m = {-27.0, 0.0, 0.8};
    goalkeeper.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    goalkeeper.opponents.clear();
    const auto angular_hold = tactics.plan(
        goalkeeper, decision::RoleManager::ROLE_GK, {-27.0, 0.0});
    if (angular_hold.duty != decision::TacticalDuty::GoalkeeperHold ||
        angular_hold.position_m[1] < 0.70) {
        std::cerr << "goalkeeper hold did not cover the near-post angle\n";
        return 1;
    }

    world::WorldSnapshot stopped = attack;
    stopped.play_mode = world::PlayMode::TheirFreeKick;
    const auto formation = tactics.plan(
        stopped, decision::RoleManager::ROLE_ST, {3.0, -2.0});
    if (formation.duty != decision::TacticalDuty::Formation) {
        std::cerr << "open-play tactics leaked into a restart\n";
        return 1;
    }

    world::WorldSnapshot team_defense = base_snapshot();
    team_defense.player_number = 4;
    team_defense.self.position_m = {-4.0, 1.0, 0.8};
    team_defense.teammates = {
        player(1, -27.0, 0.0, true),
        player(2, -9.0, 4.0, true),
        player(3, -9.0, -4.0, true),
        player(4, -4.0, 1.0, true),
        player(5, -1.0, 7.0, true),
        player(6, 7.0, 5.0, true),
        player(7, 8.0, -5.0, true),
    };
    team_defense.opponents = {
        player(1, 0.1, 0.8, false),
        player(2, -8.0, 3.0, false),
        player(3, -7.5, -3.0, false),
    };
    team_defense.ball.position_m = {0.0, 1.0, 0.11};
    team_defense.ball.velocity_valid = true;
    team_defense.ball.velocity_mps = {-2.0, 0.0, 0.0};
    const std::vector<decision::RoleAssignment> roles{
        {1, decision::RoleManager::ROLE_GK, {-27.0, 0.0}},
        {2, decision::RoleManager::ROLE_CBL, {-10.0, 4.0}},
        {3, decision::RoleManager::ROLE_CBR, {-10.0, -4.0}},
        {4, decision::RoleManager::ROLE_CDM, {-5.0, 0.0}},
        {5, decision::RoleManager::ROLE_CBM, {-3.0, 2.0}},
        {6, decision::RoleManager::ROLE_ST, {4.0, 3.0}},
        {7, decision::RoleManager::ROLE_AP, {0.0, 1.0}},
    };
    const auto team_plan = tactics.plan_all(team_defense, roles);
    int intercept_count = 0;
    std::vector<int> marked_players;
    for (const auto& assignment : team_plan.assignments) {
        if (assignment.target.duty == decision::TacticalDuty::Intercept) {
            ++intercept_count;
        }
        if (assignment.target.duty == decision::TacticalDuty::Mark) {
            marked_players.push_back(
                assignment.target.marked_opponent_player_number);
        }
    }
    std::sort(marked_players.begin(), marked_players.end());
    if (intercept_count != 1 || marked_players.size() != 2U ||
        marked_players[0] <= 0 || marked_players[0] == marked_players[1]) {
        std::cerr << "full-team planner did not assign unique defensive duties\n";
        return 1;
    }

    world::WorldSnapshot team_attack = base_snapshot();
    team_attack.server_time = 20.0;
    team_attack.player_number = 7;
    team_attack.self.position_m = {0.3, 0.0, 0.8};
    team_attack.teammates = {
        player(1, -27.0, 0.0, true),
        player(2, -10.0, 4.0, true),
        player(3, -10.0, -4.0, true),
        player(4, -5.0, 0.0, true),
        player(5, -3.0, 2.0, true),
        player(6, 4.0, 3.0, true),
        player(7, 0.3, 0.0, true),
    };
    for (auto& teammate : team_attack.teammates) {
        teammate.last_seen_time = team_attack.server_time;
    }
    team_attack.opponents = {player(1, 8.0, 6.0, false)};
    team_attack.opponents.front().last_seen_time = team_attack.server_time;
    const auto attack_plan = tactics.plan_all(team_attack, roles);
    const auto* striker_support = attack_plan.for_role(
        decision::RoleManager::ROLE_ST);
    const auto* central_support = attack_plan.for_role(
        decision::RoleManager::ROLE_CBM);
    if (striker_support == nullptr || central_support == nullptr ||
        !attack_plan.fresh || attack_plan.revision == 0U ||
        attack_plan.tactical_state.ball_owner_player_number != 7 ||
        (striker_support->target.duty != decision::TacticalDuty::Support &&
         striker_support->target.duty != decision::TacticalDuty::Unmark) ||
        (central_support->target.duty != decision::TacticalDuty::Support &&
         central_support->target.duty != decision::TacticalDuty::Unmark) ||
        std::hypot(
            striker_support->target.position_m[0] -
                central_support->target.position_m[0],
            striker_support->target.position_m[1] -
                central_support->target.position_m[1]) < 2.0) {
        std::cerr << "joint attack plan did not allocate separated support lanes\n";
        return 1;
    }

    world::WorldSnapshot keeper_claim = team_defense;
    keeper_claim.server_time = 21.0;
    keeper_claim.ball.position_m = {-25.2, 1.0, 0.11};
    keeper_claim.ball.velocity_valid = false;
    keeper_claim.opponents = {player(1, -18.0, 6.0, false)};
    keeper_claim.opponents.front().last_seen_time = keeper_claim.server_time;
    keeper_claim.teammates[0].position_m = {-26.0, 0.5, 0.8};
    for (auto& teammate : keeper_claim.teammates) {
        teammate.last_seen_time = keeper_claim.server_time;
    }
    const auto keeper_claim_plan = tactics.plan_all(keeper_claim, roles);
    const auto* keeper_duty = keeper_claim_plan.for_role(
        decision::RoleManager::ROLE_GK);
    const auto* ap_cover = keeper_claim_plan.for_role(
        decision::RoleManager::ROLE_AP);
    if (keeper_duty == nullptr || ap_cover == nullptr ||
        keeper_duty->target.duty !=
            decision::TacticalDuty::GoalkeeperSmother ||
        ap_cover->target.duty != decision::TacticalDuty::Cover ||
        decision::field_geometry::is_in_our_goalie_area(
            ap_cover->target.position_m)) {
        std::cerr << "team did not protect the goalkeeper's smother claim\n";
        return 1;
    }

    auto reversed_roles = roles;
    std::reverse(reversed_roles.begin(), reversed_roles.end());
    const auto reordered_plan = tactics.plan_all(team_defense, reversed_roles);
    for (int number = 1; number <= 7; ++number) {
        const auto* original = team_plan.for_player(number);
        const auto* reordered = reordered_plan.for_player(number);
        if (original == nullptr || reordered == nullptr ||
            original->role_id != reordered->role_id ||
            original->target.duty != reordered->target.duty ||
            original->target.marked_opponent_player_number !=
                reordered->target.marked_opponent_player_number ||
            team_plan.revision != reordered_plan.revision ||
            std::hypot(
                original->target.position_m[0] - reordered->target.position_m[0],
                original->target.position_m[1] - reordered->target.position_m[1]) >
                1.0e-9) {
            std::cerr << "team plan depended on role-assignment input order\n";
            return 1;
        }
    }

    world::WorldSnapshot stale_ball = team_defense;
    stale_ball.ball.visible = false;
    stale_ball.ball.position_age_s = 0.76;
    const auto stale_plan = tactics.plan_all(stale_ball, roles);
    const auto* stale_keeper = stale_plan.for_role(
        decision::RoleManager::ROLE_GK);
    const bool field_player_left_shape = std::any_of(
        stale_plan.assignments.begin(), stale_plan.assignments.end(),
        [](const decision::TeamTacticalAssignment& assignment) {
            return assignment.role_id != decision::RoleManager::ROLE_GK &&
                assignment.target.duty != decision::TacticalDuty::Formation;
        });
    if (stale_plan.fresh || stale_plan.revision == 0U ||
        field_player_left_shape || stale_keeper == nullptr ||
        stale_keeper->target.duty !=
            decision::TacticalDuty::GoalkeeperHold ||
        std::abs(
            stale_keeper->target.position_m[0] -
            (-decision::field_geometry::kActualHalfLengthM +
             decision::field_geometry::kGkHoldDepthM)) > 1.0e-9 ||
        std::abs(stale_keeper->target.position_m[1]) > 1.0e-9) {
        std::cerr << "stale ball did not produce the safe goalkeeper hold\n";
        return 1;
    }

    // A local goalkeeper owns a longer near-contact track because its torso
    // naturally occludes the ball during a smother. Field players must still
    // remain in formation under the same globally stale observation.
    world::WorldSnapshot occluded_keeper = stale_ball;
    occluded_keeper.player_number = 1;
    occluded_keeper.self.position_m = {-25.65, 0.94, 0.8};
    occluded_keeper.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    occluded_keeper.ball.position_m = {-25.32, 0.94, 0.11};
    occluded_keeper.ball.position_age_s = 1.0;
    occluded_keeper.ball.near_contact_track = true;
    occluded_keeper.ball.velocity_valid = false;
    occluded_keeper.opponents = {player(3, -25.32, 0.94, false)};
    occluded_keeper.opponents.front().last_seen_time =
        occluded_keeper.server_time;
    const auto occluded_plan = tactics.plan_all(occluded_keeper, roles);
    const auto* occluded_keeper_duty = occluded_plan.for_role(
        decision::RoleManager::ROLE_GK);
    const bool occluded_field_player_left_shape = std::any_of(
        occluded_plan.assignments.begin(), occluded_plan.assignments.end(),
        [](const decision::TeamTacticalAssignment& assignment) {
            return assignment.role_id != decision::RoleManager::ROLE_GK &&
                assignment.target.duty != decision::TacticalDuty::Formation;
        });
    if (occluded_plan.fresh || occluded_keeper_duty == nullptr ||
        occluded_keeper_duty->target.duty !=
            decision::TacticalDuty::GoalkeeperSmother ||
        occluded_field_player_left_shape) {
        std::cerr << "goalkeeper abandoned an occluded near-contact smother\n";
        return 1;
    }
    return 0;
}
