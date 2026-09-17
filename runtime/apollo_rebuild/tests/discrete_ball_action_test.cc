// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/role_behaviors.h"

#include <cassert>
#include <cmath>
#include <variant>

int main() {
    world::WorldSnapshot snapshot;
    snapshot.player_number = 7;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.server_time = 10.0;
    snapshot.self.position_m = {0.0, 0.0, 0.8};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {0.32, 0.04, 0.11};
    snapshot.ball.velocity_valid = true;
    snapshot.ball.velocity_mps = {0.0, 0.0, 0.0};

    decision::Blackboard blackboard;
    decision::RoleManager role_manager;
    decision::APBehavior attacker;

    // Keep a short Walk command in the release slot so the default-on dynamic
    // selector receives its two-frame first right of refusal.
    const auto dwell = attacker.make_command(snapshot, blackboard, role_manager);
    const auto* dwell_walk = std::get_if<decision::WalkCommand>(&dwell);
    assert(dwell_walk != nullptr);
    assert(!dwell_walk->target_absolute);
    assert(dwell_walk->target_2d_m[0] == 0.0);
    assert(dwell_walk->target_2d_m[1] == 0.0);

    snapshot.server_time = 10.12;
    const auto release = attacker.make_command(snapshot, blackboard, role_manager);
    const auto* kick = std::get_if<decision::KickCommand>(&release);
    assert(kick != nullptr);
    assert(kick->mode == decision::KickMode::TargetedPass);
    assert(kick->target_point_m.has_value());
    assert(kick->requested_ball_speed_mps == 1.43);
    assert(kick->allow_forward_contact_fallback);

    snapshot.server_time = 10.50;
    const auto retained = attacker.make_command(snapshot, blackboard, role_manager);
    assert(std::holds_alternative<decision::KickCommand>(retained));

    // Opponent restarts must never release a local ball action even when the
    // geometric slot remains valid.
    attacker.reset_state();
    snapshot.server_time = 20.0;
    snapshot.play_mode = world::PlayMode::TheirFreeKick;
    snapshot.play_mode_group = world::PlayModeGroup::TheirKick;
    const auto restart = attacker.make_command(snapshot, blackboard, role_manager);
    assert(std::holds_alternative<decision::WalkCommand>(restart));

    // Open-play navigation to a far target behind the robot must first emit a
    // pure-yaw command.  This makes the default-on RapidTurn policy reachable
    // instead of asking the stable walk actor to translate sideways while it
    // turns.  Near-ball precision remains protected in WalkRunner itself.
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {0.0, 0.0, 0.8};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.ball.position_m = {5.0, 0.0, 0.11};
    decision::Blackboard navigation_blackboard;
    navigation_blackboard.set<std::array<double, 2>>(
        decision::Blackboard::kKeyRolePos, {-8.0, 0.0});
    decision::SimpleRoleBehavior center_back(
        decision::RoleManager::ROLE_CBM, false);
    const auto turn_first = center_back.make_command(
        snapshot, navigation_blackboard);
    const auto* turn_walk = std::get_if<decision::WalkCommand>(&turn_first);
    assert(turn_walk != nullptr);
    assert(!turn_walk->target_absolute);
    assert(turn_walk->target_2d_m[0] == 0.0);
    assert(turn_walk->target_2d_m[1] == 0.0);
    assert(turn_walk->orientation_deg.has_value());
    assert(std::abs(std::abs(*turn_walk->orientation_deg) - 180.0) < 1.0e-6);
    return 0;
}
