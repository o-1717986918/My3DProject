// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/role_behaviors.h"

#include <cmath>
#include <iostream>
#include <variant>

namespace {

world::WorldSnapshot make_open_pass_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.team_name = "My3D";
    snapshot.player_number = 7;
    snapshot.server_time = 1.0;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {-0.33, 0.0, 0.8};
    snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {0.0, 0.0, 0.11};
    snapshot.teammates.resize(7);
    for (int number = 1; number <= 7; ++number) {
        auto& player = snapshot.teammates[static_cast<std::size_t>(number - 1)];
        player.player_number = number;
        player.is_teammate = true;
    }
    auto& receiver = snapshot.teammates[5];
    receiver.seen = true;
    receiver.last_seen_time = snapshot.server_time;
    receiver.position_m = {1.0, 0.0, 0.8};
    return snapshot;
}

}  // namespace

int main() {
    world::WorldSnapshot misaligned_snapshot = make_open_pass_snapshot();
    constexpr double kYaw30HalfRadians = 0.2617993877991494;
    misaligned_snapshot.self.orientation_wxyz = {
        std::cos(kYaw30HalfRadians), 0.0, 0.0,
        std::sin(kYaw30HalfRadians)};
    decision::APBehavior misaligned_behavior;
    decision::Blackboard misaligned_blackboard;
    decision::RoleManager misaligned_role_manager;
    misaligned_behavior.make_command(
        misaligned_snapshot, misaligned_blackboard, misaligned_role_manager,
        true, true);
    if (misaligned_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "misaligned targeted pass bypassed the angle envelope\n";
        return 1;
    }

    world::WorldSnapshot snapshot = make_open_pass_snapshot();
    decision::APBehavior behavior;
    decision::Blackboard blackboard;
    decision::RoleManager role_manager;

    const decision::HighLevelCommand waiting = behavior.make_command(
        snapshot, blackboard, role_manager, true, true);
    if (!std::holds_alternative<decision::WalkCommand>(waiting) ||
        !blackboard.exists(decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "passer did not wait for a ready acknowledgement\n";
        return 1;
    }

    const auto selected = blackboard.get<strategy::CooperativeAction>(
        decision::Blackboard::kKeySelectedCooperativeAction);

    // The proposal must survive temporary receiver occlusion. Execution is
    // still impossible until the receiver sends a matching Ready packet.
    snapshot.server_time = 1.005;
    snapshot.teammates[5].seen = false;
    snapshot.teammates[5].last_seen_time = -1.0;
    snapshot.teammates[5].position_m = {0.0, 0.0, 0.8};
    const decision::HighLevelCommand occluded_wait = behavior.make_command(
        snapshot, blackboard, role_manager, true, true);
    if (std::holds_alternative<decision::KickCommand>(occluded_wait)) {
        std::cerr << "passer released while receiver was occluded and not ready\n";
        return 1;
    }

    snapshot.server_time = 1.01;
    snapshot.teammates[5].seen = true;
    snapshot.teammates[5].last_seen_time = snapshot.server_time;
    snapshot.teammates[5].position_m = {1.0, 0.0, 0.8};
    snapshot.self.position_m = {-0.35, 0.3, 0.8};
    const decision::HighLevelCommand still_waiting = behavior.make_command(
        snapshot, blackboard, role_manager, true, true);
    if (std::holds_alternative<decision::KickCommand>(still_waiting)) {
        std::cerr << "passer released after setup drift without ready acknowledgement\n";
        return 1;
    }

    snapshot.server_time = 1.02;
    snapshot.self.position_m = {-0.33, 0.0, 0.8};
    snapshot.team_comm_snapshot.pass_intents.push_back({
        selected.target_player_number,
        51,
        comm::PassIntentState::Ready,
        snapshot.player_number,
        selected.target_player_number,
        selected.sequence_id,
        selected.target_point_m[0],
        selected.target_point_m[1],
        selected.requested_ball_speed_mps,
        selected.predicted_ball_time_s,
    });
    const decision::HighLevelCommand stabilizing = behavior.make_command(
        snapshot, blackboard, role_manager, true, true);
    if (std::holds_alternative<decision::KickCommand>(stabilizing)) {
        std::cerr << "ready pass skipped the stable setup hold\n";
        return 1;
    }
    snapshot.server_time = 1.63;
    const decision::HighLevelCommand released = behavior.make_command(
        snapshot, blackboard, role_manager, true, true);
    if (!std::holds_alternative<decision::KickCommand>(released)) {
        std::cerr << "ready pass was not released as a kick\n";
        return 1;
    }
    const auto& kick = std::get<decision::KickCommand>(released);
    if (kick.mode != decision::KickMode::TargetedPass ||
        !kick.target_point_m.has_value() ||
        kick.receiver_player_number != selected.target_player_number ||
        kick.sequence_id != selected.sequence_id ||
        kick.action_id != selected.action_id) {
        std::cerr << "targeted pass metadata was not preserved\n";
        return 1;
    }

    decision::APBehavior disabled_behavior;
    decision::Blackboard disabled_blackboard;
    const decision::HighLevelCommand fallback = disabled_behavior.make_command(
        make_open_pass_snapshot(), disabled_blackboard, role_manager, false);
    if (disabled_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "disabled strategy did not retain the safe dribble fallback\n";
        return 1;
    }
    if (const auto* fallback_kick = std::get_if<decision::KickCommand>(&fallback);
        fallback_kick != nullptr &&
        fallback_kick->mode != decision::KickMode::ForwardContact) {
        std::cerr << "disabled strategy emitted a strategy-dependent kick\n";
        return 1;
    }

    decision::APBehavior risk_behavior;
    decision::Blackboard risk_blackboard;
    risk_blackboard.set(
        decision::Blackboard::kKeyTacticalTarget,
        decision::TacticalTarget{
            decision::TacticalDuty::Cover,
            {-3.0, 1.0},
            std::array<double, 2>{0.0, 0.0},
            0,
            0.8});
    static_cast<void>(risk_behavior.make_command(
        make_open_pass_snapshot(), risk_blackboard, role_manager, false));
    if (risk_blackboard.get<decision::TacticalTarget>(
            decision::Blackboard::kKeyTacticalTarget).duty !=
        decision::TacticalDuty::Cover) {
        std::cerr << "AP behavior overwrote the team tactical duty\n";
        return 1;
    }
    return 0;
}
