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
    for (const char* name : {
             "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
             "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
             "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
             "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"}) {
        snapshot.self.joint_positions_deg[name] = 0.0;
        snapshot.self.joint_velocities_deg_s[name] = 0.0;
    }
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
    const auto misaligned_command = misaligned_behavior.make_command(
        misaligned_snapshot, misaligned_blackboard, misaligned_role_manager,
        true, true);
    if (!misaligned_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction) ||
        std::holds_alternative<decision::KickCommand>(misaligned_command)) {
        std::cerr << "misaligned targeted pass was not retained for safe alignment\n";
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
        comm::PassIntentAuthor::Receiver,
        snapshot.player_number,
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

    // A farther receiver must keep its distance-conditioned speed all the way
    // through the Ready handshake and exact release. This prevents the
    // planner from advertising a long pass that the motion layer interprets
    // as the old fixed 2 m request.
    world::WorldSnapshot range_snapshot = make_open_pass_snapshot();
    range_snapshot.self.position_m = {-0.31, 0.04, 0.8};
    range_snapshot.teammates[5].position_m = {3.5, 0.0, 0.8};
    decision::APBehavior range_behavior;
    decision::Blackboard range_blackboard;
    static_cast<void>(range_behavior.make_command(
        range_snapshot, range_blackboard, role_manager, true, true));
    if (!range_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "distance-conditioned pass was not selected\n";
        return 1;
    }
    const auto range_selected = range_blackboard.get<strategy::CooperativeAction>(
        decision::Blackboard::kKeySelectedCooperativeAction);
    if (range_selected.requested_ball_speed_mps <= 1.63) {
        std::cerr << "far pass retained the short-pass speed\n";
        return 1;
    }
    range_snapshot.server_time = 1.02;
    range_snapshot.team_comm_snapshot.pass_intents.push_back({
        range_selected.target_player_number,
        52,
        comm::PassIntentState::Ready,
        range_snapshot.player_number,
        range_selected.target_player_number,
        range_selected.sequence_id,
        range_selected.target_point_m[0],
        range_selected.target_point_m[1],
        range_selected.requested_ball_speed_mps,
        range_selected.predicted_ball_time_s,
        comm::PassIntentAuthor::Receiver,
        range_snapshot.player_number,
    });
    static_cast<void>(range_behavior.make_command(
        range_snapshot, range_blackboard, role_manager, true, true));
    range_snapshot.server_time = 1.63;
    const auto range_release = range_behavior.make_command(
        range_snapshot, range_blackboard, role_manager, true, true);
    const auto* range_kick = std::get_if<decision::KickCommand>(&range_release);
    if (range_kick == nullptr ||
        range_kick->mode != decision::KickMode::TargetedPass ||
        std::abs(
            range_kick->requested_ball_speed_mps -
            range_selected.requested_ball_speed_mps) > 1.0e-9) {
        std::cerr << "distance-conditioned pass did not reach kick release\n";
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

    // A failed Ready handshake must yield the ball to a local action for a
    // bounded interval instead of immediately proposing the same pass again.
    decision::APBehavior expired_pass_behavior;
    decision::Blackboard expired_pass_blackboard;
    world::WorldSnapshot expired_pass_snapshot = make_open_pass_snapshot();
    static_cast<void>(expired_pass_behavior.make_command(
        expired_pass_snapshot, expired_pass_blackboard, role_manager,
        true, true));
    expired_pass_snapshot.server_time = 3.6;
    expired_pass_snapshot.teammates[5].last_seen_time =
        expired_pass_snapshot.server_time;
    const auto terminal_recovery = expired_pass_behavior.make_command(
        expired_pass_snapshot, expired_pass_blackboard, role_manager,
        true, true);
    static_cast<void>(terminal_recovery);
    const auto& terminal_recovery_plan = expired_pass_blackboard.get<
        strategy::PlanningResult>(decision::Blackboard::kKeyStrategyPlan);
    if (!terminal_recovery_plan.selected.has_value() ||
        terminal_recovery_plan.selected->category ==
            strategy::ActionCategory::Pass) {
        std::cerr << "expired pass blocked immediate local recovery\n";
        return 1;
    }
    expired_pass_snapshot.server_time = 4.5;
    expired_pass_snapshot.teammates[5].last_seen_time =
        expired_pass_snapshot.server_time;
    static_cast<void>(expired_pass_behavior.make_command(
        expired_pass_snapshot, expired_pass_blackboard, role_manager,
        true, true));
    const auto& recovery_plan = expired_pass_blackboard.get<
        strategy::PlanningResult>(decision::Blackboard::kKeyStrategyPlan);
    if (!recovery_plan.selected.has_value() ||
        recovery_plan.selected->category == strategy::ActionCategory::Pass) {
        std::cerr << "expired pass immediately monopolized the ball again\n";
        return 1;
    }

    // Generic lateral travel is intentionally composed from an in-place turn
    // and a later forward walk until a stable omnidirectional actor exists.
    world::WorldSnapshot turn_snapshot = make_open_pass_snapshot();
    turn_snapshot.player_number = 6;
    turn_snapshot.self.position_m = {0.0, 0.0, 0.8};
    turn_snapshot.self.orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
    turn_snapshot.teammates.clear();
    turn_snapshot.opponents.clear();
    decision::SimpleRoleBehavior turn_behavior(
        decision::RoleManager::ROLE_ST, false);
    decision::Blackboard turn_blackboard;
    turn_blackboard.set(
        decision::Blackboard::kKeyTacticalTarget,
        decision::TacticalTarget{
            decision::TacticalDuty::Formation,
            {0.0, 5.0},
            std::array<double, 2>{0.0, 0.0},
            0,
            0.5});
    const auto turn_first = turn_behavior.make_command(
        turn_snapshot, turn_blackboard);
    const auto* turn_walk = std::get_if<decision::WalkCommand>(&turn_first);
    if (turn_walk == nullptr || turn_walk->target_absolute ||
        std::hypot(
            turn_walk->target_2d_m[0], turn_walk->target_2d_m[1]) > 1.0e-9 ||
        !turn_walk->orientation_deg.has_value() ||
        std::abs(*turn_walk->orientation_deg - 90.0) > 1.0e-6) {
        std::cerr << "large lateral request was not converted to turn-first\n";
        return 1;
    }

    world::WorldSnapshot braking_snapshot = make_open_pass_snapshot();
    braking_snapshot.teammates.clear();
    braking_snapshot.self.position_m = {-0.50, -0.04, 0.8};
    // The strategy admission gate spans the same 0.50 m/s envelope as the
    // motion runner; the pre-settle controller, not admission, owns braking.
    braking_snapshot.self.lin_vel_b = {0.45, 0.0, 0.0};
    decision::APBehavior braking_behavior;
    decision::Blackboard braking_blackboard;
    const auto braking_command = braking_behavior.make_command(
        braking_snapshot, braking_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(braking_command)) {
        std::cerr << "speed-aware kick setup did not brake before the release slot\n";
        return 1;
    }
    braking_snapshot.server_time += 0.05;
    braking_snapshot.self.lin_vel_b = {0.10, 0.0, 0.0};
    const auto braking_debounce = braking_behavior.make_command(
        braking_snapshot, braking_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(braking_debounce)) {
        std::cerr << "pre-settle brake accepted a single low-speed sample\n";
        return 1;
    }

    // The calm, low-speed checks admit a local action into setup; they must
    // not cancel it on the next cycle merely because the positioning command
    // has accelerated the torso.  The action remains bounded by its own
    // timeout, ball visibility and opponent-race cancellation.
    world::WorldSnapshot committed_snapshot = make_open_pass_snapshot();
    committed_snapshot.teammates.clear();
    committed_snapshot.self.position_m = {-0.50, -0.04, 0.8};
    decision::APBehavior committed_behavior;
    decision::Blackboard committed_blackboard;
    const auto committed_entry = committed_behavior.make_command(
        committed_snapshot, committed_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::WalkCommand>(committed_entry)) {
        std::cerr << "local action did not enter bounded precision setup\n";
        return 1;
    }
    committed_snapshot.server_time += 0.02;
    committed_snapshot.self.lin_vel_b = {0.40, 0.0, 0.0};
    decision::Blackboard committed_next_blackboard;
    const auto committed_next = committed_behavior.make_command(
        committed_snapshot, committed_next_blackboard, role_manager,
        false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(committed_next) ||
        !committed_next_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction) ||
        committed_next_blackboard.get<strategy::CooperativeAction>(
            decision::Blackboard::kKeySelectedCooperativeAction).category !=
            strategy::ActionCategory::Dribble) {
        std::cerr << "local action was cancelled by its own setup speed\n";
        return 1;
    }
    committed_snapshot.server_time += 0.02;
    decision::Blackboard lost_race_blackboard;
    decision::TeamPlan lost_race_plan;
    lost_race_plan.tactical_state.possession =
        strategy::PossessionOwner::Theirs;
    lost_race_plan.tactical_state.nearest_teammate_ball_time_s = 1.0;
    lost_race_plan.tactical_state.nearest_opponent_ball_time_s = 0.0;
    lost_race_blackboard.set(
        decision::Blackboard::kKeyTeamPlan, lost_race_plan);
    static_cast<void>(committed_behavior.make_command(
        committed_snapshot, lost_race_blackboard, role_manager, false, true));
    if (lost_race_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "local action survived a decisive opponent race loss\n";
        return 1;
    }

    world::WorldSnapshot procedural_snapshot = make_open_pass_snapshot();
    procedural_snapshot.self.position_m = {-0.32, -0.04, 0.8};
    decision::APBehavior procedural_behavior;
    decision::Blackboard procedural_blackboard;
    const auto procedural_stabilizing = procedural_behavior.make_command(
        procedural_snapshot, procedural_blackboard, role_manager, false, true);
    if (std::holds_alternative<decision::KickCommand>(procedural_stabilizing)) {
        std::cerr << "procedural dribble skipped the neutral-phase debounce\n";
        return 1;
    }
    procedural_snapshot.server_time += 0.05;
    // The first sample entered the strict +/-20 mm release gate.  Simulate
    // the 1.2 mm cross-frame drift observed on the real server; it must stay
    // latched inside the separately validated +/-25 mm dispatch boundary.
    procedural_snapshot.self.position_m[1] = -0.0612;
    const auto procedural_release = procedural_behavior.make_command(
        procedural_snapshot, procedural_blackboard, role_manager, false, true);
    const auto* procedural_kick =
        std::get_if<decision::KickCommand>(&procedural_release);
    if (procedural_kick == nullptr ||
        procedural_kick->mode != decision::KickMode::DribbleTouch ||
        procedural_kick->allow_forward_contact_fallback ||
        !procedural_kick->target_point_m.has_value() ||
        std::abs(procedural_kick->requested_ball_speed_mps - 0.90) > 1.0e-9) {
        std::cerr << "enabled procedural dribble did not emit its exact contract\n";
        return 1;
    }

    // The decision release contract must include every dynamic guard used by
    // the procedural runner. Otherwise a high-rate gait transition is
    // announced as a kick and rejected one cycle later by the motion layer.
    world::WorldSnapshot transition_snapshot = make_open_pass_snapshot();
    transition_snapshot.teammates.clear();
    transition_snapshot.self.position_m = {-0.32, -0.04, 0.8};
    transition_snapshot.self.gyro_deg_s[0] = 40.0;
    transition_snapshot.self.joint_velocities_deg_s["Right_Knee_Pitch"] = 80.0;
    decision::APBehavior transition_behavior;
    decision::Blackboard transition_blackboard;
    const auto dynamic_hold = transition_behavior.make_command(
        transition_snapshot, transition_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(dynamic_hold)) {
        std::cerr << "procedural kick ignored its shared dynamic entry guard\n";
        return 1;
    }
    transition_snapshot.server_time += 0.05;
    transition_snapshot.self.gyro_deg_s[0] = 0.0;
    transition_snapshot.self.joint_velocities_deg_s["Right_Knee_Pitch"] = 0.0;
    const auto transition_settle = transition_behavior.make_command(
        transition_snapshot, transition_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(transition_settle)) {
        std::cerr << "procedural transition skipped its stable-entry debounce\n";
        return 1;
    }
    transition_snapshot.server_time += 0.05;
    const auto guarded_release = transition_behavior.make_command(
        transition_snapshot, transition_blackboard, role_manager, false, true);
    if (const auto* guarded_kick =
            std::get_if<decision::KickCommand>(&guarded_release);
        guarded_kick == nullptr ||
        guarded_kick->mode != decision::KickMode::DribbleTouch) {
        std::cerr << "procedural kick did not release after dynamic settling\n";
        return 1;
    }
    decision::ExecutionFeedback rejected_dribble;
    rejected_dribble.request_kind = decision::MotionRequestKind::Kick;
    rejected_dribble.status = decision::ExecutionStatus::Rejected;
    rejected_dribble.server_time = procedural_snapshot.server_time;
    rejected_dribble.cooperative_action_id = procedural_kick->action_id;
    rejected_dribble.sequence_id = procedural_kick->sequence_id;
    procedural_behavior.apply_execution_feedback(rejected_dribble);
    procedural_snapshot.server_time += 0.01;
    const auto retry_after_rejection = procedural_behavior.make_command(
        procedural_snapshot, procedural_blackboard, role_manager, false, true);
    if (const auto* repeated =
            std::get_if<decision::KickCommand>(&retry_after_rejection);
        repeated != nullptr && repeated->action_id == procedural_kick->action_id) {
        std::cerr << "rejected dribble remained latched until nominal timeout\n";
        return 1;
    }

    world::WorldSnapshot pressured_snapshot = make_open_pass_snapshot();
    pressured_snapshot.teammates.clear();
    pressured_snapshot.self.position_m = {-0.34, 0.0, 0.8};
    decision::APBehavior pressured_behavior;
    decision::Blackboard pressured_blackboard;
    decision::TeamPlan pressured_team_plan;
    pressured_team_plan.tactical_state.possession =
        strategy::PossessionOwner::Theirs;
    pressured_team_plan.tactical_state.phase =
        strategy::TacticalPhase::Defend;
    pressured_team_plan.tactical_state.ball_owner_player_number = 1;
    pressured_team_plan.tactical_state.ball_owner_is_teammate = false;
    pressured_team_plan.tactical_state.nearest_opponent_ball_time_s = 0.0;
    pressured_blackboard.set(
        decision::Blackboard::kKeyTeamPlan, pressured_team_plan);
    const auto pressured_settle = pressured_behavior.make_command(
        pressured_snapshot, pressured_blackboard, role_manager, false, true);
    if (std::holds_alternative<decision::KickCommand>(pressured_settle)) {
        std::cerr << "pressured contact skipped its base-action debounce\n";
        return 1;
    }
    pressured_snapshot.server_time += 0.26;
    const auto pressured_release = pressured_behavior.make_command(
        pressured_snapshot, pressured_blackboard, role_manager, false, true);
    const auto* pressured_contact =
        std::get_if<decision::KickCommand>(&pressured_release);
    if (pressured_contact == nullptr ||
        pressured_contact->mode != decision::KickMode::ForwardContact) {
        std::cerr << "pressured AP did not preserve the base contact path\n";
        return 1;
    }

    // A live match may never settle inside the centimetre-scale procedural
    // slot. After a continuous near-ball attempt, recover the original
    // Apollo walk-through contact explicitly instead of oscillating forever.
    world::WorldSnapshot fallback_snapshot = make_open_pass_snapshot();
    fallback_snapshot.teammates.clear();
    fallback_snapshot.self.position_m = {-0.50, -0.12, 0.8};
    decision::APBehavior fallback_behavior;
    decision::Blackboard fallback_blackboard;
    const auto fallback_setup = fallback_behavior.make_command(
        fallback_snapshot, fallback_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::WalkCommand>(fallback_setup)) {
        std::cerr << "broad contact pose skipped the bounded setup attempt\n";
        return 1;
    }
    for (const double now : {1.20, 1.40}) {
        fallback_snapshot.server_time = now;
        const auto still_setting_up = fallback_behavior.make_command(
            fallback_snapshot, fallback_blackboard, role_manager, false, true);
        if (std::holds_alternative<decision::KickCommand>(still_setting_up)) {
            std::cerr << "forward-contact fallback fired before its timeout\n";
            return 1;
        }
    }
    fallback_snapshot.server_time = 1.46;
    const auto precision_window = fallback_behavior.make_command(
        fallback_snapshot, fallback_blackboard, role_manager, false, true);
    if (std::holds_alternative<decision::KickCommand>(precision_window)) {
        std::cerr << "precision action inherited the legacy fast fallback delay\n";
        return 1;
    }
    fallback_snapshot.server_time = 2.21;
    const auto fallback_release = fallback_behavior.make_command(
        fallback_snapshot, fallback_blackboard, role_manager, false, true);
    const auto* fallback_contact =
        std::get_if<decision::KickCommand>(&fallback_release);
    if (fallback_contact == nullptr ||
        fallback_contact->mode != decision::KickMode::DribbleTouch ||
        !fallback_contact->allow_forward_contact_fallback) {
        std::cerr << "stalled setup did not request an explicit contact fallback"
                  << " variant=" << fallback_release.index();
        if (fallback_blackboard.exists(
                decision::Blackboard::kKeySelectedCooperativeAction)) {
            std::cerr << " selected=" << strategy::to_string(
                fallback_blackboard.get<strategy::CooperativeAction>(
                    decision::Blackboard::kKeySelectedCooperativeAction).category);
        }
        std::cerr << '\n';
        return 1;
    }

    // Preserve the original Apollo tempo when no precision action has been
    // admitted: the longer window above is specific to an explicit local or
    // pass action, not a blanket delay on contested forward contact.
    world::WorldSnapshot legacy_snapshot = make_open_pass_snapshot();
    legacy_snapshot.teammates.clear();
    legacy_snapshot.self.position_m = {-0.50, -0.12, 0.8};
    decision::APBehavior legacy_behavior;
    decision::Blackboard legacy_blackboard;
    (void)legacy_behavior.make_command(
        legacy_snapshot, legacy_blackboard, role_manager, false, false);
    legacy_snapshot.server_time = 1.46;
    const auto legacy_release = legacy_behavior.make_command(
        legacy_snapshot, legacy_blackboard, role_manager, false, false);
    const auto* legacy_contact =
        std::get_if<decision::KickCommand>(&legacy_release);
    if (legacy_contact == nullptr ||
        legacy_contact->mode != decision::KickMode::ForwardContact) {
        std::cerr << "legacy forward contact lost its fast fallback window\n";
        return 1;
    }

    // Switching from the legacy path to a newly admitted local action must
    // start a fresh setup timer even when both requests point in nearly the
    // same direction.
    world::WorldSnapshot switched_snapshot = make_open_pass_snapshot();
    switched_snapshot.teammates.clear();
    switched_snapshot.self.position_m = {-0.50, -0.12, 0.8};
    decision::APBehavior switched_behavior;
    decision::Blackboard switched_blackboard;
    (void)switched_behavior.make_command(
        switched_snapshot, switched_blackboard, role_manager, false, false);
    switched_snapshot.server_time = 1.44;
    (void)switched_behavior.make_command(
        switched_snapshot, switched_blackboard, role_manager, false, false);
    switched_snapshot.server_time = 1.46;
    const auto switched_to_precision = switched_behavior.make_command(
        switched_snapshot, switched_blackboard, role_manager, false, true);
    if (std::holds_alternative<decision::KickCommand>(switched_to_precision)) {
        std::cerr << "new local action inherited a prior forward setup timer\n";
        return 1;
    }

    // A short fallback timer must not steal a valid exact-action slot. Start
    // in the same broad pose, arrive at the procedural slot after the timer,
    // then allow the normal release debounce to complete.
    world::WorldSnapshot preferred_snapshot = make_open_pass_snapshot();
    preferred_snapshot.teammates.clear();
    preferred_snapshot.self.position_m = {-0.50, -0.12, 0.8};
    decision::APBehavior preferred_behavior;
    decision::Blackboard preferred_blackboard;
    (void)preferred_behavior.make_command(
        preferred_snapshot, preferred_blackboard, role_manager, false, true);
    preferred_snapshot.server_time = 1.46;
    preferred_snapshot.self.position_m = {-0.32, -0.04, 0.8};
    const auto preferred_settle = preferred_behavior.make_command(
        preferred_snapshot, preferred_blackboard, role_manager, false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(preferred_settle)) {
        std::cerr << "fallback timer pre-empted exact-action settling\n";
        return 1;
    }
    preferred_snapshot.server_time = 1.51;
    const auto preferred_release = preferred_behavior.make_command(
        preferred_snapshot, preferred_blackboard, role_manager, false, true);
    const auto* preferred_kick =
        std::get_if<decision::KickCommand>(&preferred_release);
    if (preferred_kick == nullptr ||
        preferred_kick->mode != decision::KickMode::DribbleTouch ||
        !preferred_kick->target_point_m.has_value()) {
        std::cerr << "exact procedural release lost to contact fallback\n";
        return 1;
    }

    // At the hard timeout the accepted 25 mm post-entry drift band remains a
    // real procedural slot. The fallback must not use the narrower 20 mm
    // first-entry threshold to steal the two-cycle dynamic-settle debounce.
    world::WorldSnapshot latched_timeout_snapshot = make_open_pass_snapshot();
    latched_timeout_snapshot.teammates.clear();
    latched_timeout_snapshot.self.position_m = {-0.32, -0.04, 0.8};
    latched_timeout_snapshot.self.gyro_deg_s[0] = 40.0;
    decision::APBehavior latched_timeout_behavior;
    decision::Blackboard latched_timeout_blackboard;
    (void)latched_timeout_behavior.make_command(
        latched_timeout_snapshot, latched_timeout_blackboard,
        role_manager, false, true);
    for (const double now : {1.40, 1.80, 2.20, 2.60}) {
        latched_timeout_snapshot.server_time = now;
        const auto dynamic_wait = latched_timeout_behavior.make_command(
            latched_timeout_snapshot, latched_timeout_blackboard,
            role_manager, false, true);
        if (!std::holds_alternative<decision::NeutralCommand>(dynamic_wait)) {
            std::cerr << "dynamic release guard did not survive to hard timeout\n";
            return 1;
        }
    }
    latched_timeout_snapshot.server_time = 2.84;
    latched_timeout_snapshot.self.position_m[1] = -0.0644;
    latched_timeout_snapshot.self.gyro_deg_s[0] = 0.0;
    const auto timeout_settle = latched_timeout_behavior.make_command(
        latched_timeout_snapshot, latched_timeout_blackboard,
        role_manager, false, true);
    if (!std::holds_alternative<decision::NeutralCommand>(timeout_settle)) {
        std::cerr << "fallback stole the latched release slot at hard timeout\n";
        return 1;
    }
    latched_timeout_snapshot.server_time += 0.05;
    const auto timeout_release = latched_timeout_behavior.make_command(
        latched_timeout_snapshot, latched_timeout_blackboard,
        role_manager, false, true);
    if (const auto* timeout_kick =
            std::get_if<decision::KickCommand>(&timeout_release);
        timeout_kick == nullptr ||
        timeout_kick->mode != decision::KickMode::DribbleTouch) {
        std::cerr << "latched release did not complete after hard timeout\n";
        return 1;
    }

    world::WorldSnapshot shot_snapshot = make_open_pass_snapshot();
    shot_snapshot.ball.position_m = {23.5, 0.0, 0.11};
    shot_snapshot.self.position_m = {23.1725, -0.04, 0.8};
    decision::APBehavior shot_behavior;
    decision::Blackboard shot_blackboard;
    const auto shot_stabilizing = shot_behavior.make_command(
        shot_snapshot, shot_blackboard, role_manager, false, true);
    if (std::holds_alternative<decision::KickCommand>(shot_stabilizing) ||
        !shot_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "procedural shot skipped alignment or was not selected\n";
        return 1;
    }
    shot_snapshot.server_time += 0.10;
    const auto shot_release = shot_behavior.make_command(
        shot_snapshot, shot_blackboard, role_manager, false, true);
    const auto* shot_kick = std::get_if<decision::KickCommand>(&shot_release);
    if (shot_kick == nullptr || shot_kick->mode != decision::KickMode::Shot ||
        !shot_kick->target_point_m.has_value() ||
        std::abs(shot_kick->requested_ball_speed_mps - 2.50) > 1.0e-9) {
        std::cerr << "enabled procedural shot did not emit its exact contract\n";
        return 1;
    }

    world::WorldSnapshot clear_snapshot = make_open_pass_snapshot();
    clear_snapshot.ball.position_m = {-20.0, 0.0, 0.11};
    clear_snapshot.self.position_m = {-20.326, -0.04, 0.8};
    decision::APBehavior clear_behavior;
    decision::Blackboard clear_blackboard;
    const auto clear_stabilizing = clear_behavior.make_command(
        clear_snapshot, clear_blackboard, role_manager, false, true);
    if (std::holds_alternative<decision::KickCommand>(clear_stabilizing) ||
        !clear_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "procedural clear skipped alignment or was not selected\n";
        return 1;
    }
    clear_snapshot.server_time += 0.10;
    const auto clear_release = clear_behavior.make_command(
        clear_snapshot, clear_blackboard, role_manager, false, true);
    const auto* clear_kick = std::get_if<decision::KickCommand>(&clear_release);
    if (clear_kick == nullptr || clear_kick->mode != decision::KickMode::Clear ||
        !clear_kick->target_point_m.has_value() ||
        std::abs(clear_kick->requested_ball_speed_mps - 3.50) > 1.0e-9 ||
        std::abs(
            math::planar_dist(
                std::array<double, 2>{
                    clear_snapshot.ball.position_m[0],
                    clear_snapshot.ball.position_m[1]},
                *clear_kick->target_point_m) -
            6.0) > 1.0e-9) {
        std::cerr << "enabled procedural clear did not emit its exact contract\n";
        return 1;
    }

    world::WorldSnapshot goalkeeper_snapshot = clear_snapshot;
    goalkeeper_snapshot.player_number = 1;
    decision::GKBehavior goalkeeper_behavior;
    decision::Blackboard goalkeeper_blackboard;
    goalkeeper_blackboard.set(
        decision::Blackboard::kKeyTacticalTarget,
        decision::TacticalTarget{
            decision::TacticalDuty::GoalkeeperSmother,
            {-20.0, 0.0},
            std::array<double, 2>{-20.0, 0.0},
            0,
            0.9});
    const auto goalkeeper_stabilizing = goalkeeper_behavior.make_command(
        goalkeeper_snapshot, goalkeeper_blackboard, true);
    if (std::holds_alternative<decision::KickCommand>(
            goalkeeper_stabilizing) ||
        !goalkeeper_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction)) {
        std::cerr << "goalkeeper smother did not enter the clear lifecycle\n";
        return 1;
    }
    const std::uint32_t committed_goalkeeper_clear_id =
        goalkeeper_blackboard.get<strategy::CooperativeAction>(
            decision::Blackboard::kKeySelectedCooperativeAction).action_id;
    // Keep the local ball pose unchanged while moving the world position by
    // two centimetres. The planner will generate a different quantized clear
    // target, but the keeper must retain its already admitted action long
    // enough to finish the release debounce.
    goalkeeper_snapshot.server_time += 0.02;
    goalkeeper_snapshot.ball.position_m[0] += 0.02;
    goalkeeper_snapshot.self.position_m[0] += 0.02;
    const auto goalkeeper_committed = goalkeeper_behavior.make_command(
        goalkeeper_snapshot, goalkeeper_blackboard, true);
    if (std::holds_alternative<decision::KickCommand>(goalkeeper_committed) ||
        !goalkeeper_blackboard.exists(
            decision::Blackboard::kKeySelectedCooperativeAction) ||
        goalkeeper_blackboard.get<strategy::CooperativeAction>(
            decision::Blackboard::kKeySelectedCooperativeAction).action_id !=
            committed_goalkeeper_clear_id) {
        std::cerr << "goalkeeper clear was replanned during precision setup\n";
        return 1;
    }
    goalkeeper_snapshot.server_time += 0.08;
    const auto goalkeeper_release = goalkeeper_behavior.make_command(
        goalkeeper_snapshot, goalkeeper_blackboard, true);
    const auto* goalkeeper_clear =
        std::get_if<decision::KickCommand>(&goalkeeper_release);
    if (goalkeeper_clear == nullptr ||
        goalkeeper_clear->mode != decision::KickMode::Clear) {
        std::cerr << "goalkeeper smother did not release a contracted clear\n";
        return 1;
    }

    world::WorldSnapshot emergency_goalkeeper = make_open_pass_snapshot();
    emergency_goalkeeper.player_number = 1;
    emergency_goalkeeper.ball.position_m = {-27.2835, 1.41732, 0.11};
    emergency_goalkeeper.self.position_m = {-26.964, 1.292, 0.8};
    const double emergency_half_yaw_rad = math::deg_to_rad(90.0 * 0.5);
    emergency_goalkeeper.self.orientation_wxyz = {
        std::cos(emergency_half_yaw_rad),
        0.0,
        0.0,
        std::sin(emergency_half_yaw_rad)};
    decision::GKBehavior emergency_goalkeeper_behavior;
    decision::Blackboard emergency_goalkeeper_blackboard;
    emergency_goalkeeper_blackboard.set(
        decision::Blackboard::kKeyTacticalTarget,
        decision::TacticalTarget{
            decision::TacticalDuty::GoalkeeperSmother,
            {-27.2835, 1.41732},
            std::array<double, 2>{-27.2835, 1.41732},
            0,
            1.0});
    const auto emergency_block = emergency_goalkeeper_behavior.make_command(
        emergency_goalkeeper, emergency_goalkeeper_blackboard, true);
    if (!std::holds_alternative<decision::NeutralCommand>(emergency_block)) {
        std::cerr << "goalkeeper abandoned its last-line body block\n";
        return 1;
    }

    // Regression from the 2026-09-06 full match: after closing down a shot,
    // the ball was just behind the torso plane and duty changed back to Hold.
    // The generic turn-first retreat then removed the keeper from the goal
    // path. The body guard must not depend on retaining Smother for one more
    // noisy perception cycle.
    world::WorldSnapshot behind_goalkeeper = make_open_pass_snapshot();
    behind_goalkeeper.player_number = 1;
    behind_goalkeeper.ball.position_m = {-26.7069, -1.14694, 0.11};
    behind_goalkeeper.self.position_m = {-26.214, -1.177, 0.8};
    const double behind_half_yaw_rad = math::deg_to_rad(-86.5087 * 0.5);
    behind_goalkeeper.self.orientation_wxyz = {
        std::cos(behind_half_yaw_rad),
        0.0,
        0.0,
        std::sin(behind_half_yaw_rad)};
    decision::GKBehavior behind_goalkeeper_behavior;
    decision::Blackboard behind_goalkeeper_blackboard;
    behind_goalkeeper_blackboard.set(
        decision::Blackboard::kKeyTacticalTarget,
        decision::TacticalTarget{
            decision::TacticalDuty::GoalkeeperHold,
            {-26.789, -1.02812},
            std::array<double, 2>{
                behind_goalkeeper.ball.position_m[0],
                behind_goalkeeper.ball.position_m[1]},
            0,
            0.9});
    const auto behind_block = behind_goalkeeper_behavior.make_command(
        behind_goalkeeper, behind_goalkeeper_blackboard, true);
    if (!std::holds_alternative<decision::NeutralCommand>(behind_block)) {
        std::cerr << "goalkeeper turned away from a ball behind its body plane\n";
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

    world::WorldSnapshot receive_snapshot = make_open_pass_snapshot();
    receive_snapshot.player_number = 6;
    receive_snapshot.server_time = 30.0;
    receive_snapshot.self.position_m = {1.0, 0.0, 0.8};
    receive_snapshot.teammates.clear();
    receive_snapshot.opponents.clear();
    receive_snapshot.ball.position_m = {0.0, 0.0, 0.11};
    receive_snapshot.ball.velocity_valid = true;
    receive_snapshot.ball.velocity_mps = {2.0, 0.0, 0.0};
    receive_snapshot.team_comm_snapshot.pass_intents.push_back({
        7,
        100,
        comm::PassIntentState::Commanded,
        7,
        6,
        42,
        3.0,
        0.0,
        1.43,
        1.5,
        comm::PassIntentAuthor::Passer,
        6,
    });
    decision::SimpleRoleBehavior receiver_behavior(
        decision::RoleManager::ROLE_ST, false);
    decision::Blackboard receiver_blackboard;
    const auto set_receiver_formation = [&]() {
        receiver_blackboard.set(
            decision::Blackboard::kKeyRolePos,
            std::array<double, 2>{4.0, 2.0});
        receiver_blackboard.set(
            decision::Blackboard::kKeyTacticalTarget,
            decision::TacticalTarget{
                decision::TacticalDuty::Formation,
                {4.0, 2.0},
                std::nullopt,
                0,
                0.5});
    };
    set_receiver_formation();
    const auto dynamic_receive = receiver_behavior.make_command(
        receive_snapshot, receiver_blackboard);
    const auto* receive_walk = std::get_if<decision::WalkCommand>(
        &dynamic_receive);
    const auto& receive_target = receiver_blackboard.get<
        decision::TacticalTarget>(decision::Blackboard::kKeyTacticalTarget);
    if (receive_walk == nullptr ||
        receive_target.duty != decision::TacticalDuty::Receive ||
        receive_target.position_m[0] <= 0.0 ||
        receive_target.position_m[0] >= 2.9) {
        std::cerr << "moving pass did not produce a reachable intercept intent\n";
        return 1;
    }

    receive_snapshot.server_time = 30.1;
    receive_snapshot.team_comm_snapshot.pass_intents.clear();
    set_receiver_formation();
    static_cast<void>(receiver_behavior.make_command(
        receive_snapshot, receiver_blackboard));
    if (receiver_blackboard.get<decision::TacticalTarget>(
            decision::Blackboard::kKeyTacticalTarget).duty !=
        decision::TacticalDuty::Receive) {
        std::cerr << "receive intent did not survive a speech-slot gap\n";
        return 1;
    }

    receive_snapshot.server_time = 30.2;
    receive_snapshot.team_comm_snapshot.pass_intents.push_back({
        7,
        101,
        comm::PassIntentState::Timeout,
        7,
        6,
        42,
        3.0,
        0.0,
        1.43,
        1.5,
        comm::PassIntentAuthor::Passer,
        6,
    });
    set_receiver_formation();
    static_cast<void>(receiver_behavior.make_command(
        receive_snapshot, receiver_blackboard));
    if (receiver_blackboard.get<decision::TacticalTarget>(
            decision::Blackboard::kKeyTacticalTarget).duty ==
        decision::TacticalDuty::Receive) {
        std::cerr << "terminal pass outcome did not cancel receive intent\n";
        return 1;
    }
    return 0;
}
