// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/decision_manager.h"
#include "src/decision/role_behaviors.h"

#include <cstdint>
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
        player.seen = true;
        player.last_seen_time = snapshot.server_time;
        player.position_m = {-8.0, static_cast<double>(number), 0.8};
        player.fallen = number >= 2 && number <= 5;
    }
    snapshot.teammates[0].position_m = {-14.0, 0.0, 0.8};
    snapshot.teammates[5].position_m = {1.0, 0.0, 0.8};
    snapshot.teammates[6].position_m = snapshot.self.position_m;
    return snapshot;
}

void add_ready_intent(
    world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& selected) {
    comm::PassIntentRecord ready;
    ready.sender_player_number = selected.target_player_number;
    ready.server_cycle = 51;
    ready.state = comm::PassIntentState::Ready;
    ready.passer_player_number = snapshot.player_number;
    ready.receiver_player_number = selected.target_player_number;
    ready.sequence_id = selected.sequence_id;
    ready.target_x_m = selected.target_point_m[0];
    ready.target_y_m = selected.target_point_m[1];
    ready.requested_ball_speed_mps = selected.requested_ball_speed_mps;
    ready.predicted_ball_time_s = selected.predicted_ball_time_s;
    ready.author = comm::PassIntentAuthor::Receiver;
    ready.peer_player_number = snapshot.player_number;
    snapshot.team_comm_snapshot.pass_intents.push_back(ready);
}

bool establish_targeted_pass(
    decision::DecisionManager& manager,
    world::WorldSnapshot& snapshot,
    strategy::CooperativeAction* selected,
    decision::KickCommand* kick) {
    const auto waiting = manager.decide(snapshot);
    const auto* committed = manager.selected_cooperative_action();
    if (!std::holds_alternative<decision::WalkCommand>(waiting) ||
        committed == nullptr) {
        std::cerr << "decision manager did not establish a pass commitment\n";
        return false;
    }
    *selected = *committed;
    add_ready_intent(snapshot, *selected);

    snapshot.server_time = 1.01;
    const auto stabilizing = manager.decide(snapshot);
    if (std::holds_alternative<decision::KickCommand>(stabilizing)) {
        std::cerr << "pass released before the contact pose debounce elapsed\n";
        return false;
    }

    snapshot.server_time = 1.30;
    const auto released = manager.decide(snapshot);
    const auto* released_kick = std::get_if<decision::KickCommand>(&released);
    if (released_kick == nullptr ||
        released_kick->mode != decision::KickMode::TargetedPass ||
        released_kick->action_id != selected->action_id ||
        released_kick->sequence_id != selected->sequence_id) {
        std::cerr << "decision manager did not release the committed pass\n";
        return false;
    }
    *kick = *released_kick;
    return true;
}

decision::ExecutionFeedback make_kick_feedback(
    decision::ExecutionStatus status,
    const decision::KickCommand& kick,
    std::uint64_t request_id,
    double server_time) {
    decision::ExecutionFeedback feedback;
    feedback.request_id = request_id;
    feedback.server_time = server_time;
    feedback.status = status;
    feedback.request_kind = decision::MotionRequestKind::Kick;
    feedback.cooperative_action_id = kick.action_id;
    feedback.sequence_id = kick.sequence_id;
    return feedback;
}

bool active_pass_is_retained(
    const decision::HighLevelCommand& command,
    const decision::DecisionManager& manager,
    const decision::KickCommand& expected) {
    const auto* kick = std::get_if<decision::KickCommand>(&command);
    const auto* selected = manager.selected_cooperative_action();
    return kick != nullptr &&
           kick->mode == decision::KickMode::TargetedPass &&
           kick->action_id == expected.action_id &&
           kick->sequence_id == expected.sequence_id &&
           selected != nullptr &&
           selected->action_id == expected.action_id &&
           selected->sequence_id == expected.sequence_id;
}

bool test_failure_cancels_and_replans(decision::ExecutionStatus status) {
    decision::DecisionManager manager(true, true);
    world::WorldSnapshot snapshot = make_open_pass_snapshot();
    strategy::CooperativeAction selected;
    decision::KickCommand kick;
    if (!establish_targeted_pass(manager, snapshot, &selected, &kick)) {
        return false;
    }

    // Repeating the decision without motion feedback must retain the active
    // request. Issuing KickCommand is not execution completion.
    snapshot.server_time = 1.31;
    const auto still_running = manager.decide(snapshot);
    const auto* running_kick = std::get_if<decision::KickCommand>(&still_running);
    if (running_kick == nullptr ||
        running_kick->action_id != kick.action_id ||
        manager.selected_cooperative_action() == nullptr) {
        std::cerr << "issuing a kick was incorrectly treated as completion\n";
        return false;
    }

    snapshot.server_time = 1.32;
    const auto failed_command = manager.decide(
        snapshot,
        make_kick_feedback(status, kick, 100U, 1.31));
    const auto* terminal_action = manager.selected_cooperative_action();
    const auto* terminal_intent = manager.outgoing_pass_intent();
    const auto expected_terminal = status == decision::ExecutionStatus::Rejected
        ? comm::PassIntentState::Cancelled
        : comm::PassIntentState::Timeout;
    if (terminal_action == nullptr || terminal_intent == nullptr ||
        terminal_action->action_id == kick.action_id ||
        terminal_intent->state != expected_terminal) {
        std::cerr << "matching failed kick did not publish a terminal outcome"
                  << " terminal_action=" << (terminal_action != nullptr)
                  << " terminal_intent=" << (terminal_intent != nullptr);
        if (terminal_action != nullptr) {
            std::cerr << " action_id=" << terminal_action->action_id
                      << " failed_action_id=" << kick.action_id
                      << " category="
                      << static_cast<int>(terminal_action->category);
        }
        if (terminal_intent != nullptr) {
            std::cerr << " intent_state="
                      << static_cast<int>(terminal_intent->state)
                      << " expected_state="
                      << static_cast<int>(expected_terminal);
        }
        std::cerr << '\n';
        return false;
    }
    if (manager.strategy_plan() == nullptr ||
        !manager.strategy_plan()->selected.has_value()) {
        std::cerr << "failed kick did not trigger a fresh strategy plan\n";
        return false;
    }
    if (const auto* failed_kick =
            std::get_if<decision::KickCommand>(&failed_command);
        failed_kick != nullptr &&
        failed_kick->mode == decision::KickMode::TargetedPass &&
        failed_kick->action_id == kick.action_id &&
        failed_kick->sequence_id == kick.sequence_id) {
        std::cerr << "failed targeted pass remained active\n";
        return false;
    }

    // The terminal outcome remains visible long enough to cross at least two
    // team-speech slots. It is not immediately resubmitted; after retention
    // and the retry delay, a new sequence commits.
    snapshot.server_time = 1.82;
    manager.decide(snapshot);
    const auto* retained = manager.outgoing_pass_intent();
    if (retained == nullptr || retained->state != expected_terminal) {
        std::cerr << "terminal pass outcome was not retained for broadcast\n";
        return false;
    }
    snapshot.server_time = 3.40;
    manager.decide(snapshot);
    const auto* retry = manager.selected_cooperative_action();
    if (retry == nullptr ||
        retry->category != strategy::ActionCategory::Pass ||
        retry->sequence_id == selected.sequence_id) {
        std::cerr << "pass was not recommitted with a new sequence after retry delay\n";
        return false;
    }
    return true;
}

bool test_stale_failure_is_ignored() {
    decision::DecisionManager manager(true, true);
    world::WorldSnapshot snapshot = make_open_pass_snapshot();
    strategy::CooperativeAction selected;
    decision::KickCommand kick;
    if (!establish_targeted_pass(manager, snapshot, &selected, &kick)) {
        return false;
    }

    snapshot.server_time = 1.31;
    const auto running = manager.decide(
        snapshot,
        make_kick_feedback(
            decision::ExecutionStatus::Running, kick, 200U, 1.30));
    if (!active_pass_is_retained(running, manager, kick)) {
        std::cerr << "running feedback cancelled the active pass\n";
        return false;
    }

    snapshot.server_time = 1.32;
    const auto completed = manager.decide(
        snapshot,
        make_kick_feedback(
            decision::ExecutionStatus::Completed, kick, 201U, 1.31));
    if (!active_pass_is_retained(completed, manager, kick)) {
        std::cerr << "completed feedback was misclassified as a failed pass\n";
        return false;
    }

    snapshot.server_time = 1.33;
    auto wrong_action = make_kick_feedback(
        decision::ExecutionStatus::Rejected, kick, 202U, 1.32);
    wrong_action.cooperative_action_id = kick.action_id ^ 1U;
    const auto command_after_wrong_action = manager.decide(snapshot, wrong_action);
    if (!active_pass_is_retained(command_after_wrong_action, manager, kick)) {
        std::cerr << "stale action feedback cancelled the active pass\n";
        return false;
    }

    snapshot.server_time = 1.34;
    auto wrong_sequence = make_kick_feedback(
        decision::ExecutionStatus::TimedOut, kick, 203U, 1.33);
    wrong_sequence.sequence_id = static_cast<std::uint8_t>(kick.sequence_id + 1U);
    const auto command_after_wrong_sequence =
        manager.decide(snapshot, wrong_sequence);
    if (!active_pass_is_retained(command_after_wrong_sequence, manager, kick)) {
        std::cerr << "stale sequence feedback cancelled the active pass\n";
        return false;
    }

    snapshot.server_time = 1.35;
    auto wrong_kind = make_kick_feedback(
        decision::ExecutionStatus::Rejected, kick, 204U, 1.34);
    wrong_kind.request_kind = decision::MotionRequestKind::Walk;
    const auto command_after_wrong_kind = manager.decide(snapshot, wrong_kind);
    if (!active_pass_is_retained(command_after_wrong_kind, manager, kick)) {
        std::cerr << "non-kick feedback cancelled the active pass\n";
        return false;
    }
    return true;
}

bool test_decision_instances_do_not_share_active_kicks() {
    decision::DecisionManager first(true, true);
    world::WorldSnapshot snapshot = make_open_pass_snapshot();
    strategy::CooperativeAction selected;
    decision::KickCommand kick;
    if (!establish_targeted_pass(first, snapshot, &selected, &kick)) {
        return false;
    }

    decision::DecisionManager second(true, true);
    snapshot.team_comm_snapshot.pass_intents.clear();
    snapshot.server_time = 1.31;
    const auto independent = second.decide(snapshot);
    if (std::holds_alternative<decision::KickCommand>(independent)) {
        std::cerr << "active kick leaked between decision-manager instances\n";
        return false;
    }
    const auto* second_selected = second.selected_cooperative_action();
    if (second_selected == nullptr ||
        second_selected->sequence_id != 1U) {
        std::cerr << "independent decision manager did not start fresh state\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!test_failure_cancels_and_replans(
            decision::ExecutionStatus::Rejected)) {
        return 1;
    }
    if (!test_failure_cancels_and_replans(
            decision::ExecutionStatus::TimedOut)) {
        return 1;
    }
    if (!test_stale_failure_is_ignored()) {
        return 1;
    }
    if (!test_decision_instances_do_not_share_active_kicks()) {
        return 1;
    }
    return 0;
}
