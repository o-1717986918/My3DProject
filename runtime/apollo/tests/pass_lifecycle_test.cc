// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/pass_lifecycle.h"

#include <iostream>

namespace {

strategy::CooperativeAction pass_action() {
    strategy::CooperativeAction pass;
    pass.action_id = 91U;
    pass.sequence_id = 7U;
    pass.category = strategy::ActionCategory::Pass;
    pass.pass_type = strategy::PassType::Direct;
    pass.actor_player_number = 7;
    pass.target_player_number = 6;
    pass.start_ball_point_m = {0.0, 0.0};
    pass.target_point_m = {2.0, 0.0};
    pass.requested_ball_speed_mps = 1.43;
    pass.predicted_ball_time_s = 1.5;
    return pass;
}

world::WorldSnapshot snapshot(double now) {
    world::WorldSnapshot result;
    result.player_number = 7;
    result.server_time = now;
    result.play_mode = world::PlayMode::PlayOn;
    result.ball.position_valid = true;
    result.ball.velocity_valid = true;
    result.ball.position_m = {0.0, 0.0, 0.11};
    result.ball.velocity_mps = {0.0, 0.0, 0.0};
    result.teammates.resize(7);
    auto& receiver = result.teammates[5];
    receiver.player_number = 6;
    receiver.seen = true;
    receiver.last_seen_time = now;
    receiver.position_m = {2.0, 0.0, 0.8};
    return result;
}

comm::PassIntentRecord ready_record() {
    comm::PassIntentRecord ready;
    ready.sender_player_number = 6;
    ready.server_cycle = 10;
    ready.state = comm::PassIntentState::Ready;
    ready.passer_player_number = 7;
    ready.receiver_player_number = 6;
    ready.sequence_id = 7U;
    ready.target_x_m = 2.0;
    ready.target_y_m = 0.0;
    ready.requested_ball_speed_mps = 1.43;
    ready.predicted_ball_time_s = 1.5;
    ready.author = comm::PassIntentAuthor::Receiver;
    ready.peer_player_number = 7;
    return ready;
}

decision::KickCommand kick_command() {
    decision::KickCommand kick;
    kick.mode = decision::KickMode::TargetedPass;
    kick.action_id = 91U;
    kick.sequence_id = 7U;
    kick.receiver_player_number = 6;
    kick.target_point_m = std::array<double, 2>{2.0, 0.0};
    kick.requested_ball_speed_mps = 1.43;
    return kick;
}

}  // namespace

int main() {
    decision::PassLifecycle lifecycle;
    lifecycle.start(pass_action(), 1.0);
    if (!lifecycle.active() ||
        lifecycle.state() != comm::PassIntentState::Proposed ||
        lifecycle.release_authorized() ||
        !lifecycle.outgoing().has_value()) {
        std::cerr << "proposal state was not initialized\n";
        return 1;
    }

    auto world = snapshot(1.1);
    world.team_comm_snapshot.pass_intents.push_back(ready_record());
    lifecycle.update(world);
    if (lifecycle.state() != comm::PassIntentState::Committed ||
        !lifecycle.release_authorized() ||
        lifecycle.outgoing()->state != comm::PassIntentState::Committed) {
        std::cerr << "receiver Ready did not commit the pass\n";
        return 1;
    }

    lifecycle.mark_commanded(kick_command(), world);
    decision::ExecutionFeedback completed;
    completed.request_kind = decision::MotionRequestKind::Kick;
    completed.status = decision::ExecutionStatus::Completed;
    completed.server_time = 1.2;
    completed.cooperative_action_id = 91U;
    completed.sequence_id = 7U;
    lifecycle.apply_execution_feedback(completed);
    if (lifecycle.state() != comm::PassIntentState::Commanded) {
        std::cerr << "motion completion was falsely reported as ball execution\n";
        return 1;
    }

    world = snapshot(1.25);
    world.ball.position_m = {0.5, 0.0, 0.11};
    world.ball.velocity_mps = {1.0, 0.0, 0.0};
    lifecycle.update(world);
    if (lifecycle.state() != comm::PassIntentState::Executed) {
        std::cerr << "physical ball launch was not detected\n";
        return 1;
    }

    world.server_time = 1.5;
    world.ball.position_m = {1.25, 0.0, 0.11};
    lifecycle.update(world);
    if (lifecycle.state() != comm::PassIntentState::ReceiverZone) {
        std::cerr << "receiver-zone entry was not detected\n";
        return 1;
    }

    world.server_time = 1.7;
    world.ball.position_m = {1.8, 0.0, 0.11};
    world.ball.velocity_mps = {0.2, 0.0, 0.0};
    lifecycle.update(world);
    if (lifecycle.state() != comm::PassIntentState::Received ||
        !lifecycle.terminal() || lifecycle.ready_to_clear(2.49) ||
        !lifecycle.ready_to_clear(2.51)) {
        std::cerr << "received terminal state retention is incorrect\n";
        return 1;
    }

    decision::PassLifecycle rejected;
    rejected.start(pass_action(), 3.0);
    auto rejected_world = snapshot(3.1);
    rejected_world.team_comm_snapshot.pass_intents.push_back(ready_record());
    rejected.update(rejected_world);
    rejected.mark_commanded(kick_command(), rejected_world);
    completed.status = decision::ExecutionStatus::Rejected;
    completed.server_time = 3.2;
    rejected.apply_execution_feedback(completed);
    if (rejected.state() != comm::PassIntentState::Cancelled) {
        std::cerr << "rejected motion did not cancel the lifecycle\n";
        return 1;
    }

    decision::PassLifecycle no_contact;
    no_contact.start(pass_action(), 4.0);
    auto no_contact_world = snapshot(4.1);
    no_contact_world.team_comm_snapshot.pass_intents.push_back(ready_record());
    no_contact.update(no_contact_world);
    no_contact.mark_commanded(kick_command(), no_contact_world);
    completed.status = decision::ExecutionStatus::Completed;
    completed.server_time = 4.2;
    no_contact.apply_execution_feedback(completed);
    no_contact_world.server_time = 4.61;
    no_contact.update(no_contact_world);
    if (no_contact.state() != comm::PassIntentState::Timeout) {
        std::cerr << "completed motion without ball evidence did not time out\n";
        return 1;
    }

    decision::PassLifecycle intercepted;
    intercepted.start(pass_action(), 5.0);
    auto intercepted_world = snapshot(5.1);
    intercepted_world.team_comm_snapshot.pass_intents.push_back(ready_record());
    intercepted.update(intercepted_world);
    intercepted.mark_commanded(kick_command(), intercepted_world);
    world::PlayerObservation opponent;
    opponent.player_number = 2;
    opponent.seen = true;
    opponent.last_seen_time = 5.2;
    opponent.position_m = {0.5, 0.0, 0.8};
    intercepted_world.opponents.push_back(opponent);
    intercepted_world.server_time = 5.2;
    intercepted_world.ball.position_m = {0.5, 0.0, 0.11};
    intercepted_world.ball.velocity_mps = {0.8, 0.0, 0.0};
    intercepted.update(intercepted_world);
    if (intercepted.state() != comm::PassIntentState::Intercepted) {
        std::cerr << "opponent-first physical evidence did not end the pass\n";
        return 1;
    }

    decision::PassLifecycle out;
    out.start(pass_action(), 6.0);
    auto out_world = snapshot(6.1);
    out_world.team_comm_snapshot.pass_intents.push_back(ready_record());
    out.update(out_world);
    out.mark_commanded(kick_command(), out_world);
    out_world.server_time = 6.2;
    out_world.ball.position_m = {100.0, 0.0, 0.11};
    out_world.ball.velocity_mps = {1.0, 0.0, 0.0};
    out.update(out_world);
    if (out.state() != comm::PassIntentState::Out) {
        std::cerr << "out-of-bounds physical evidence did not end the pass\n";
        return 1;
    }
    return 0;
}
