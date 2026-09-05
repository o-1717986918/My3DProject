// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/pass_lifecycle.h"

#include "src/math/math_utils.h"
#include "src/server/server_constants.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace decision {

namespace {

bool observation_fresh(
    const world::PlayerObservation& player,
    double server_time_s) {
    return player.seen ||
        (player.last_seen_time >= 0.0 &&
         server_time_s - player.last_seen_time <= 1.0);
}

double receiver_ball_distance(
    const world::WorldSnapshot& snapshot,
    int receiver_player_number,
    const strategy::Position2& ball) {
    if (snapshot.player_number == receiver_player_number) {
        return math::planar_dist(
            {snapshot.self.position_m[0], snapshot.self.position_m[1]},
            ball);
    }
    for (const auto& teammate : snapshot.teammates) {
        if (teammate.player_number == receiver_player_number &&
            !teammate.fallen &&
            observation_fresh(teammate, snapshot.server_time)) {
            return math::planar_dist(
                {teammate.position_m[0], teammate.position_m[1]},
                ball);
        }
    }
    return std::numeric_limits<double>::infinity();
}

double nearest_opponent_ball_distance(
    const world::WorldSnapshot& snapshot,
    const strategy::Position2& ball) {
    double nearest = std::numeric_limits<double>::infinity();
    const auto consider = [&](const world::PlayerObservation& opponent) {
        if (opponent.fallen ||
            !observation_fresh(opponent, snapshot.server_time)) {
            return;
        }
        nearest = std::min(
            nearest,
            math::planar_dist(
                {opponent.position_m[0], opponent.position_m[1]},
                ball));
    };
    for (const auto& opponent : snapshot.opponents) consider(opponent);
    for (const auto& opponent : snapshot.shared_opponents) consider(opponent);
    return nearest;
}

}  // namespace

bool is_terminal_pass_state(comm::PassIntentState state) {
    switch (state) {
        case comm::PassIntentState::Received:
        case comm::PassIntentState::Intercepted:
        case comm::PassIntentState::Out:
        case comm::PassIntentState::Timeout:
        case comm::PassIntentState::Cancelled:
        case comm::PassIntentState::Expired:
            return true;
        case comm::PassIntentState::Proposed:
        case comm::PassIntentState::Ready:
        case comm::PassIntentState::Committed:
        case comm::PassIntentState::Commanded:
        case comm::PassIntentState::Executed:
        case comm::PassIntentState::ReceiverZone:
            return false;
    }
    return true;
}

PassLifecycle::PassLifecycle() = default;

PassLifecycle::PassLifecycle(Parameters parameters)
    : parameters_(parameters) {}

void PassLifecycle::start(
    const strategy::CooperativeAction& pass,
    double server_time_s) {
    if (pass.category != strategy::ActionCategory::Pass ||
        pass.actor_player_number <= 0 ||
        pass.target_player_number <= 0 ||
        pass.actor_player_number == pass.target_player_number) {
        reset();
        return;
    }
    action_ = pass;
    state_ = comm::PassIntentState::Proposed;
    state_since_s_ = server_time_s;
    deadline_s_ = server_time_s + parameters_.proposal_timeout_s;
    terminal_until_s_ = 0.0;
    release_ball_position_m_ = pass.start_ball_point_m;
    motion_completed_ = false;
}

bool PassLifecycle::matching_receiver_intent(
    const comm::PassIntentRecord& intent) const {
    return action_.has_value() &&
        intent.author == comm::PassIntentAuthor::Receiver &&
        intent.sender_player_number == action_->target_player_number &&
        intent.passer_player_number == action_->actor_player_number &&
        intent.receiver_player_number == action_->target_player_number &&
        intent.sequence_id == action_->sequence_id;
}

void PassLifecycle::transition(
    comm::PassIntentState next,
    double server_time_s) {
    if (!action_.has_value() || state_ == next) return;
    state_ = next;
    state_since_s_ = server_time_s;
    if (is_terminal_pass_state(next)) {
        terminal_until_s_ =
            server_time_s + parameters_.terminal_broadcast_s;
    }
}

void PassLifecycle::update(const world::WorldSnapshot& snapshot) {
    if (!action_.has_value() || terminal()) return;

    for (const auto& intent : snapshot.team_comm_snapshot.pass_intents) {
        if (!matching_receiver_intent(intent)) continue;
        if (intent.state == comm::PassIntentState::Ready &&
            state_ == comm::PassIntentState::Proposed) {
            transition(comm::PassIntentState::Committed, snapshot.server_time);
            deadline_s_ =
                snapshot.server_time + parameters_.committed_timeout_s;
        } else if (intent.state == comm::PassIntentState::ReceiverZone &&
                   (state_ == comm::PassIntentState::Executed ||
                    state_ == comm::PassIntentState::Commanded)) {
            transition(comm::PassIntentState::ReceiverZone, snapshot.server_time);
        } else if (intent.state == comm::PassIntentState::Received ||
                   intent.state == comm::PassIntentState::Intercepted ||
                   intent.state == comm::PassIntentState::Out ||
                   intent.state == comm::PassIntentState::Timeout ||
                   intent.state == comm::PassIntentState::Cancelled) {
            transition(intent.state, snapshot.server_time);
        }
    }

    if (terminal()) return;
    if ((state_ == comm::PassIntentState::Proposed ||
         state_ == comm::PassIntentState::Committed) &&
        snapshot.server_time >= deadline_s_) {
        transition(comm::PassIntentState::Expired, snapshot.server_time);
        return;
    }
    if (state_ != comm::PassIntentState::Commanded &&
        state_ != comm::PassIntentState::Executed &&
        state_ != comm::PassIntentState::ReceiverZone) {
        return;
    }
    if (!snapshot.ball.position_valid) {
        if (snapshot.server_time >= deadline_s_) {
            transition(comm::PassIntentState::Timeout, snapshot.server_time);
        }
        return;
    }

    const strategy::Position2 ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    if (std::abs(ball[0]) > server_constants::kFieldHalfLengthM ||
        std::abs(ball[1]) > server_constants::kFieldHalfWidthM) {
        transition(comm::PassIntentState::Out, snapshot.server_time);
        return;
    }
    const double ball_speed_mps = snapshot.ball.velocity_valid
        ? math::norm2({
              snapshot.ball.velocity_mps[0],
              snapshot.ball.velocity_mps[1]})
        : 0.0;
    const double displacement_m = math::planar_dist(
        release_ball_position_m_, ball);
    if (state_ == comm::PassIntentState::Commanded &&
        (displacement_m >= parameters_.minimum_execution_displacement_m ||
         ball_speed_mps >= parameters_.minimum_execution_speed_mps)) {
        transition(comm::PassIntentState::Executed, snapshot.server_time);
    }
    if (state_ == comm::PassIntentState::Commanded) {
        if ((motion_completed_ &&
             snapshot.server_time - state_since_s_ >= 0.50) ||
            snapshot.server_time >= deadline_s_) {
            transition(comm::PassIntentState::Timeout, snapshot.server_time);
        }
        return;
    }

    const double receiver_distance_m = receiver_ball_distance(
        snapshot, action_->target_player_number, ball);
    const double opponent_distance_m = nearest_opponent_ball_distance(
        snapshot, ball);
    if (opponent_distance_m <= parameters_.possession_radius_m &&
        opponent_distance_m + 0.10 < receiver_distance_m) {
        transition(comm::PassIntentState::Intercepted, snapshot.server_time);
        return;
    }
    if (receiver_distance_m <= parameters_.possession_radius_m &&
        ball_speed_mps <= parameters_.maximum_control_speed_mps) {
        transition(comm::PassIntentState::Received, snapshot.server_time);
        return;
    }
    if (math::planar_dist(ball, action_->target_point_m) <=
        parameters_.receiver_zone_radius_m) {
        transition(comm::PassIntentState::ReceiverZone, snapshot.server_time);
    }
    if (snapshot.server_time >= deadline_s_) {
        transition(comm::PassIntentState::Timeout, snapshot.server_time);
    }
}

void PassLifecycle::mark_commanded(
    const KickCommand& command,
    const world::WorldSnapshot& snapshot) {
    if (!action_.has_value() || terminal() ||
        state_ == comm::PassIntentState::Commanded ||
        state_ == comm::PassIntentState::Executed ||
        state_ == comm::PassIntentState::ReceiverZone ||
        command.mode != KickMode::TargetedPass ||
        command.action_id != action_->action_id ||
        command.sequence_id != action_->sequence_id) {
        return;
    }
    release_ball_position_m_ = snapshot.ball.position_valid
        ? strategy::Position2{
              snapshot.ball.position_m[0], snapshot.ball.position_m[1]}
        : action_->start_ball_point_m;
    transition(comm::PassIntentState::Commanded, snapshot.server_time);
    deadline_s_ = snapshot.server_time + std::max(
        parameters_.committed_timeout_s,
        action_->predicted_ball_time_s + 2.0);
}

void PassLifecycle::apply_execution_feedback(
    const ExecutionFeedback& feedback) {
    if (!action_.has_value() || terminal() ||
        feedback.request_kind != MotionRequestKind::Kick ||
        !feedback.cooperative_action_id.has_value() ||
        !feedback.sequence_id.has_value() ||
        *feedback.cooperative_action_id != action_->action_id ||
        *feedback.sequence_id != action_->sequence_id) {
        return;
    }
    if (feedback.status == ExecutionStatus::Rejected) {
        transition(comm::PassIntentState::Cancelled, feedback.server_time);
    } else if (feedback.status == ExecutionStatus::TimedOut) {
        transition(comm::PassIntentState::Timeout, feedback.server_time);
    } else if (feedback.status == ExecutionStatus::Completed) {
        // Completion proves the motor request ended, not that the ball moved.
        motion_completed_ = true;
    }
}

void PassLifecycle::cancel(double server_time_s) {
    if (action_.has_value() && !terminal()) {
        transition(comm::PassIntentState::Cancelled, server_time_s);
    }
}

void PassLifecycle::reset() {
    action_.reset();
    state_ = comm::PassIntentState::Expired;
    state_since_s_ = 0.0;
    deadline_s_ = 0.0;
    terminal_until_s_ = 0.0;
    release_ball_position_m_ = {0.0, 0.0};
    motion_completed_ = false;
}

bool PassLifecycle::active() const {
    return action_.has_value();
}

bool PassLifecycle::terminal() const {
    return action_.has_value() && is_terminal_pass_state(state_);
}

bool PassLifecycle::release_authorized() const {
    return action_.has_value() &&
        state_ == comm::PassIntentState::Committed;
}

bool PassLifecycle::ready_to_clear(double server_time_s) const {
    return terminal() && server_time_s >= terminal_until_s_;
}

comm::PassIntentState PassLifecycle::state() const {
    return state_;
}

const strategy::CooperativeAction* PassLifecycle::action() const {
    return action_.has_value() ? &*action_ : nullptr;
}

std::optional<comm::OutgoingPassIntent> PassLifecycle::outgoing() const {
    if (!action_.has_value()) return std::nullopt;
    return comm::OutgoingPassIntent{
        state_,
        comm::PassIntentAuthor::Passer,
        action_->actor_player_number,
        action_->target_player_number,
        action_->sequence_id,
        action_->target_point_m[0],
        action_->target_point_m[1],
        action_->requested_ball_speed_mps,
        action_->predicted_ball_time_s,
    };
}

}  // namespace decision
