// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/comm/team_comm_types.h"
#include "src/decision/execution_feedback.h"
#include "src/decision/high_level_command.h"
#include "src/strategy/cooperative_action.h"
#include "src/world/world_snapshot.h"

#include <optional>

namespace decision {

/// Persistent, evidence-driven lifecycle for one cooperative pass.
///
/// Motion completion alone is not reported as Executed: the tracker requires
/// observed ball displacement or launch speed, then classifies the physical
/// outcome from receiver/opponent proximity, field bounds, and deadlines.
class PassLifecycle {
public:
    struct Parameters {
        double proposal_timeout_s{6.0};
        double committed_timeout_s{4.0};
        double minimum_execution_displacement_m{0.12};
        double minimum_execution_speed_mps{0.30};
        double receiver_zone_radius_m{1.0};
        double possession_radius_m{0.65};
        double maximum_control_speed_mps{1.5};
        double terminal_broadcast_s{0.8};
    };

    PassLifecycle();
    explicit PassLifecycle(Parameters parameters);

    void start(
        const strategy::CooperativeAction& pass,
        double server_time_s);
    void update(const world::WorldSnapshot& snapshot);
    void mark_commanded(
        const KickCommand& command,
        const world::WorldSnapshot& snapshot);
    void apply_execution_feedback(const ExecutionFeedback& feedback);
    void cancel(double server_time_s);
    void reset();

    bool active() const;
    bool terminal() const;
    bool release_authorized() const;
    bool ready_to_clear(double server_time_s) const;
    comm::PassIntentState state() const;
    const strategy::CooperativeAction* action() const;
    std::optional<comm::OutgoingPassIntent> outgoing() const;

private:
    Parameters parameters_;
    std::optional<strategy::CooperativeAction> action_;
    comm::PassIntentState state_{comm::PassIntentState::Expired};
    double state_since_s_{0.0};
    double deadline_s_{0.0};
    double terminal_until_s_{0.0};
    strategy::Position2 release_ball_position_m_{0.0, 0.0};
    bool motion_completed_{false};

    void transition(comm::PassIntentState next, double server_time_s);
    bool matching_receiver_intent(
        const comm::PassIntentRecord& intent) const;
};

bool is_terminal_pass_state(comm::PassIntentState state);

}  // namespace decision
