// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/decision/field_geometry.h"
#include "src/decision/role_manager.h"
#include "src/world/play_mode.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace decision {

/// Coordination-only phases for one awarded team restart. The coordinator
/// never creates a motion command; callers must still apply action capability
/// and legality checks before honoring execution_authorized.
enum class RestartPhase : std::uint8_t {
    Idle,
    Positioning,
    AwaitReady,
    Aligning,
    Executing,
    VerifyRelease,
    TakerLockout,
    Complete,
};

enum class RestartExecutionStatus : std::uint8_t {
    Running,
    Completed,
    Rejected,
    TimedOut,
};

enum class RestartFallbackReason : std::uint8_t {
    None,
    SoftDeadline,
    ExecutionRejected,
    ExecutionTimedOut,
    ReleaseNotObserved,
};

enum class RestartVariant : std::uint8_t {
    Primary,
    Alternate,
    Safety,
};

struct RestartExecutionFeedback {
    std::uint64_t epoch{0U};
    std::uint32_t revision{0U};
    RestartExecutionStatus status{RestartExecutionStatus::Running};
};

/// A frozen coordination plan. receiver_player_number and receiver_target_m
/// identify a player clearing/occupying the contact corridor; they do not claim
/// that a targeted-pass or shot capability exists.
struct RestartPlan {
    world::PlayMode mode{world::PlayMode::NotInitialized};
    std::uint64_t epoch{0U};
    std::uint32_t revision{0U};
    RestartVariant variant{RestartVariant::Primary};
    int taker_player_number{0};
    int receiver_player_number{0};
    field_geometry::Position2 ball_anchor_m{0.0, 0.0};
    field_geometry::Position2 contact_target_m{0.0, 0.0};
    field_geometry::Position2 receiver_target_m{0.0, 0.0};
    double contact_direction_deg{0.0};
    bool ball_anchor_valid{false};
    bool requires_receiver_ready{false};
    bool fallback{false};

    bool executable_coordination() const;
};

struct RestartCoordinatorInput {
    world::PlayMode play_mode{world::PlayMode::NotInitialized};
    double server_time_s{0.0};
    int self_player_number{0};
    /// Optional authoritative restart identifier. A nonzero value permits a
    /// repeated restart of the same mode to be distinguished without relying
    /// on every agent observing the exact transition cycle.
    std::uint64_t restart_epoch{0U};
    field_geometry::Position2 ball_position_m{0.0, 0.0};
    field_geometry::Position2 ball_velocity_mps{0.0, 0.0};
    bool ball_position_valid{false};
    bool ball_velocity_valid{false};
    std::vector<RoleAssignment> role_assignments;
    std::vector<field_geometry::Position2> opponent_positions_m;
    bool team_positioned{false};
    bool receiver_ready{false};
    bool taker_aligned{false};
    bool another_player_touched_ball{false};
    std::optional<RestartExecutionFeedback> execution_feedback;
};

struct RestartCoordinationDecision {
    RestartPhase phase{RestartPhase::Idle};
    std::optional<RestartPlan> plan;
    RestartFallbackReason fallback_reason{RestartFallbackReason::None};
    bool self_is_taker{false};
    bool self_is_receiver{false};
    bool should_position{false};
    bool wait_for_receiver{false};
    bool should_align{false};
    bool execution_authorized{false};
    bool self_locked_out{false};
    bool hard_deadline_reached{false};
};

/// Stateful restart lifecycle intended to be owned by one DecisionManager.
/// Equivalent inputs produce the same frozen taker and direction regardless of
/// RoleAssignment ordering. Cross-agent readiness/feedback consensus remains a
/// caller responsibility until the restart protocol is connected.
class RestartCoordinator {
public:
    struct Parameters {
        double soft_deadline_s{6.0};
        double hard_deadline_s{10.0};
        double release_verification_timeout_s{1.0};
        double release_distance_m{0.35};
        double release_speed_mps{0.30};
        unsigned int release_confirmation_samples{2U};
        double receiver_standoff_m{2.0};
    };

    RestartCoordinator();
    explicit RestartCoordinator(Parameters parameters);

    RestartCoordinationDecision update(const RestartCoordinatorInput& input);
    void reset();

    RestartPhase phase() const { return phase_; }
    const std::optional<RestartPlan>& plan() const { return plan_; }

private:
    Parameters parameters_;
    RestartPhase phase_{RestartPhase::Idle};
    std::optional<RestartPlan> plan_;
    world::PlayMode last_observed_mode_{world::PlayMode::NotInitialized};
    std::uint64_t next_local_epoch_{0U};
    double restart_started_at_s_{0.0};
    double execution_completed_at_s_{0.0};
    unsigned int release_confirmation_count_{0U};
    RestartFallbackReason fallback_reason_{RestartFallbackReason::None};
    bool fallback_used_{false};
    // Once execution has been authorized, conservatively retain the possible
    // first contact across fallback revisions and the OurKick -> PlayOn edge.
    bool execution_authorized_ever_{false};
    bool taker_lockout_released_{false};
    bool hard_deadline_reached_{false};

    void begin_restart(const RestartCoordinatorInput& input);
    void enter_fallback(
        RestartFallbackReason reason,
        const RestartCoordinatorInput& input);
    bool feedback_matches(const RestartExecutionFeedback& feedback) const;
    bool observe_release(const RestartCoordinatorInput& input);
    RestartCoordinationDecision decision_for(int self_player_number) const;
};

bool is_our_restart(world::PlayMode mode);
bool restart_requires_receiver(world::PlayMode mode);
std::optional<double> safe_restart_contact_direction_deg(
    world::PlayMode mode,
    const field_geometry::Position2& ball_position_m);
std::string_view to_string(RestartPhase phase);
std::string_view to_string(RestartExecutionStatus status);
std::string_view to_string(RestartFallbackReason reason);
std::string_view to_string(RestartVariant variant);

}  // namespace decision
