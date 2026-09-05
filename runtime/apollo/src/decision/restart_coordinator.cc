// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/restart_coordinator.h"

#include "src/math/math_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace decision {

namespace {

using field_geometry::Position2;

int player_for_role(
    const std::vector<RoleAssignment>& assignments,
    int role_id,
    int excluded_player_number = 0) {
    int selected = 0;
    for (const auto& assignment : assignments) {
        if (assignment.role_id != role_id || assignment.player_number <= 0 ||
            assignment.player_number == excluded_player_number) {
            continue;
        }
        if (selected == 0 || assignment.player_number < selected) {
            selected = assignment.player_number;
        }
    }
    return selected;
}

std::uint64_t next_nonzero_epoch(std::uint64_t* epoch) {
    if (epoch == nullptr) return 1U;
    ++(*epoch);
    if (*epoch == 0U) *epoch = 1U;
    return *epoch;
}

Position2 unit_direction(double direction_deg) {
    const double radians = math::deg_to_rad(direction_deg);
    return {std::cos(radians), std::sin(radians)};
}

Position2 receiver_target(
    const Position2& ball,
    double direction_deg,
    double standoff_m) {
    const Position2 direction = unit_direction(direction_deg);
    Position2 target{
        ball[0] + direction[0] * standoff_m,
        ball[1] + direction[1] * standoff_m,
    };
    constexpr double margin = field_geometry::kFormationFieldMarginM;
    target[0] = std::clamp(
        target[0], -field_geometry::kActualHalfLengthM + margin,
        field_geometry::kActualHalfLengthM - margin);
    target[1] = std::clamp(
        target[1], -field_geometry::kActualHalfWidthM + margin,
        field_geometry::kActualHalfWidthM - margin);
    return target;
}

bool finite_position(const Position2& position) {
    return std::isfinite(position[0]) && std::isfinite(position[1]);
}

bool inside_actual_field(const Position2& position) {
    return finite_position(position) &&
        std::abs(position[0]) <= field_geometry::kActualHalfLengthM &&
        std::abs(position[1]) <= field_geometry::kActualHalfWidthM;
}

bool valid_parameters(const RestartCoordinator::Parameters& parameters) {
    return std::isfinite(parameters.soft_deadline_s) &&
        std::isfinite(parameters.hard_deadline_s) &&
        std::isfinite(parameters.release_verification_timeout_s) &&
        std::isfinite(parameters.release_distance_m) &&
        std::isfinite(parameters.release_speed_mps) &&
        std::isfinite(parameters.receiver_standoff_m) &&
        parameters.soft_deadline_s > 0.0 &&
        parameters.hard_deadline_s > parameters.soft_deadline_s &&
        parameters.release_verification_timeout_s > 0.0 &&
        parameters.release_distance_m > 0.0 &&
        parameters.release_speed_mps >= 0.0 &&
        parameters.release_confirmation_samples > 0U &&
        parameters.receiver_standoff_m > 0.0;
}

bool has_restart_alternate(world::PlayMode mode) {
    return mode != world::PlayMode::OurKickOff;
}

Position2 restart_contact_target(
    world::PlayMode mode,
    const Position2& ball,
    RestartVariant variant) {
    const double half_length = field_geometry::kActualHalfLengthM;
    const double half_width = field_geometry::kActualHalfWidthM;
    const double side = ball[1] < 0.0 ? -1.0 : 1.0;
    Position2 target{half_length, 0.0};

    if (variant == RestartVariant::Safety) {
        if (mode == world::PlayMode::OurCornerKick) {
            target = {half_length - 7.0, 0.0};
        } else {
            target = {
                std::clamp(ball[0] + 4.0, -half_length + 1.0, half_length - 1.0),
                ball[1] - side * std::min(4.0, std::abs(ball[1]))};
        }
    } else if (mode == world::PlayMode::OurKickOff) {
        target = {5.0, 0.0};
    } else if (mode == world::PlayMode::OurGoalKick) {
        target = {
            std::min(ball[0] + 7.0, half_length - 1.0),
            variant == RestartVariant::Primary ? 4.0 : -4.0};
    } else if (mode == world::PlayMode::OurPenaltyKick ||
               mode == world::PlayMode::OurPenaltyShoot) {
        target = {
            half_length,
            variant == RestartVariant::Primary ? 0.8 : -0.8};
    } else if (mode == world::PlayMode::OurCornerKick) {
        target = variant == RestartVariant::Primary
            ? Position2{
                  half_length - 4.0,
                  side * (half_width - 6.0)}
            : Position2{half_length - 7.0, 0.0};
    } else if (mode == world::PlayMode::OurThrowIn) {
        target = variant == RestartVariant::Primary
            ? Position2{
                  std::clamp(ball[0] + 4.0, -half_length + 1.0, half_length - 1.0),
                  side * std::max(0.0, std::abs(ball[1]) - 4.0)}
            : Position2{
                  std::clamp(ball[0] + 1.5, -half_length + 1.0, half_length - 1.0),
                  side * std::max(0.0, std::abs(ball[1]) - 6.0)};
    } else if (variant == RestartVariant::Alternate) {
        target = {
            std::clamp(ball[0] + 5.0, -half_length + 1.0, half_length - 1.0),
            std::abs(ball[1]) < 1.0
                ? 4.0
                : ball[1] - side * std::min(4.0, std::abs(ball[1]))};
    }

    target[0] = std::clamp(target[0], -half_length, half_length);
    target[1] = std::clamp(target[1], -half_width, half_width);
    return target;
}

double restart_lane_clearance(
    const Position2& ball,
    const Position2& target,
    const std::vector<Position2>& opponents) {
    double clearance = std::numeric_limits<double>::infinity();
    for (const auto& opponent : opponents) {
        if (!std::isfinite(opponent[0]) || !std::isfinite(opponent[1])) continue;
        clearance = std::min(
            clearance,
            math::point_segment_distance(opponent, ball, target));
    }
    return clearance;
}

RestartVariant select_restart_variant(
    const RestartCoordinatorInput& input,
    std::uint64_t epoch) {
    if (!has_restart_alternate(input.play_mode)) {
        return RestartVariant::Primary;
    }
    const Position2 primary = restart_contact_target(
        input.play_mode, input.ball_position_m, RestartVariant::Primary);
    const Position2 alternate = restart_contact_target(
        input.play_mode, input.ball_position_m, RestartVariant::Alternate);
    const double primary_clearance = restart_lane_clearance(
        input.ball_position_m, primary, input.opponent_positions_m);
    const double alternate_clearance = restart_lane_clearance(
        input.ball_position_m, alternate, input.opponent_positions_m);
    if (alternate_clearance > primary_clearance + 0.25) {
        return RestartVariant::Alternate;
    }
    if (primary_clearance > alternate_clearance + 0.25) {
        return RestartVariant::Primary;
    }
    return epoch % 2U == 0U
        ? RestartVariant::Alternate
        : RestartVariant::Primary;
}

}  // namespace

bool RestartPlan::executable_coordination() const {
    return is_our_restart(mode) && epoch != 0U && revision != 0U &&
        taker_player_number > 0 && ball_anchor_valid &&
        inside_actual_field(ball_anchor_m) &&
        inside_actual_field(contact_target_m) &&
        inside_actual_field(receiver_target_m) &&
        std::isfinite(contact_direction_deg) &&
        (!requires_receiver_ready || receiver_player_number > 0);
}

RestartCoordinator::RestartCoordinator()
    : RestartCoordinator(Parameters{}) {}

RestartCoordinator::RestartCoordinator(Parameters parameters)
    : parameters_(parameters) {
    if (!valid_parameters(parameters_)) {
        throw std::invalid_argument("invalid restart coordinator parameters");
    }
}

void RestartCoordinator::reset() {
    phase_ = RestartPhase::Idle;
    plan_.reset();
    last_observed_mode_ = world::PlayMode::NotInitialized;
    restart_started_at_s_ = 0.0;
    execution_completed_at_s_ = 0.0;
    release_confirmation_count_ = 0U;
    fallback_reason_ = RestartFallbackReason::None;
    fallback_used_ = false;
    execution_authorized_ever_ = false;
    taker_lockout_released_ = false;
    hard_deadline_reached_ = false;
}

void RestartCoordinator::begin_restart(const RestartCoordinatorInput& input) {
    const int taker_role = input.play_mode == world::PlayMode::OurGoalKick
        ? RoleManager::ROLE_GK
        : RoleManager::ROLE_AP;
    const int taker = player_for_role(input.role_assignments, taker_role);
    const std::uint64_t epoch = input.restart_epoch != 0U
        ? input.restart_epoch
        : next_nonzero_epoch(&next_local_epoch_);
    next_local_epoch_ = std::max(next_local_epoch_, epoch);
    const bool finite_ball = input.ball_position_valid &&
        inside_actual_field(input.ball_position_m);
    const RestartVariant variant = finite_ball
        ? select_restart_variant(input, epoch)
        : RestartVariant::Primary;
    const bool needs_receiver = restart_requires_receiver(input.play_mode);
    const int receiver = needs_receiver
        ? player_for_role(input.role_assignments, RoleManager::ROLE_ST, taker)
        : 0;

    const Position2 contact_target = finite_ball
        ? restart_contact_target(input.play_mode, input.ball_position_m, variant)
        : Position2{0.0, 0.0};
    const Position2 contact_delta = math::vec2_sub(
        contact_target, input.ball_position_m);
    const std::optional<double> direction = finite_ball &&
        finite_position(contact_target) && math::norm2(contact_delta) > 1.0e-6
        ? std::optional<double>{math::vector_angle_deg(contact_delta)}
        : std::nullopt;
    RestartPlan plan;
    plan.mode = input.play_mode;
    plan.epoch = epoch;
    plan.revision = 1U;
    plan.variant = variant;
    plan.taker_player_number = taker;
    plan.receiver_player_number = receiver;
    plan.ball_anchor_m = finite_ball
        ? input.ball_position_m
        : Position2{0.0, 0.0};
    plan.contact_target_m = contact_target;
    plan.contact_direction_deg = direction.value_or(0.0);
    plan.ball_anchor_valid = finite_ball && direction.has_value();
    plan.requires_receiver_ready = needs_receiver;
    plan.receiver_target_m = plan.ball_anchor_valid
        ? receiver_target(
              plan.ball_anchor_m, plan.contact_direction_deg,
              parameters_.receiver_standoff_m)
        : Position2{0.0, 0.0};
    plan_ = plan;

    phase_ = RestartPhase::Positioning;
    restart_started_at_s_ = input.server_time_s;
    execution_completed_at_s_ = 0.0;
    release_confirmation_count_ = 0U;
    fallback_reason_ = RestartFallbackReason::None;
    fallback_used_ = false;
    execution_authorized_ever_ = false;
    taker_lockout_released_ = false;
    hard_deadline_reached_ = false;
}

void RestartCoordinator::enter_fallback(
    RestartFallbackReason reason,
    const RestartCoordinatorInput& input) {
    if (!plan_.has_value() || fallback_used_) {
        phase_ = RestartPhase::Complete;
        return;
    }
    fallback_used_ = true;
    fallback_reason_ = reason;
    ++plan_->revision;
    if (plan_->revision == 0U) plan_->revision = 1U;
    plan_->fallback = true;
    plan_->variant = RestartVariant::Safety;
    plan_->requires_receiver_ready = false;
    plan_->receiver_player_number = 0;
    if (input.ball_position_valid && inside_actual_field(input.ball_position_m)) {
        const Position2 contact_target = restart_contact_target(
            plan_->mode, input.ball_position_m, RestartVariant::Safety);
        const Position2 delta = math::vec2_sub(
            contact_target, input.ball_position_m);
        if (math::norm2(delta) > 1.0e-6) {
            plan_->ball_anchor_m = input.ball_position_m;
            plan_->contact_target_m = contact_target;
            plan_->contact_direction_deg = math::vector_angle_deg(delta);
            plan_->ball_anchor_valid = true;
        }
    }
    plan_->receiver_target_m = receiver_target(
        plan_->ball_anchor_m, plan_->contact_direction_deg,
        parameters_.receiver_standoff_m);
    phase_ = RestartPhase::Aligning;
    execution_completed_at_s_ = 0.0;
    release_confirmation_count_ = 0U;
}

bool RestartCoordinator::feedback_matches(
    const RestartExecutionFeedback& feedback) const {
    return plan_.has_value() && feedback.epoch == plan_->epoch &&
        feedback.revision == plan_->revision;
}

bool RestartCoordinator::observe_release(const RestartCoordinatorInput& input) {
    if (!plan_.has_value() || !input.ball_position_valid ||
        !input.ball_velocity_valid) {
        release_confirmation_count_ = 0U;
        return false;
    }
    const double displacement = math::planar_dist(
        input.ball_position_m, plan_->ball_anchor_m);
    const double speed = math::norm2(input.ball_velocity_mps);
    if (!std::isfinite(displacement) || !std::isfinite(speed) ||
        displacement < parameters_.release_distance_m ||
        speed < parameters_.release_speed_mps) {
        release_confirmation_count_ = 0U;
        return false;
    }
    ++release_confirmation_count_;
    return release_confirmation_count_ >=
        parameters_.release_confirmation_samples;
}

RestartCoordinationDecision RestartCoordinator::decision_for(
    int self_player_number) const {
    RestartCoordinationDecision decision;
    decision.phase = phase_;
    decision.plan = plan_;
    decision.fallback_reason = fallback_reason_;
    decision.hard_deadline_reached = hard_deadline_reached_;
    if (!plan_.has_value()) return decision;

    decision.self_is_taker =
        self_player_number > 0 && self_player_number == plan_->taker_player_number;
    decision.self_is_receiver = self_player_number > 0 &&
        self_player_number == plan_->receiver_player_number;
    decision.should_position = phase_ == RestartPhase::Positioning ||
        phase_ == RestartPhase::AwaitReady;
    decision.wait_for_receiver = phase_ == RestartPhase::AwaitReady;
    decision.should_align = phase_ == RestartPhase::Aligning &&
        decision.self_is_taker;
    decision.execution_authorized = phase_ == RestartPhase::Executing &&
        decision.self_is_taker && plan_->executable_coordination();
    decision.self_locked_out = phase_ == RestartPhase::TakerLockout &&
        decision.self_is_taker;
    return decision;
}

RestartCoordinationDecision RestartCoordinator::update(
    const RestartCoordinatorInput& input) {
    if (!std::isfinite(input.server_time_s)) {
        return decision_for(input.self_player_number);
    }

    const bool our_restart = is_our_restart(input.play_mode);
    const bool mode_entry = our_restart &&
        input.play_mode != last_observed_mode_;
    const bool authoritative_new_epoch = our_restart &&
        input.restart_epoch != 0U &&
        (!plan_.has_value() || input.restart_epoch > plan_->epoch);
    if (mode_entry || authoritative_new_epoch) {
        begin_restart(input);
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    if (!plan_.has_value()) {
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    if (input.play_mode == world::PlayMode::PlayOn) {
        if (input.another_player_touched_ball) {
            taker_lockout_released_ = true;
            phase_ = RestartPhase::Complete;
        } else if (taker_lockout_released_) {
            phase_ = RestartPhase::Complete;
        } else if (execution_authorized_ever_ ||
                   phase_ == RestartPhase::VerifyRelease ||
                   phase_ == RestartPhase::TakerLockout) {
            phase_ = RestartPhase::TakerLockout;
        } else if (phase_ != RestartPhase::Complete) {
            phase_ = RestartPhase::Complete;
        }
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    if (!our_restart || input.play_mode != plan_->mode) {
        phase_ = RestartPhase::Complete;
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    if (execution_authorized_ever_ && input.another_player_touched_ball) {
        taker_lockout_released_ = true;
        phase_ = RestartPhase::Complete;
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    if (phase_ == RestartPhase::TakerLockout) {
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    const double elapsed = std::max(0.0, input.server_time_s - restart_started_at_s_);
    if (phase_ != RestartPhase::Complete && elapsed >= parameters_.hard_deadline_s) {
        phase_ = RestartPhase::Complete;
        hard_deadline_reached_ = true;
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    const bool before_execution = phase_ == RestartPhase::Positioning ||
        phase_ == RestartPhase::AwaitReady || phase_ == RestartPhase::Aligning;
    if (before_execution && elapsed >= parameters_.soft_deadline_s &&
        !fallback_used_) {
        enter_fallback(RestartFallbackReason::SoftDeadline, input);
        last_observed_mode_ = input.play_mode;
        return decision_for(input.self_player_number);
    }

    switch (phase_) {
        case RestartPhase::Positioning:
            if (input.team_positioned && plan_->executable_coordination()) {
                if (plan_->requires_receiver_ready) {
                    phase_ = RestartPhase::AwaitReady;
                } else if (input.taker_aligned) {
                    phase_ = RestartPhase::Executing;
                    execution_authorized_ever_ = true;
                } else {
                    phase_ = RestartPhase::Aligning;
                }
            }
            break;
        case RestartPhase::AwaitReady:
            if (input.receiver_ready) {
                if (input.taker_aligned) {
                    phase_ = RestartPhase::Executing;
                    execution_authorized_ever_ = true;
                } else {
                    phase_ = RestartPhase::Aligning;
                }
            }
            break;
        case RestartPhase::Aligning:
            if (input.taker_aligned && plan_->executable_coordination()) {
                phase_ = RestartPhase::Executing;
                execution_authorized_ever_ = true;
            }
            break;
        case RestartPhase::Executing:
            if (observe_release(input)) {
                phase_ = RestartPhase::TakerLockout;
                break;
            }
            if (input.execution_feedback.has_value() &&
                feedback_matches(*input.execution_feedback)) {
                switch (input.execution_feedback->status) {
                    case RestartExecutionStatus::Running:
                        break;
                    case RestartExecutionStatus::Completed:
                        phase_ = RestartPhase::VerifyRelease;
                        execution_completed_at_s_ = input.server_time_s;
                        break;
                    case RestartExecutionStatus::Rejected:
                        enter_fallback(
                            RestartFallbackReason::ExecutionRejected, input);
                        break;
                    case RestartExecutionStatus::TimedOut:
                        enter_fallback(
                            RestartFallbackReason::ExecutionTimedOut, input);
                        break;
                }
            }
            break;
        case RestartPhase::VerifyRelease:
            if (observe_release(input)) {
                phase_ = RestartPhase::TakerLockout;
            } else if (input.server_time_s - execution_completed_at_s_ >=
                       parameters_.release_verification_timeout_s) {
                enter_fallback(
                    RestartFallbackReason::ReleaseNotObserved, input);
            }
            break;
        case RestartPhase::Idle:
        case RestartPhase::TakerLockout:
        case RestartPhase::Complete:
            break;
    }

    last_observed_mode_ = input.play_mode;
    return decision_for(input.self_player_number);
}

bool is_our_restart(world::PlayMode mode) {
    switch (mode) {
        case world::PlayMode::OurKickOff:
        case world::PlayMode::OurThrowIn:
        case world::PlayMode::OurCornerKick:
        case world::PlayMode::OurGoalKick:
        case world::PlayMode::OurOffside:
        case world::PlayMode::OurFreeKick:
        case world::PlayMode::OurDirectFreeKick:
        case world::PlayMode::OurPenaltyKick:
        case world::PlayMode::OurPenaltyShoot:
            return true;
        default:
            return false;
    }
}

bool restart_requires_receiver(world::PlayMode mode) {
    switch (mode) {
        case world::PlayMode::OurThrowIn:
        case world::PlayMode::OurCornerKick:
        case world::PlayMode::OurOffside:
        case world::PlayMode::OurFreeKick:
        case world::PlayMode::OurDirectFreeKick:
            return true;
        default:
            return false;
    }
}

std::optional<double> safe_restart_contact_direction_deg(
    world::PlayMode mode,
    const Position2& ball) {
    if (!is_our_restart(mode) || !std::isfinite(ball[0]) ||
        !std::isfinite(ball[1])) {
        return std::nullopt;
    }

    if (mode == world::PlayMode::OurGoalKick ||
        mode == world::PlayMode::OurKickOff) {
        return 0.0;
    }
    const Position2 target = restart_contact_target(
        mode, ball, RestartVariant::Primary);

    const Position2 delta = math::vec2_sub(target, ball);
    if (math::norm2(delta) <= 1.0e-6) return 0.0;
    return math::vector_angle_deg(delta);
}

std::string_view to_string(RestartPhase phase) {
    switch (phase) {
        case RestartPhase::Idle: return "Idle";
        case RestartPhase::Positioning: return "Positioning";
        case RestartPhase::AwaitReady: return "AwaitReady";
        case RestartPhase::Aligning: return "Aligning";
        case RestartPhase::Executing: return "Executing";
        case RestartPhase::VerifyRelease: return "VerifyRelease";
        case RestartPhase::TakerLockout: return "TakerLockout";
        case RestartPhase::Complete: return "Complete";
    }
    return "Idle";
}

std::string_view to_string(RestartExecutionStatus status) {
    switch (status) {
        case RestartExecutionStatus::Running: return "Running";
        case RestartExecutionStatus::Completed: return "Completed";
        case RestartExecutionStatus::Rejected: return "Rejected";
        case RestartExecutionStatus::TimedOut: return "TimedOut";
    }
    return "Running";
}

std::string_view to_string(RestartFallbackReason reason) {
    switch (reason) {
        case RestartFallbackReason::None: return "None";
        case RestartFallbackReason::SoftDeadline: return "SoftDeadline";
        case RestartFallbackReason::ExecutionRejected: return "ExecutionRejected";
        case RestartFallbackReason::ExecutionTimedOut: return "ExecutionTimedOut";
        case RestartFallbackReason::ReleaseNotObserved: return "ReleaseNotObserved";
    }
    return "None";
}

std::string_view to_string(RestartVariant variant) {
    switch (variant) {
        case RestartVariant::Primary: return "Primary";
        case RestartVariant::Alternate: return "Alternate";
        case RestartVariant::Safety: return "Safety";
    }
    return "Primary";
}

}  // namespace decision
