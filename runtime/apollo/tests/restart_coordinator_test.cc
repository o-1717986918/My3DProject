// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/restart_coordinator.h"

#include "src/math/math_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using decision::RestartCoordinationDecision;
using decision::RestartCoordinator;
using decision::RestartCoordinatorInput;
using decision::RestartExecutionFeedback;
using decision::RestartExecutionStatus;
using decision::RestartFallbackReason;
using decision::RestartPhase;
using decision::RoleAssignment;

std::vector<RoleAssignment> assignments() {
    return {
        {6, decision::RoleManager::ROLE_ST, {3.0, 2.0}},
        {2, decision::RoleManager::ROLE_CBL, {-8.0, 4.0}},
        {7, decision::RoleManager::ROLE_AP, {0.0, 0.0}},
        {1, decision::RoleManager::ROLE_GK, {-27.0, 0.0}},
        {5, decision::RoleManager::ROLE_CBM, {-2.0, 3.0}},
        {3, decision::RoleManager::ROLE_CBR, {-8.0, -4.0}},
        {4, decision::RoleManager::ROLE_CDM, {-5.0, 0.0}},
    };
}

RestartCoordinatorInput input_for(
    world::PlayMode mode,
    double time_s,
    std::uint64_t epoch = 41U) {
    RestartCoordinatorInput input;
    input.play_mode = mode;
    input.server_time_s = time_s;
    input.self_player_number = 7;
    input.restart_epoch = epoch;
    input.ball_position_m = {0.0, 0.0};
    input.ball_position_valid = true;
    input.role_assignments = assignments();
    return input;
}

bool expect(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << message << '\n';
    return false;
}

bool test_frozen_plan_and_success_lifecycle() {
    RestartCoordinator coordinator;
    auto input = input_for(world::PlayMode::OurFreeKick, 10.0);
    auto decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Positioning &&
                decision.plan.has_value() &&
                decision.plan->epoch == 41U &&
                decision.plan->revision == 1U &&
                decision.plan->taker_player_number == 7 &&
                decision.plan->receiver_player_number == 6 &&
                decision.plan->requires_receiver_ready &&
                decision.plan->executable_coordination(),
            "free-kick plan was not frozen deterministically")) {
        return false;
    }

    // Later role churn must not change the frozen participants.
    input.server_time_s = 10.1;
    input.role_assignments[2].role_id = decision::RoleManager::ROLE_CBM;
    input.role_assignments[4].role_id = decision::RoleManager::ROLE_AP;
    input.team_positioned = true;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::AwaitReady &&
                decision.plan->taker_player_number == 7 &&
                decision.wait_for_receiver && decision.should_position,
            "role churn changed the frozen taker or skipped receiver readiness")) {
        return false;
    }

    input.server_time_s = 10.2;
    input.receiver_ready = true;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Aligning &&
                decision.should_align && !decision.execution_authorized,
            "ready receiver did not advance to alignment")) {
        return false;
    }

    input.server_time_s = 10.3;
    input.taker_aligned = true;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Executing &&
                decision.execution_authorized,
            "aligned taker was not authorized exactly in Executing")) {
        return false;
    }

    input.server_time_s = 10.4;
    input.execution_feedback = RestartExecutionFeedback{
        40U, 1U, RestartExecutionStatus::Completed};
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Executing,
            "stale execution feedback changed the active restart")) {
        return false;
    }

    input.server_time_s = 10.5;
    input.execution_feedback = RestartExecutionFeedback{
        41U, 1U, RestartExecutionStatus::Completed};
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::VerifyRelease &&
                !decision.execution_authorized,
            "matching completion did not enter release verification")) {
        return false;
    }

    input.execution_feedback.reset();
    input.ball_position_m = {0.5, 0.0};
    input.ball_velocity_mps = {0.8, 0.0};
    input.ball_velocity_valid = true;
    input.server_time_s = 10.6;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::VerifyRelease,
            "single noisy sample confirmed restart release")) {
        return false;
    }
    input.server_time_s = 10.7;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::TakerLockout &&
                decision.self_locked_out,
            "confirmed release did not lock out the taker")) {
        return false;
    }

    input.play_mode = world::PlayMode::PlayOn;
    input.restart_epoch = 0U;
    input.server_time_s = 10.8;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::TakerLockout &&
                decision.self_locked_out,
            "OurKick to PlayOn cleared the double-touch lockout")) {
        return false;
    }
    input.server_time_s = 10.9;
    input.another_player_touched_ball = true;
    decision = coordinator.update(input);
    if (!expect(
        decision.phase == RestartPhase::Complete &&
            !decision.self_locked_out,
        "another-player touch did not complete the restart lockout")) {
        return false;
    }
    input.server_time_s = 11.0;
    input.another_player_touched_ball = false;
    decision = coordinator.update(input);
    return expect(
        decision.phase == RestartPhase::Complete &&
            !decision.self_locked_out,
        "completed double-touch lockout was not latched");
}

bool test_authoritative_play_on_lockout() {
    RestartCoordinator coordinator;
    auto input = input_for(world::PlayMode::OurKickOff, 20.0, 52U);
    auto decision = coordinator.update(input);
    input.server_time_s = 20.1;
    input.team_positioned = true;
    decision = coordinator.update(input);
    if (!expect(decision.phase == RestartPhase::Aligning,
                "kickoff unexpectedly required a receiver")) {
        return false;
    }
    input.server_time_s = 20.2;
    input.taker_aligned = true;
    decision = coordinator.update(input);
    if (!expect(decision.phase == RestartPhase::Executing,
                "kickoff did not reach execution")) {
        return false;
    }
    input.play_mode = world::PlayMode::PlayOn;
    input.restart_epoch = 0U;
    input.server_time_s = 20.3;
    decision = coordinator.update(input);
    return expect(
        decision.phase == RestartPhase::TakerLockout &&
            decision.self_locked_out,
        "authoritative PlayOn did not preserve the executing taker's lockout");
}

bool test_goal_kick_taker_and_epoch_replan() {
    RestartCoordinator coordinator;
    auto input = input_for(world::PlayMode::OurGoalKick, 30.0, 61U);
    input.self_player_number = 1;
    input.ball_position_m = {-23.0, 0.0};
    auto decision = coordinator.update(input);
    if (!expect(
            decision.plan.has_value() &&
                decision.plan->taker_player_number == 1 &&
                decision.plan->receiver_player_number == 0 &&
                !decision.plan->requires_receiver_ready &&
                decision.self_is_taker,
            "goal kick did not freeze the assigned goalkeeper as taker")) {
        return false;
    }
    input.server_time_s = 30.1;
    input.team_positioned = true;
    decision = coordinator.update(input);
    if (!expect(decision.phase == RestartPhase::Aligning,
                "goal kick did not bypass receiver readiness")) {
        return false;
    }

    input.server_time_s = 30.2;
    input.restart_epoch = 60U;
    input.role_assignments[3].role_id = decision::RoleManager::ROLE_CBM;
    input.role_assignments[1].role_id = decision::RoleManager::ROLE_GK;
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Aligning &&
                decision.plan->epoch == 61U &&
                decision.plan->taker_player_number == 1,
            "delayed older epoch replaced the active restart plan")) {
        return false;
    }

    input.server_time_s = 30.3;
    input.restart_epoch = 62U;
    decision = coordinator.update(input);
    return expect(
        decision.phase == RestartPhase::Positioning &&
            decision.plan->epoch == 62U && decision.plan->revision == 1U &&
            decision.plan->taker_player_number == 2,
        "new authoritative epoch did not freeze a new goal-kick plan");
}

bool test_soft_deadline_and_single_fallback() {
    RestartCoordinator::Parameters parameters;
    parameters.soft_deadline_s = 2.0;
    parameters.hard_deadline_s = 5.0;
    parameters.release_verification_timeout_s = 0.5;
    RestartCoordinator coordinator(parameters);
    auto input = input_for(world::PlayMode::OurFreeKick, 40.0, 71U);
    coordinator.update(input);

    input.server_time_s = 42.1;
    input.ball_position_m = {
        std::numeric_limits<double>::quiet_NaN(), 0.0};
    auto decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Aligning &&
                decision.plan->fallback && decision.plan->revision == 2U &&
                !decision.plan->requires_receiver_ready &&
                decision.plan->ball_anchor_valid &&
                std::isfinite(decision.plan->ball_anchor_m[0]) &&
                decision.fallback_reason == RestartFallbackReason::SoftDeadline,
            "soft deadline fallback discarded its last valid ball anchor")) {
        return false;
    }

    input.server_time_s = 42.2;
    input.taker_aligned = true;
    decision = coordinator.update(input);
    if (!expect(decision.phase == RestartPhase::Executing,
                "fallback could not advance to execution")) {
        return false;
    }
    input.server_time_s = 42.25;
    input.execution_feedback = RestartExecutionFeedback{
        71U, 1U, RestartExecutionStatus::Completed};
    decision = coordinator.update(input);
    if (!expect(
            decision.phase == RestartPhase::Executing,
            "feedback for a superseded plan revision was accepted")) {
        return false;
    }
    input.server_time_s = 42.3;
    input.execution_feedback = RestartExecutionFeedback{
        71U, 2U, RestartExecutionStatus::Rejected};
    decision = coordinator.update(input);
    return expect(
        decision.phase == RestartPhase::Complete &&
            decision.plan->revision == 2U,
        "a rejected fallback incorrectly created a second fallback");
}

bool test_execution_failure_and_release_timeout_fallbacks() {
    RestartCoordinator::Parameters parameters;
    parameters.soft_deadline_s = 4.0;
    parameters.hard_deadline_s = 8.0;
    parameters.release_verification_timeout_s = 0.5;

    RestartCoordinator rejected(parameters);
    auto input = input_for(world::PlayMode::OurKickOff, 50.0, 81U);
    rejected.update(input);
    input.server_time_s = 50.1;
    input.team_positioned = true;
    rejected.update(input);
    input.server_time_s = 50.2;
    input.taker_aligned = true;
    rejected.update(input);
    input.server_time_s = 50.3;
    input.execution_feedback = RestartExecutionFeedback{
        81U, 1U, RestartExecutionStatus::TimedOut};
    auto decision = rejected.update(input);
    if (!expect(
            decision.phase == RestartPhase::Aligning &&
                decision.plan->revision == 2U &&
                decision.fallback_reason == RestartFallbackReason::ExecutionTimedOut,
            "execution timeout did not consume the fallback attempt")) {
        return false;
    }
    input.play_mode = world::PlayMode::PlayOn;
    input.restart_epoch = 0U;
    input.execution_feedback.reset();
    input.server_time_s = 50.4;
    decision = rejected.update(input);
    if (!expect(
            decision.phase == RestartPhase::TakerLockout &&
                decision.self_locked_out,
            "fallback reset the possible first-contact lockout")) {
        return false;
    }

    RestartCoordinator no_release(parameters);
    input = input_for(world::PlayMode::OurKickOff, 60.0, 91U);
    no_release.update(input);
    input.server_time_s = 60.1;
    input.team_positioned = true;
    no_release.update(input);
    input.server_time_s = 60.2;
    input.taker_aligned = true;
    no_release.update(input);
    input.server_time_s = 60.3;
    input.execution_feedback = RestartExecutionFeedback{
        91U, 1U, RestartExecutionStatus::Completed};
    decision = no_release.update(input);
    if (!expect(decision.phase == RestartPhase::VerifyRelease,
                "completed contact skipped release verification")) {
        return false;
    }
    input.execution_feedback.reset();
    input.server_time_s = 60.81;
    decision = no_release.update(input);
    return expect(
        decision.phase == RestartPhase::Aligning &&
            decision.plan->revision == 2U &&
            decision.fallback_reason == RestartFallbackReason::ReleaseNotObserved,
        "missing release evidence did not trigger the one fallback");
}

bool test_hard_deadline_and_invalid_plan() {
    RestartCoordinator::Parameters parameters;
    parameters.soft_deadline_s = 2.0;
    parameters.hard_deadline_s = 3.0;
    RestartCoordinator coordinator(parameters);
    auto input = input_for(world::PlayMode::OurFreeKick, 70.0, 101U);
    input.role_assignments.erase(
        std::remove_if(
            input.role_assignments.begin(), input.role_assignments.end(),
            [](const RoleAssignment& assignment) {
                return assignment.role_id == decision::RoleManager::ROLE_AP;
            }),
        input.role_assignments.end());
    auto decision = coordinator.update(input);
    if (!expect(
            decision.plan.has_value() &&
                !decision.plan->executable_coordination(),
            "missing AP produced executable restart coordination")) {
        return false;
    }
    input.server_time_s = 71.0;
    input.team_positioned = true;
    decision = coordinator.update(input);
    if (!expect(decision.phase == RestartPhase::Positioning,
                "invalid plan advanced toward contact")) {
        return false;
    }
    input.server_time_s = 73.0;
    decision = coordinator.update(input);
    return expect(
        decision.phase == RestartPhase::Complete &&
            decision.hard_deadline_reached &&
            !decision.execution_authorized,
        "hard deadline did not terminate invalid coordination");
}

bool test_determinism_and_legal_directions() {
    auto forward_component = [](double degrees) {
        return std::cos(math::deg_to_rad(degrees));
    };
    auto lateral_component = [](double degrees) {
        return std::sin(math::deg_to_rad(degrees));
    };

    const auto upper_corner = decision::safe_restart_contact_direction_deg(
        world::PlayMode::OurCornerKick,
        {decision::field_geometry::kActualHalfLengthM,
         decision::field_geometry::kActualHalfWidthM});
    const auto lower_corner = decision::safe_restart_contact_direction_deg(
        world::PlayMode::OurCornerKick,
        {decision::field_geometry::kActualHalfLengthM,
         -decision::field_geometry::kActualHalfWidthM});
    const auto upper_throw = decision::safe_restart_contact_direction_deg(
        world::PlayMode::OurThrowIn, {3.0, 18.0});
    const auto lower_throw = decision::safe_restart_contact_direction_deg(
        world::PlayMode::OurThrowIn, {3.0, -18.0});
    const auto kickoff = decision::safe_restart_contact_direction_deg(
        world::PlayMode::OurKickOff, {0.0, 0.0});
    const auto goal_kick = decision::safe_restart_contact_direction_deg(
        world::PlayMode::OurGoalKick, {-23.0, 0.0});
    if (!expect(
            upper_corner.has_value() && lower_corner.has_value() &&
                forward_component(*upper_corner) < 0.0 &&
                lateral_component(*upper_corner) < 0.0 &&
                forward_component(*lower_corner) < 0.0 &&
                lateral_component(*lower_corner) > 0.0 &&
                upper_throw.has_value() && lateral_component(*upper_throw) < 0.0 &&
                lower_throw.has_value() && lateral_component(*lower_throw) > 0.0 &&
                kickoff.has_value() && forward_component(*kickoff) > 0.99 &&
                goal_kick.has_value() && forward_component(*goal_kick) > 0.99 &&
                !decision::safe_restart_contact_direction_deg(
                    world::PlayMode::TheirCornerKick, {27.5, 18.0}).has_value(),
            "restart direction did not point into the legal field corridor")) {
        return false;
    }

    RestartCoordinator first;
    RestartCoordinator second;
    auto lhs = input_for(world::PlayMode::OurFreeKick, 80.0, 111U);
    auto rhs = lhs;
    std::reverse(rhs.role_assignments.begin(), rhs.role_assignments.end());
    const auto first_decision = first.update(lhs);
    const auto second_decision = second.update(rhs);
    return expect(
        first_decision.plan.has_value() && second_decision.plan.has_value() &&
            first_decision.plan->taker_player_number ==
                second_decision.plan->taker_player_number &&
            first_decision.plan->receiver_player_number ==
                second_decision.plan->receiver_player_number &&
            std::abs(first_decision.plan->contact_direction_deg -
                     second_decision.plan->contact_direction_deg) < 1.0e-12,
        "RoleAssignment ordering changed the frozen restart plan");
}

bool test_parameter_validation() {
    RestartCoordinator::Parameters invalid;
    invalid.soft_deadline_s = invalid.hard_deadline_s;
    try {
        const RestartCoordinator ignored(invalid);
        (void)ignored;
    } catch (const std::invalid_argument&) {
        return true;
    }
    return expect(false, "invalid restart deadlines were accepted");
}

}  // namespace

int main() {
    if (!test_frozen_plan_and_success_lifecycle()) return 1;
    if (!test_authoritative_play_on_lockout()) return 1;
    if (!test_goal_kick_taker_and_epoch_replan()) return 1;
    if (!test_soft_deadline_and_single_fallback()) return 1;
    if (!test_execution_failure_and_release_timeout_fallbacks()) return 1;
    if (!test_hard_deadline_and_invalid_plan()) return 1;
    if (!test_determinism_and_legal_directions()) return 1;
    if (!test_parameter_validation()) return 1;
    return 0;
}
