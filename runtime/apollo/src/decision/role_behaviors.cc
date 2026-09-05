// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/role_behaviors.h"

#include "src/decision/behavior_nodes.h"
#include "src/decision/kick_contract.h"
#include "src/decision/walk_planner.h"
#include "src/decision/field_geometry.h"
#include "src/decision/role_manager.h"
#include "src/math/math_utils.h"
#include "src/strategy/action_capability.h"
#include "src/strategy/reach_time_model.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <optional>

namespace decision {

namespace {

constexpr double kWalkMaxSpeedM = 3.0;
constexpr double kWalkBrakeDecelMps2 = 1.0;
constexpr double kWalkStopRadiusM = 0.15;
constexpr double kWalkHeadingSlowStartDeg = 35.0;
// The currently deployed walk is substantially more reliable in its forward
// domain than while translating sideways. Above this heading error, compose a
// pure turn with a later forward walk instead of asking the policy to strafe or
// backpedal. This is a runtime fallback, not a learned omnidirectional claim.
constexpr double kWalkHeadingStopDeg = 55.0;
// The exact-physics kick table was validated with the ball 0.31--0.40 m in
// front and at most 0.08 m to either side.  Keep the decision release gate in
// the same domain instead of starting the one-second macro from the former
// coarse dribble distance (up to 0.85 m).
// The walk controller brakes inside a 0.15 m target radius. Aim slightly past
// the desired setup slot so that braking converges near 0.35 m behind the ball.
constexpr double kDribbleApproachDistanceM = 0.22;
constexpr double kDribblePrecisionEntryDistanceM = 1.25;
// Canonical server contact pose. Use the same longitudinal target for the
// precision controller and release gate. The small 0.04 m band includes the
// observed -0.30 m monitor setup while keeping the lower edge at the explicit
// 0.30 m minimum ball-contact distance.
constexpr double kKickContactBehindM = 0.34;
constexpr double kDribbleCommandBehindM = 0.34;
constexpr double kForwardContactBallLocalYM = 0.0;
constexpr double kProceduralDribbleBallLocalYM = 0.04;
// Enter the neutral hold near the rear edge of the validated [0.30, 0.34] m
// slot. The server carries a few centimetres of approach momentum after the
// command switch; this converges near the 0.32 m optimization centre instead
// of leaving the envelope before the debounce completes.
constexpr double kProceduralDribbleBallLocalXM = 0.335;
// Enter the release latch inside +/-20 mm, then permit at most 5 mm of
// measured one-frame drift while the neutral command removes the remaining
// gait phase.  The wider motion-layer boundary is backed by a 100/100 exact
// physics held-out replay over y=[0.015, 0.065] m.
constexpr double kProceduralDribbleLatchedLateralToleranceM = 0.025;
constexpr double kProceduralPassBallLocalXM = 0.31;
constexpr double kProceduralPassBallLocalYM = -0.04;
constexpr double kProceduralPassBallPositionToleranceM = 0.02;
// The server walk controller brakes about 2.3 cm long and 2.2 cm low in the
// narrow strong-kick slot. These are approach set-points only; release is
// still checked against each anchor's independently validated physical slot.
constexpr double kProceduralStrongKickCommandBehindM = 0.3155;
constexpr double kProceduralShotCommandBallLocalYM = 0.0620;
// The mirrored right-team clear calibration settles about 1.8 cm above its
// command, so aim lower while retaining the same measured release envelope.
constexpr double kProceduralClearCommandBallLocalYM = 0.0220;
// The learned walk has a small-command dead zone of roughly 0.08--0.10 m/s.
// A gain of eight makes the final centimetre-scale reverse correction
// observable in the server while the explicit speed clamps below retain the
// same safety envelope.
constexpr double kDribbleLongitudinalGain = 8.0;
constexpr double kDribbleLateralGain = 4.0;
constexpr double kDribbleMaxForwardSetupSpeedMps = 0.85;
constexpr double kDribbleMaxReverseSetupSpeedMps = 0.35;
constexpr double kDribbleMaxLateralSetupSpeedMps = 0.35;
constexpr double kDribbleSideDistanceM = 0.8;
constexpr double kDribbleSideClearanceM = 0.55;
constexpr double kDribbleSideStepBehindThresholdM = 0.1;
constexpr double kDribbleMaxLateralOffsetM = 0.08;
constexpr double kDribbleMaxAheadM = 0.15;
// Ported from the validated Python competition path. The motion layer drives
// forward for 0.65 s and stabilizes for 0.35 s, so the decision layer owns the
// KickCommand variant for the complete one-second macro.
constexpr double kKickDurationS = 1.25;
constexpr double kKickCooldownS = 0.5;
// Debounce the release pose for several decision cycles.  This is a safety
// guard for the accepted fallback action, not a claim that zero-command walk
// converges to a phase-independent joint state; the transition policy owns
// that problem.  A longer 0.60 s hold starved valid passes because normal
// server yaw sway repeatedly reset the timer.
constexpr double kKickSetupStableHoldS = 0.25;
// Two neutral cycles remove the dynamic walk phase before a static-base kick
// trajectory starts.  Dribble uses a separately validated 5 mm latch margin
// so this short hold cannot be lost to ordinary one-frame localization sway.
// Range-pass retains immediate release until its own expanded pose envelope is
// independently evaluated; shot/clear keep their calibrated two-cycle hold.
constexpr double kProceduralKickSetupStableHoldS = 0.04;
// The deployed walk commonly retains 0.22--0.35 m/s of measured torso motion
// after entering its neutral command. Requiring less than 0.20 m/s starved
// every contact in a complete comparison match. The procedural runner repeats
// this same bound before it captures the current pose.
constexpr double kKickSetupMaximumPlanarSpeedMps = 0.50;
constexpr double kKickMinBallDistanceM = 0.30;
constexpr double kKickMaxBallDistanceM = 0.41;
// Server zero-command sway is about two centimetres peak-to-peak. A 3 cm
// release band remains well inside the residual runner's 9 cm contact
// envelope while allowing the stable-hold timer to survive one gait cycle.
constexpr double kKickSetupLongitudinalToleranceM = 0.04;
constexpr double kKickSetupLateralToleranceM = 0.03;
// The zero-command policy oscillates around roughly two degrees on the server.
// Three degrees remains comfortably below the ten-degree action promotion gate
// while admitting a continuous debounce window for the stable fallback.
constexpr double kKickMaxOrientationErrorDeg = 3.0;
constexpr double kKickSetupOrientationGain = 3.0;
// Brake before the exact release slot rather than waiting until the body has
// already crossed it.  This wider corridor does not authorize contact; it only
// switches from the walking actor to neutral while residual momentum decays.
constexpr double kKickPreSettleLongitudinalToleranceM = 0.08;
constexpr double kKickPreSettleLateralToleranceM = 0.08;
constexpr double kKickPreSettleMaximumYawErrorDeg = 8.0;
// Runtime traces show that the walk policy may accelerate for one or two
// frames after a zero-command switch.  Use a conservative measured stopping
// envelope to start neutral capture before that residual gait phase carries
// the torso through the centimetre-scale contact slot.
constexpr double kKickPreSettleEffectiveDecelMps2 = 0.35;
constexpr double kKickPreSettlePaddingM = 0.05;
constexpr double kKickPreSettleMaximumDistanceM = 0.45;
constexpr double kKickPreSettleEntrySpeedMps = 0.25;
constexpr double kKickPreSettleExitSpeedMps = 0.20;
constexpr double kKickPreSettleStableHoldS = 0.10;
// Preserve the original Apollo walk-through-ball behavior as an explicit,
// observable last resort. It is available only after a sustained near-ball
// setup attempt and inside this wider contact corridor.
// Non-pass actions recover the original Apollo contact tempo quickly when
// precision setup stalls. A targeted pass keeps the longer window because a
// fixed forward contact is not semantically equivalent to the agreed pass.
constexpr double kForwardContactFastFallbackDelayS = 0.45;
constexpr double kForwardContactPassFallbackDelayS = 1.20;
constexpr double kForwardContactFallbackMinimumBehindM = 0.20;
constexpr double kForwardContactFallbackMaximumBehindM = 0.60;
constexpr double kForwardContactFallbackMaximumLateralM = 0.18;
constexpr double kForwardContactFallbackMaximumYawErrorDeg = 15.0;
constexpr double kForwardContactFallbackMaximumPlanarSpeedMps = 0.65;
// Longer than the largest sparse replay-test interval while still shorter
// than the fallback delay, so a genuinely abandoned attempt cannot inherit a
// completed timeout on re-entry.
constexpr double kKickSetupContinuityTimeoutS = 1.00;
constexpr double kKickSetupDirectionResetDeg = 20.0;
constexpr double kRejectedPassRetryDelayS = 2.0;

bool is_our_set_play(const world::WorldSnapshot& snapshot) {
    return snapshot.play_mode_group == world::PlayModeGroup::OurKick;
}

std::array<double, 2> role_position_from_blackboard(const Blackboard& blackboard) {
    if (!blackboard.exists(Blackboard::kKeyRolePos)) {
        return {0.0, 0.0};
    }
    return blackboard.get<std::array<double, 2>>(Blackboard::kKeyRolePos);
}

const RestartCoordinationDecision* restart_decision_from_blackboard(
    const Blackboard& blackboard) {
    if (!blackboard.exists(Blackboard::kKeyRestartDecision)) return nullptr;
    return &blackboard.get<RestartCoordinationDecision>(
        Blackboard::kKeyRestartDecision);
}

void synchronize_restart_contact_state(
    APState& state,
    const RestartCoordinationDecision& decision) {
    if (!state.active_kick_command.has_value() ||
        !state.active_kick_command->restart_epoch.has_value() ||
        !state.active_kick_command->restart_revision.has_value()) {
        return;
    }
    const bool matching_plan = decision.plan.has_value() &&
        *state.active_kick_command->restart_epoch == decision.plan->epoch &&
        *state.active_kick_command->restart_revision == decision.plan->revision;
    if (decision.execution_authorized && matching_plan) return;

    state.active_kick_command.reset();
    state.kick_active_until_s = 0.0;
    if (!matching_plan) {
        state.dribble_ready = false;
        state.kick_setup_stable_since_s = 0.0;
    }
}

WalkCommand make_walk_command(const std::array<double, 2>& target_position_m) {
    WalkCommand command;
    command.target_2d_m = target_position_m;
    command.target_absolute = true;
    command.orientation_deg = std::nullopt;
    command.orientation_absolute = true;
    return command;
}

double walk_speed_command(double distance_m, double heading_error_abs_deg) {
    // Physical braking curve: v = sqrt(2 * a * (d - stop_radius)). Continuous
    // through zero at d = kWalkStopRadiusM and feasible to track for a robot
    // with peak deceleration a ~= kWalkBrakeDecelMps2.
    if (distance_m <= kWalkStopRadiusM) {
        return 0.0;
    }
    const double effective_dist = distance_m - kWalkStopRadiusM;
    const double brake = std::sqrt(2.0 * kWalkBrakeDecelMps2 * effective_dist);
    double speed = std::min(kWalkMaxSpeedM, brake);

    if (heading_error_abs_deg > kWalkHeadingSlowStartDeg) {
        const double t = std::clamp(
            (kWalkHeadingStopDeg - heading_error_abs_deg) /
                (kWalkHeadingStopDeg - kWalkHeadingSlowStartDeg),
            0.0,
            1.0);
        speed *= t;
    }

    return speed;
}

std::optional<double> orientation_to_point_from_self(
    const world::WorldSnapshot& snapshot,
    const std::array<double, 2>& target) {
    const double dx = target[0] - snapshot.self.position_m[0];
    const double dy = target[1] - snapshot.self.position_m[1];
    if (dx * dx + dy * dy <= 1e-6) {
        return std::nullopt;
    }
    return math::vector_angle_deg({dx, dy});
}

std::optional<double> orientation_to_ball_from_self(const world::WorldSnapshot& snapshot) {
    return orientation_to_point_from_self(
        snapshot,
        {snapshot.ball.position_m[0], snapshot.ball.position_m[1]});
}

WalkCommand make_walk_command_avoiding(
    const std::array<double, 2>& target_position_m,
    const world::WorldSnapshot& snapshot,
    std::optional<double> opponent_x_threshold = std::nullopt,
    bool avoid_field_boundaries = true,
    bool orient_to_ball = true,
    std::optional<int> role_id = std::nullopt,
    bool suppress_heading_slowdown = false,
    bool avoid_obstacles = true) {
    const std::array<double, 2> self{snapshot.self.position_m[0], snapshot.self.position_m[1]};
    const double dist = math::planar_dist(self, target_position_m);

    if (dist < field_geometry::kNearTargetM) {
        WalkCommand command = make_walk_command(target_position_m);
        command.orientation_deg = orient_to_ball
            ? orientation_to_ball_from_self(snapshot)
            : std::nullopt;
        command.orientation_absolute = true;
        command.role_id = role_id;
        return command;
    }

    const auto plan = plan_walk(
        self,
        target_position_m,
        snapshot,
        snapshot.player_number,
        opponent_x_threshold,
        avoid_field_boundaries,
        avoid_obstacles);
    const double heading_rad = math::deg_to_rad(plan.heading_deg);
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(snapshot.self.orientation_wxyz);
    const double rel_heading = heading_rad - math::deg_to_rad(self_yaw_deg);
    const double heading_error_abs_deg = std::abs(math::normalize_deg(plan.heading_deg - self_yaw_deg));

    // Generic navigation now composes turn-in-place and forward movement. Only
    // an explicitly requested precision mover may bypass the heading slowdown;
    // ordinary close-range tactics no longer silently demand the weak lateral
    // domain from the deployed walk actor.
    const bool strafe = suppress_heading_slowdown;
    const double speed =
        walk_speed_command(dist, strafe ? 0.0 : heading_error_abs_deg);

    WalkCommand command;
    command.target_2d_m = {speed * std::cos(rel_heading), speed * std::sin(rel_heading)};
    command.target_absolute = false;
    command.orientation_deg = (strafe && orient_to_ball)
        ? orientation_to_ball_from_self(snapshot).value_or(plan.heading_deg)
        : plan.heading_deg;
    command.orientation_absolute = true;
    command.role_id = role_id;
    return command;
}

bool match_role(const Blackboard& blackboard, int role_id) {
    return current_role_from_blackboard(blackboard) == role_id;
}

struct APDecisionContext {
    const world::WorldSnapshot& snapshot;
    APState& state;
    bool procedural_kick_enabled{false};
    std::array<double, 2> ball{0.0, 0.0};
    std::array<double, 2> self{0.0, 0.0};
    double ball_distance{0.0};
};

using APNodePtr = bt::NodePtr<APDecisionContext>;

struct GKDecisionContext {
    const world::WorldSnapshot& snapshot;
    std::array<double, 2> ball{0.0, 0.0};
    std::array<double, 2> self{0.0, 0.0};
    double ball_distance{0.0};
};

HighLevelCommand make_dribble_command(
    APDecisionContext& context,
    double absolute_direction_deg,
    const strategy::CooperativeAction* cooperative_action = nullptr,
    bool allow_kick = true,
    int motion_role_id = RoleManager::ROLE_AP,
    const RestartPlan* restart_plan = nullptr) {
    const double now = context.snapshot.server_time;
    if (context.state.kick_active_until_s > now) {
        return context.state.active_kick_command.value_or(KickCommand{});
    }
    if (context.state.kick_active_until_s > 0.0) {
        context.state.kick_active_until_s = 0.0;
        context.state.dribble_ready = false;
        context.state.kick_pre_settling = false;
        context.state.kick_pre_settle_stable_since_s = 0.0;
        context.state.kick_setup_stable_since_s = 0.0;
        context.state.kick_setup_started_s = 0.0;
        context.state.kick_setup_last_update_s = 0.0;
        context.state.active_kick_command.reset();
    }

    const double direction_rad = math::deg_to_rad(absolute_direction_deg);
    const std::array<double, 2> direction{
        std::cos(direction_rad),
        std::sin(direction_rad),
    };
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            context.snapshot.self.orientation_wxyz);
    const double orientation_error_deg = std::abs(
        math::normalize_deg(absolute_direction_deg - self_yaw_deg));

    const bool setup_discontinuous =
        context.state.kick_setup_last_update_s > 0.0 &&
        now - context.state.kick_setup_last_update_s >
            kKickSetupContinuityTimeoutS;
    const bool setup_direction_changed =
        context.state.kick_setup_started_s > 0.0 &&
        std::abs(math::normalize_deg(
            absolute_direction_deg - context.state.kick_setup_direction_deg)) >
            kKickSetupDirectionResetDeg;
    if (setup_discontinuous || setup_direction_changed ||
        !context.snapshot.ball.position_valid ||
        context.ball_distance > kDribblePrecisionEntryDistanceM) {
        context.state.kick_setup_started_s = 0.0;
        context.state.kick_setup_stable_since_s = 0.0;
        context.state.kick_pre_settling = false;
        context.state.kick_pre_settle_stable_since_s = 0.0;
    }
    if (context.snapshot.ball.position_valid &&
        context.ball_distance <= kDribblePrecisionEntryDistanceM &&
        context.state.kick_setup_started_s <= 0.0) {
        context.state.kick_setup_started_s = now;
        context.state.kick_setup_direction_deg = absolute_direction_deg;
    }
    context.state.kick_setup_last_update_s = now;
    const std::array<double, 2> perpendicular{-direction[1], direction[0]};
    const std::array<double, 2> push_target{
        context.ball[0] + direction[0] * field_geometry::kPushPastBallM,
        context.ball[1] + direction[1] * field_geometry::kPushPastBallM,
    };
    const std::array<double, 2> self_from_ball{
        context.self[0] - context.ball[0],
        context.self[1] - context.ball[1],
    };
    const double along_direction =
        self_from_ball[0] * direction[0] + self_from_ball[1] * direction[1];
    const double signed_lateral_offset =
        self_from_ball[0] * perpendicular[0] + self_from_ball[1] * perpendicular[1];
    const double lateral_offset = std::abs(signed_lateral_offset);
    const bool use_procedural_dribble =
        context.procedural_kick_enabled &&
        (cooperative_action == nullptr ||
         cooperative_action->category == strategy::ActionCategory::Dribble) &&
        restart_plan == nullptr &&
        context.snapshot.play_mode == world::PlayMode::PlayOn;
    const bool use_procedural_shot =
        context.procedural_kick_enabled && cooperative_action != nullptr &&
        cooperative_action->category == strategy::ActionCategory::Shoot &&
        restart_plan == nullptr &&
        context.snapshot.play_mode == world::PlayMode::PlayOn;
    const bool use_procedural_clear =
        context.procedural_kick_enabled && cooperative_action != nullptr &&
        cooperative_action->category == strategy::ActionCategory::Clear &&
        restart_plan == nullptr &&
        context.snapshot.play_mode == world::PlayMode::PlayOn;
    const bool use_procedural_range_pass =
        context.procedural_kick_enabled && cooperative_action != nullptr &&
        cooperative_action->category == strategy::ActionCategory::Pass &&
        cooperative_action->requested_ball_speed_mps >
            decision::kick_contract::kParameterizedPassRequestedSpeedMps +
                0.20 &&
        restart_plan == nullptr &&
        context.snapshot.play_mode == world::PlayMode::PlayOn;
    const bool use_procedural_strong_kick =
        use_procedural_shot || use_procedural_clear;
    const bool use_procedural_kick =
        use_procedural_dribble || use_procedural_range_pass ||
        use_procedural_strong_kick;
    const char* kick_mode = cooperative_action == nullptr
        ? (restart_plan == nullptr ? "forward" : "restart")
        : cooperative_action->category == strategy::ActionCategory::Pass
            ? "pass"
            : cooperative_action->category == strategy::ActionCategory::Shoot
                ? "shot"
                : cooperative_action->category == strategy::ActionCategory::Clear
                    ? "clear"
                    : "dribble";

    double contact_behind_m = kKickContactBehindM;
    double command_behind_m = kDribbleCommandBehindM;
    double release_ball_local_y_m = kForwardContactBallLocalYM;
    double command_ball_local_y_m = kForwardContactBallLocalYM;
    double longitudinal_tolerance_m = kKickSetupLongitudinalToleranceM;
    double lateral_tolerance_m = kKickSetupLateralToleranceM;
    double procedural_max_orientation_error_deg =
        decision::kick_contract::kProceduralDribbleMaximumTargetAngleDeg;
    if (use_procedural_range_pass) {
        contact_behind_m = kProceduralPassBallLocalXM;
        command_behind_m = kProceduralPassBallLocalXM;
        release_ball_local_y_m = kProceduralPassBallLocalYM;
        command_ball_local_y_m = kProceduralPassBallLocalYM;
        longitudinal_tolerance_m = kProceduralPassBallPositionToleranceM;
        lateral_tolerance_m = kProceduralPassBallPositionToleranceM;
        procedural_max_orientation_error_deg =
            decision::kick_contract::kParameterizedPassMaximumTargetAngleDeg;
    } else if (use_procedural_clear) {
        contact_behind_m =
            decision::kick_contract::kProceduralClearBallLocalXM;
        command_behind_m = kProceduralStrongKickCommandBehindM;
        release_ball_local_y_m =
            decision::kick_contract::kProceduralClearBallLocalYM;
        command_ball_local_y_m = kProceduralClearCommandBallLocalYM;
        longitudinal_tolerance_m =
            decision::kick_contract::kProceduralClearBallLocalXRangeM;
        lateral_tolerance_m =
            decision::kick_contract::kProceduralClearBallLocalYRangeM;
        procedural_max_orientation_error_deg =
            decision::kick_contract::kProceduralClearMaximumTargetAngleDeg;
    } else if (use_procedural_shot) {
        contact_behind_m =
            decision::kick_contract::kProceduralShotBallLocalXM;
        command_behind_m = kProceduralStrongKickCommandBehindM;
        release_ball_local_y_m =
            decision::kick_contract::kProceduralShotBallLocalYM;
        command_ball_local_y_m = kProceduralShotCommandBallLocalYM;
        longitudinal_tolerance_m =
            decision::kick_contract::kProceduralShotBallLocalXRangeM;
        lateral_tolerance_m =
            decision::kick_contract::kProceduralShotBallLocalYRangeM;
        procedural_max_orientation_error_deg =
            decision::kick_contract::kProceduralShotMaximumTargetAngleDeg;
    } else if (use_procedural_dribble) {
        contact_behind_m = kProceduralDribbleBallLocalXM;
        command_behind_m = kProceduralDribbleBallLocalXM;
        release_ball_local_y_m = kProceduralDribbleBallLocalYM;
        command_ball_local_y_m = kProceduralDribbleBallLocalYM;
        longitudinal_tolerance_m = 0.035;
        lateral_tolerance_m = 0.02;
    }
    const bool needs_side_step =
        along_direction > -kDribbleSideStepBehindThresholdM &&
        lateral_offset < kDribbleSideClearanceM;
    std::array<double, 2> approach_target{
        context.ball[0] - direction[0] * kDribbleApproachDistanceM,
        context.ball[1] - direction[1] * kDribbleApproachDistanceM,
    };
    if (needs_side_step) {
        const double side_sign = signed_lateral_offset < 0.0 ? -1.0 : 1.0;
        approach_target = {
            context.ball[0] + perpendicular[0] * side_sign * kDribbleSideDistanceM,
            context.ball[1] + perpendicular[1] * side_sign * kDribbleSideDistanceM,
        };
    }
    // The table's full ball-position range is a robustness envelope, not the
    // desired release set. Converge to one repeatable canonical contact pose;
    // the residual selector can still absorb the remaining bounded error.
    const double behind_distance = -along_direction;
    const double release_lateral_error =
        signed_lateral_offset + release_ball_local_y_m;
    const double command_lateral_error =
        signed_lateral_offset + command_ball_local_y_m;
    const bool position_ready =
        std::abs(behind_distance - contact_behind_m) <=
            longitudinal_tolerance_m &&
        std::abs(release_lateral_error) <= lateral_tolerance_m;
    const bool latched_position_ready =
        position_ready ||
        (use_procedural_dribble && context.state.dribble_ready &&
         std::abs(behind_distance - contact_behind_m) <=
             longitudinal_tolerance_m &&
         std::abs(release_lateral_error) <=
             kProceduralDribbleLatchedLateralToleranceM);
    const double required_orientation_error_deg = use_procedural_kick
        ? procedural_max_orientation_error_deg
        : kKickMaxOrientationErrorDeg;
    const double planar_speed_mps = math::norm2({
        context.snapshot.self.lin_vel_b[0],
        context.snapshot.self.lin_vel_b[1],
    });
    const auto trace_setup_gate = [&](int gate, const char* phase) {
        if (context.state.last_kick_setup_gate == gate) return;
        context.state.last_kick_setup_gate = gate;
        std::cerr
            << "MY3D_KICK_SETUP player=" << context.snapshot.player_number
            << " mode=" << kick_mode
            << " phase=" << phase
            << " ball_distance=" << context.ball_distance
            << " behind_error=" << behind_distance - contact_behind_m
            << " lateral_error=" << release_lateral_error
            << " yaw_error_deg=" << orientation_error_deg
            << " speed=" << planar_speed_mps
            << '\n';
    };
    const bool fallback_contact_pose =
        context.snapshot.ball.position_valid &&
        context.ball_distance >= kForwardContactFallbackMinimumBehindM &&
        context.ball_distance <= kForwardContactFallbackMaximumBehindM &&
        behind_distance >= kForwardContactFallbackMinimumBehindM &&
        behind_distance <= kForwardContactFallbackMaximumBehindM &&
        lateral_offset <= kForwardContactFallbackMaximumLateralM &&
        orientation_error_deg <= kForwardContactFallbackMaximumYawErrorDeg &&
        planar_speed_mps <= kForwardContactFallbackMaximumPlanarSpeedMps;
    const bool is_targeted_pass =
        cooperative_action != nullptr &&
        cooperative_action->category == strategy::ActionCategory::Pass;
    const double fallback_delay_s = is_targeted_pass
        ? kForwardContactPassFallbackDelayS
        : kForwardContactFastFallbackDelayS;
    const bool fallback_due =
        context.state.kick_setup_started_s > 0.0 &&
        now - context.state.kick_setup_started_s >=
            fallback_delay_s;
    const bool fallback_allowed =
        // Never let a timeout pre-empt an exact procedural release slot. The
        // controlled 0.45 s run reached 14.7 mm longitudinal and 0.5 mm
        // lateral error, but the old ordering emitted fallback before the
        // readiness latch could enter its short neutral debounce.
        allow_kick && fallback_due && fallback_contact_pose &&
        !(use_procedural_kick && position_ready) &&
        now >= context.state.next_kick_allowed_s;
    const auto make_fallback_command = [&]() {
        trace_setup_gate(8, "fallback-forward-contact");
        context.state.kick_active_until_s = now + kKickDurationS;
        context.state.next_kick_allowed_s =
            context.state.kick_active_until_s + kKickCooldownS;
        KickCommand fallback_command;
        fallback_command.allow_forward_contact_fallback = true;
        if (cooperative_action != nullptr) {
            fallback_command.target_point_m = cooperative_action->target_point_m;
            fallback_command.requested_ball_speed_mps =
                cooperative_action->requested_ball_speed_mps;
            fallback_command.action_id = cooperative_action->action_id;
            fallback_command.sequence_id = cooperative_action->sequence_id;
            if (cooperative_action->category ==
                strategy::ActionCategory::Pass) {
                fallback_command.mode = KickMode::TargetedPass;
                fallback_command.receiver_player_number =
                    cooperative_action->target_player_number;
            } else if (cooperative_action->category ==
                       strategy::ActionCategory::Shoot) {
                fallback_command.mode = KickMode::Shot;
            } else if (cooperative_action->category ==
                       strategy::ActionCategory::Clear) {
                fallback_command.mode = KickMode::Clear;
            } else {
                fallback_command.mode = KickMode::DribbleTouch;
            }
        }
        if (restart_plan != nullptr) {
            fallback_command.restart_epoch = restart_plan->epoch;
            fallback_command.restart_revision = restart_plan->revision;
        }
        context.state.active_kick_command = fallback_command;
        return fallback_command;
    };

    if (!context.state.dribble_ready && position_ready) {
        context.state.dribble_ready = true;
    }

    const bool push_position_valid =
        along_direction <= kDribbleMaxAheadM &&
        lateral_offset <= kDribbleMaxLateralOffsetM &&
        context.ball_distance <= field_geometry::kPushBallEngageDistanceM;
    if (context.state.dribble_ready &&
        (!push_position_valid || !latched_position_ready)) {
        context.state.dribble_ready = false;
        context.state.kick_setup_stable_since_s = 0.0;
    }

    if (!context.state.dribble_ready) {
        if (fallback_allowed) {
            return make_fallback_command();
        }
        if (use_procedural_kick && context.state.kick_pre_settling) {
            if (planar_speed_mps > kKickPreSettleExitSpeedMps) {
                context.state.kick_pre_settle_stable_since_s = 0.0;
                trace_setup_gate(9, "pre-settle");
                return NeutralCommand{};
            }
            if (context.state.kick_pre_settle_stable_since_s <= 0.0) {
                context.state.kick_pre_settle_stable_since_s = now;
            }
            if (now - context.state.kick_pre_settle_stable_since_s <
                kKickPreSettleStableHoldS) {
                trace_setup_gate(9, "pre-settle");
                return NeutralCommand{};
            }
            context.state.kick_pre_settling = false;
            context.state.kick_pre_settle_stable_since_s = 0.0;
        }
        const double speed_aware_stopping_distance_m = std::clamp(
            planar_speed_mps * planar_speed_mps /
                    (2.0 * kKickPreSettleEffectiveDecelMps2) +
                kKickPreSettlePaddingM,
            kKickPreSettleLongitudinalToleranceM,
            kKickPreSettleMaximumDistanceM);
        const bool near_release_slot =
            behind_distance - contact_behind_m >=
                -kKickPreSettleLongitudinalToleranceM &&
            behind_distance - contact_behind_m <=
                speed_aware_stopping_distance_m &&
            std::abs(command_lateral_error) <=
                kKickPreSettleLateralToleranceM &&
            orientation_error_deg <= kKickPreSettleMaximumYawErrorDeg;
        if (use_procedural_kick && near_release_slot &&
            planar_speed_mps > kKickPreSettleEntrySpeedMps) {
            // The previous controller waited for the centimetre-scale slot
            // before braking.  Natural play then crossed the slot at roughly
            // 0.7--0.8 m/s and never satisfied the release debounce.  Start
            // neutral capture in a bounded pre-slot corridor, then resume
            // precision positioning once the body is slow enough.
            context.state.kick_pre_settling = true;
            context.state.kick_pre_settle_stable_since_s = 0.0;
            trace_setup_gate(9, "pre-settle");
            return NeutralCommand{};
        }
        if (needs_side_step) {
            // Exit the ball's front/side hazard region by turning toward an
            // offset waypoint and walking forward. Do not override the travel
            // heading with the final kick heading: that used to turn this
            // branch into a slow and unstable lateral strafe.
            trace_setup_gate(1, "side-relocate");
            return make_walk_command_avoiding(
                approach_target, context.snapshot, std::nullopt, true, false,
                motion_role_id, false, false);
        }
        // The generic walk-to-point controller brakes on a circular stop
        // radius. On a diagonal approach it can therefore stop too close to
        // the ball while still outside the kick table's lateral envelope. In
        // the near field, close the longitudinal and lateral errors
        // independently in the requested kick frame. The world model keeps a
        // bounded ball track through the expected torso occlusion, so this
        // controller can finish lining up after direct vision disappears.
        if (!needs_side_step && context.snapshot.ball.position_valid &&
            context.ball_distance <= kDribblePrecisionEntryDistanceM) {
            trace_setup_gate(2, "precision-position");
            const double forward_speed = std::clamp(
                kDribbleLongitudinalGain *
                    (behind_distance - command_behind_m),
                -kDribbleMaxReverseSetupSpeedMps,
                kDribbleMaxForwardSetupSpeedMps);
            const double lateral_speed = std::clamp(
                -kDribbleLateralGain * command_lateral_error,
                -kDribbleMaxLateralSetupSpeedMps,
                kDribbleMaxLateralSetupSpeedMps);
            const std::array<double, 2> velocity_world{
                direction[0] * forward_speed +
                    perpendicular[0] * lateral_speed,
                direction[1] * forward_speed +
                    perpendicular[1] * lateral_speed,
            };
            WalkCommand precision_command;
            precision_command.target_2d_m =
                math::rotate_2d(velocity_world, -self_yaw_deg);
            precision_command.target_absolute = false;
            precision_command.orientation_deg = absolute_direction_deg;
            precision_command.orientation_absolute = true;
            precision_command.orientation_gain = kKickSetupOrientationGain;
            precision_command.role_id = motion_role_id;
            return precision_command;
        }
        trace_setup_gate(3, "approach");
        WalkCommand approach_command = make_walk_command_avoiding(
            approach_target, context.snapshot, std::nullopt, true, false,
            motion_role_id, false, false);
        approach_command.orientation_deg = absolute_direction_deg;
        approach_command.orientation_absolute = true;
        return approach_command;
    }

    const bool contact_state_stable =
        context.snapshot.ball.position_valid && latched_position_ready &&
        orientation_error_deg <= required_orientation_error_deg &&
        planar_speed_mps <= kKickSetupMaximumPlanarSpeedMps;
    if (contact_state_stable) {
        if (context.state.kick_setup_stable_since_s <= 0.0) {
            context.state.kick_setup_stable_since_s = now;
        }
    } else {
        context.state.kick_setup_stable_since_s = 0.0;
    }
    const bool contact_state_confirmed =
        context.state.kick_setup_stable_since_s > 0.0 &&
        now - context.state.kick_setup_stable_since_s >=
            (use_procedural_range_pass
                ? 0.0
                : use_procedural_kick
                    ? kProceduralKickSetupStableHoldS
                    : kKickSetupStableHoldS);
    const bool legal_kick =
        (context.snapshot.play_mode == world::PlayMode::PlayOn ||
         context.snapshot.play_mode_group == world::PlayModeGroup::OurKick) &&
        context.snapshot.ball.position_valid &&
        now >= context.state.next_kick_allowed_s &&
        context.ball_distance >= kKickMinBallDistanceM &&
        context.ball_distance <= kKickMaxBallDistanceM &&
        orientation_error_deg <= required_orientation_error_deg &&
        contact_state_confirmed;
    // Readiness is a hard release gate. Once dribble_ready is latched the
    // robot can drift slightly inside the valid contact envelope; gating only
    // on setup_ready here would otherwise permit a later unacknowledged kick.
    if (!allow_kick) {
        // A passer that is waiting for Ready must not keep driving through the
        // ball. The position gate above returns to precision alignment after
        // any drift; while it remains valid, command zero translation and only
        // close the heading error so no unannounced dribble invalidates it.
        trace_setup_gate(4, "wait-receiver");
        WalkCommand hold_command = make_walk_command(context.self);
        hold_command.orientation_deg = absolute_direction_deg;
        hold_command.orientation_absolute = true;
        hold_command.orientation_gain = kKickSetupOrientationGain;
        hold_command.role_id = motion_role_id;
        return hold_command;
    }
    if (legal_kick) {
        trace_setup_gate(5, "release");
        context.state.kick_pre_settling = false;
        context.state.kick_pre_settle_stable_since_s = 0.0;
        context.state.kick_active_until_s = now + kKickDurationS;
        context.state.next_kick_allowed_s =
            context.state.kick_active_until_s + kKickCooldownS;
        KickCommand kick_command;
        // If the specialized runner rejects its live joint phase after the
        // bounded setup timeout, MotionManager may execute the explicit
        // original forward-contact fallback instead of dropping the action.
        // Before that timeout a rejection remains visible and fail-closed.
        kick_command.allow_forward_contact_fallback = fallback_due;
        if (cooperative_action != nullptr &&
            cooperative_action->category == strategy::ActionCategory::Pass) {
            kick_command.target_point_m = cooperative_action->target_point_m;
            kick_command.requested_ball_speed_mps =
                cooperative_action->requested_ball_speed_mps;
            kick_command.receiver_player_number =
                cooperative_action->target_player_number;
            kick_command.action_id = cooperative_action->action_id;
            kick_command.sequence_id = cooperative_action->sequence_id;
            kick_command.mode = KickMode::TargetedPass;
        } else if (use_procedural_shot) {
            kick_command.target_point_m = cooperative_action->target_point_m;
            kick_command.requested_ball_speed_mps =
                cooperative_action->requested_ball_speed_mps;
            kick_command.action_id = cooperative_action->action_id;
            kick_command.mode = KickMode::Shot;
        } else if (use_procedural_clear) {
            kick_command.target_point_m = cooperative_action->target_point_m;
            kick_command.requested_ball_speed_mps =
                cooperative_action->requested_ball_speed_mps;
            kick_command.action_id = cooperative_action->action_id;
            kick_command.mode = KickMode::Clear;
        } else if (use_procedural_dribble) {
            kick_command.target_point_m = cooperative_action != nullptr
                ? std::optional<std::array<double, 2>>{
                      cooperative_action->target_point_m}
                : std::optional<std::array<double, 2>>{
                      std::array<double, 2>{
                          context.ball[0] + direction[0] * 0.55,
                          context.ball[1] + direction[1] * 0.55}};
            kick_command.requested_ball_speed_mps =
                decision::kick_contract::kProceduralDribbleRequestedSpeedMps;
            if (cooperative_action != nullptr) {
                kick_command.action_id = cooperative_action->action_id;
            }
            kick_command.mode = KickMode::DribbleTouch;
        }
        if (restart_plan != nullptr) {
            kick_command.restart_epoch = restart_plan->epoch;
            kick_command.restart_revision = restart_plan->revision;
        }
        context.state.active_kick_command = kick_command;
        return kick_command;
    }

    if (fallback_allowed) {
        return make_fallback_command();
    }

    if (context.state.dribble_ready) {
        // Alignment phase: once inside the trained contact envelope, stop
        // translating and turn in place. Continuing toward push_target here
        // drove through the ball before the 2-degree release gate could close.
        // The procedural anchor was calibrated from a high-gain neutral
        // stance. Holding it here also removes zero-command gait sway, making
        // the 0.25 s release debounce physically achievable without relaxing
        // the validated ball-slot or body-speed contracts.
        if (orientation_error_deg > required_orientation_error_deg) {
            trace_setup_gate(6, "turn-in-place");
            WalkCommand align_command = make_walk_command(context.self);
            align_command.orientation_deg = absolute_direction_deg;
            align_command.orientation_absolute = true;
            align_command.orientation_gain = kKickSetupOrientationGain;
            align_command.role_id = motion_role_id;
            return align_command;
        }
        trace_setup_gate(7, "settle");
        if (use_procedural_kick) {
            return NeutralCommand{};
        }
        WalkCommand align_command = make_walk_command(context.self);
        align_command.orientation_deg = absolute_direction_deg;
        align_command.orientation_absolute = true;
        align_command.orientation_gain = kKickSetupOrientationGain;
        align_command.role_id = motion_role_id;
        return align_command;
    }

    trace_setup_gate(8, "push-fallback");
    return make_walk_command_avoiding(
        push_target, context.snapshot, std::nullopt, true, true,
        motion_role_id, false, false);
}

HighLevelCommand make_ap_push_ball_to_goal(APDecisionContext& context) {
    // During our-kick set plays push diagonally toward the relay teammate's
    // side (45° off the ball→goal axis) instead of straight at the opponent
    // goal — the relay takes the ball cleanly without the AP chasing its own
    // pass (double-touch). Outside our-kick (PlayOn, their-kick) keep aiming
    // at the goal.
    std::array<double, 2> goal_direction;
    if (context.snapshot.play_mode_group == world::PlayModeGroup::OurKick) {
        const std::array<double, 2> their_goal = field_geometry::actual_their_goal_center_target();
        const std::array<double, 2> goal_dir = math::vec2_unit_or(
            math::vec2_sub(their_goal, context.ball), {1.0, 0.0});
        // vector_angle_deg is atan2, so the relay distance cancels — rotating
        // goal_dir by 45° gives the diagonal push heading directly.
        goal_direction = math::rotate_2d(goal_dir, 45.0);
    } else {
        const std::array<double, 2> their_goal = field_geometry::actual_their_goal_center_target();
        goal_direction = math::vec2_sub(their_goal, context.ball);
    }
    const double absolute_direction_deg = math::norm2(goal_direction) > 1e-6
        ? math::vector_angle_deg(goal_direction)
        : 0.0;
    return make_dribble_command(context, absolute_direction_deg);
}

bool pass_commit_is_valid(
    const world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& pass,
    const strategy::ActionCapabilityRegistry& capabilities) {
    if (pass.category != strategy::ActionCategory::Pass ||
        pass.target_player_number <= 0 ||
        pass.target_player_number == snapshot.player_number) {
        return false;
    }
    const auto receiver_index = static_cast<std::size_t>(pass.target_player_number - 1);
    if (receiver_index >= snapshot.teammates.size()) return false;
    const auto& receiver = snapshot.teammates[receiver_index];
    // The planner required a usable receiver observation before creating this
    // commitment. Preserve the bounded proposal through later occlusion: the
    // passer still cannot release until a matching receiver-authored Ready
    // packet arrives, and the six-second commit deadline remains authoritative.
    // A positively observed fall is the one receiver-state exception.
    if (receiver.fallen) return false;
    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    strategy::CooperativeAction current_request = pass;
    current_request.start_ball_point_m = ball;
    return std::abs(pass.target_point_m[0]) < field_geometry::kActualHalfLengthM - 0.5 &&
           std::abs(pass.target_point_m[1]) < field_geometry::kActualHalfWidthM - 0.5 &&
           math::planar_dist(ball, pass.start_ball_point_m) <= 0.75 &&
           capabilities.supported(current_request);
}

void publish_pass_lifecycle(
    const PassLifecycle& lifecycle,
    Blackboard& blackboard) {
    if (const auto* action = lifecycle.action(); action != nullptr) {
        blackboard.set(
            Blackboard::kKeySelectedCooperativeAction, *action);
    }
    if (const auto outgoing = lifecycle.outgoing(); outgoing.has_value()) {
        blackboard.set(Blackboard::kKeyOutgoingPassIntent, *outgoing);
    }
}

const comm::PassIntentRecord* latest_pass_intent(
    const world::WorldSnapshot& snapshot) {
    const comm::PassIntentRecord* best = nullptr;
    for (const auto& intent : snapshot.team_comm_snapshot.pass_intents) {
        if (intent.author != comm::PassIntentAuthor::Passer ||
            intent.sender_player_number != intent.passer_player_number ||
            intent.receiver_player_number != snapshot.player_number ||
            intent.passer_player_number == snapshot.player_number) {
            continue;
        }
        if (std::abs(intent.target_x_m) >= field_geometry::kActualHalfLengthM ||
            std::abs(intent.target_y_m) >= field_geometry::kActualHalfWidthM) {
            continue;
        }
        if (best == nullptr || intent.server_cycle > best->server_cycle) {
            best = &intent;
        }
    }
    return best;
}

std::array<double, 2> receive_intercept_target(
    const world::WorldSnapshot& snapshot,
    const comm::PassIntentRecord& intent) {
    const std::array<double, 2> planned{
        intent.target_x_m, intent.target_y_m};
    const bool ball_in_motion =
        intent.state == comm::PassIntentState::Commanded ||
        intent.state == comm::PassIntentState::Executed ||
        intent.state == comm::PassIntentState::ReceiverZone;
    if (!ball_in_motion || !snapshot.ball.position_valid ||
        !snapshot.ball.velocity_valid) {
        return planned;
    }
    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const std::array<double, 2> velocity{
        snapshot.ball.velocity_mps[0], snapshot.ball.velocity_mps[1]};
    const double speed_mps = math::norm2(velocity);
    if (speed_mps < 0.20) return ball;

    const std::array<double, 2> direction = math::vec2_scale(
        velocity, 1.0 / speed_mps);
    const std::array<double, 2> self{
        snapshot.self.position_m[0], snapshot.self.position_m[1]};
    const strategy::ReachTimeModel reach_model(
        strategy::ReachTimeModel::Parameters{
            0.90, 120.0, 0.20, 0.35, 0.15});
    constexpr double kRollingDecelerationMps2 = 0.30;
    for (double arrival_s = 0.20; arrival_s <= 2.40; arrival_s += 0.20) {
        const double travel_s = std::min(
            arrival_s, speed_mps / kRollingDecelerationMps2);
        const double distance_m = std::max(
            0.0,
            speed_mps * travel_s -
                0.5 * kRollingDecelerationMps2 * travel_s * travel_s);
        std::array<double, 2> candidate = math::vec2_add(
            ball, math::vec2_scale(direction, distance_m));
        candidate[0] = std::clamp(
            candidate[0],
            -field_geometry::kActualHalfLengthM + 0.5,
            field_geometry::kActualHalfLengthM - 0.5);
        candidate[1] = std::clamp(
            candidate[1],
            -field_geometry::kActualHalfWidthM + 0.5,
            field_geometry::kActualHalfWidthM - 0.5);
        if (reach_model.estimate_s(self, candidate) <= arrival_s + 0.10) {
            return candidate;
        }
    }
    return planned;
}

bool is_gk_our_goal_kick(const GKDecisionContext& context) {
    return context.snapshot.play_mode == world::PlayMode::OurGoalKick;
}

}  // namespace

int current_role_from_blackboard(const Blackboard& blackboard) {
    return blackboard.exists(Blackboard::kKeyCurrentRole)
        ? blackboard.get<int>(Blackboard::kKeyCurrentRole)
        : -1;
}

bool APBehavior::matches(const Blackboard& blackboard) const {
    return match_role(blackboard, RoleManager::ROLE_AP);
}

void APBehavior::apply_execution_feedback(
    const ExecutionFeedback& feedback) const {
    state_.pass_lifecycle.apply_execution_feedback(feedback);
    if (feedback.request_kind != MotionRequestKind::Kick ||
        !is_failure(feedback.status) ||
        !state_.active_kick_command.has_value()) {
        return;
    }

    const auto& active_kick = *state_.active_kick_command;
    if ((feedback.cooperative_action_id.has_value() &&
         active_kick.action_id != *feedback.cooperative_action_id) ||
        (feedback.sequence_id.has_value() &&
         active_kick.sequence_id != *feedback.sequence_id) ||
        (active_kick.action_id != 0U &&
         !feedback.cooperative_action_id.has_value())) {
        return;
    }

    const bool matching_pass =
        active_kick.mode == KickMode::TargetedPass &&
        state_.committed_pass.has_value() &&
        state_.committed_pass->action_id == active_kick.action_id &&
        state_.committed_pass->sequence_id == active_kick.sequence_id;
    state_.active_kick_command.reset();
    state_.kick_active_until_s = 0.0;
    state_.dribble_ready = false;
    state_.kick_pre_settling = false;
    state_.kick_pre_settle_stable_since_s = 0.0;
    state_.kick_setup_stable_since_s = 0.0;
    if (matching_pass) {
        state_.pass_commit_until_s = 0.0;
        state_.pass_retry_after_s = std::max(
            state_.pass_retry_after_s,
            feedback.server_time + kRejectedPassRetryDelayS);
    }
}

HighLevelCommand APBehavior::make_command(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard,
    RoleManager& role_manager,
    bool enable_pass_strategy,
    bool enable_targeted_kick) const {
    static const APNodePtr ap_tree = bt::command<APDecisionContext>(make_ap_push_ball_to_goal);

    if (!is_our_set_play(snapshot)) {
        state_.set_play_released = false;
    }

    APDecisionContext context{
        snapshot,
        state_,
        enable_targeted_kick,
        {snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
        {snapshot.self.position_m[0], snapshot.self.position_m[1]},
        0.0};
    context.ball_distance = math::planar_dist(context.ball, context.self);
    if (snapshot.play_mode != world::PlayMode::PlayOn &&
        state_.pass_lifecycle.active() &&
        !state_.pass_lifecycle.terminal()) {
        state_.pass_lifecycle.cancel(snapshot.server_time);
    }
    state_.pass_lifecycle.update(snapshot);
    if (state_.pass_lifecycle.terminal() &&
        state_.pass_lifecycle.state() != comm::PassIntentState::Received &&
        state_.pass_retry_after_s <= 0.0) {
        // A proposal that received no Ready response must not immediately
        // monopolize the ball again. Arm this delay exactly once per pass;
        // extending it on every terminal-state tick would make retries
        // impossible while the outcome is retained for team broadcast.
        state_.pass_retry_after_s =
            snapshot.server_time + kRejectedPassRetryDelayS;
    }
    if (state_.pass_lifecycle.ready_to_clear(snapshot.server_time)) {
        state_.pass_lifecycle.reset();
        state_.committed_pass.reset();
        state_.active_kick_command.reset();
        state_.pass_commit_until_s = 0.0;
    }
    publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
    // TeamTactics already assigned a phase/risk-aware AP target. Preserve it
    // so ProtectLead can cover a lane instead of being silently overwritten by
    // the legacy pressure default. Direct unit callers without a team plan
    // still receive the historical pressure behavior.
    if (!blackboard.exists(Blackboard::kKeyTacticalTarget)) {
        blackboard.set(
            Blackboard::kKeyTacticalTarget,
            TacticalTarget{TacticalDuty::Pressure, context.ball, context.ball, 0, 1.0});
    }

    if (is_our_set_play(snapshot)) {
        if (const auto* restart = restart_decision_from_blackboard(blackboard);
            restart != nullptr) {
            synchronize_restart_contact_state(state_, *restart);
            if (!restart->plan.has_value() || !restart->self_is_taker ||
                restart->self_locked_out ||
                restart->phase == RestartPhase::Complete) {
                return make_walk_command_avoiding(
                    role_position_from_blackboard(blackboard), snapshot,
                    std::nullopt, true, true, RoleManager::ROLE_AP);
            }
            return make_dribble_command(
                context,
                restart->plan->contact_direction_deg,
                nullptr,
                restart->execution_authorized,
                RoleManager::ROLE_AP,
                &*restart->plan);
        }
    }

    const strategy::ActionCapabilityRegistry capabilities(enable_targeted_kick);
    if (enable_targeted_kick &&
        snapshot.play_mode == world::PlayMode::PlayOn) {
        const strategy::TacticalState tactical_state = blackboard.exists(
                Blackboard::kKeyTeamPlan)
            ? blackboard.get<TeamPlan>(
                  Blackboard::kKeyTeamPlan).tactical_state
            : strategy::build_tactical_state(snapshot);
        strategy::PlanningResult plan = action_planner_.plan(
            snapshot, capabilities, enable_pass_strategy, tactical_state);
        const double planar_speed_mps = math::norm2({
            snapshot.self.lin_vel_b[0], snapshot.self.lin_vel_b[1]});
        constexpr double kSpecialistSetupMaximumBallDistanceM = 0.65;
        constexpr double kSpecialistSetupMaximumSpeedMps = 0.35;
        constexpr double kSpecialistMinimumOpponentEtaS = 1.0;
        const bool controlled_specialist_setup =
            context.ball_distance <= kSpecialistSetupMaximumBallDistanceM &&
            planar_speed_mps <= kSpecialistSetupMaximumSpeedMps &&
            tactical_state.possession == strategy::PossessionOwner::Ours &&
            tactical_state.ball_owner_is_teammate &&
            tactical_state.ball_owner_player_number == snapshot.player_number &&
            tactical_state.nearest_opponent_ball_time_s >=
                kSpecialistMinimumOpponentEtaS;
        if (plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Pass &&
            snapshot.server_time < state_.pass_retry_after_s) {
            const auto fallback = std::find_if(
                plan.candidates.begin(), plan.candidates.end(),
                [](const strategy::CooperativeAction& candidate) {
                    return candidate.category !=
                               strategy::ActionCategory::Pass &&
                           candidate.category !=
                               strategy::ActionCategory::Hold;
                });
            if (fallback != plan.candidates.end()) {
                plan.selected = *fallback;
            } else {
                const auto hold = std::find_if(
                    plan.candidates.begin(), plan.candidates.end(),
                    [](const strategy::CooperativeAction& candidate) {
                        return candidate.category ==
                            strategy::ActionCategory::Hold;
                    });
                if (hold == plan.candidates.end()) {
                    plan.selected.reset();
                } else {
                    plan.selected = *hold;
                }
            }
        }
        blackboard.set(Blackboard::kKeyStrategyPlan, plan);

        const auto lifecycle_state = state_.pass_lifecycle.state();
        const bool awaiting_release =
            lifecycle_state == comm::PassIntentState::Proposed ||
            lifecycle_state == comm::PassIntentState::Committed;
        if (state_.committed_pass.has_value() && awaiting_release &&
            (snapshot.server_time >= state_.pass_commit_until_s ||
             !controlled_specialist_setup ||
             !pass_commit_is_valid(
                 snapshot, *state_.committed_pass, capabilities))) {
            state_.pass_lifecycle.cancel(snapshot.server_time);
            state_.pass_commit_until_s = 0.0;
        }

        // Keep broadcasting a terminal pass outcome, but do not freeze the
        // ball owner while doing so.  The selected-action rewrite above has
        // already excluded a new pass during the retry window, allowing the
        // original local dribble/move fallback to resume immediately.
        if (state_.pass_lifecycle.terminal()) {
            publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
        }
        const bool kick_still_active =
            state_.active_kick_command.has_value() &&
            snapshot.server_time < state_.kick_active_until_s;
        const bool tracking_outcome =
            lifecycle_state == comm::PassIntentState::Executed ||
            lifecycle_state == comm::PassIntentState::ReceiverZone ||
            (lifecycle_state == comm::PassIntentState::Commanded &&
             !kick_still_active);
        if (tracking_outcome) {
            publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
            const TacticalTarget target = blackboard.exists(
                    Blackboard::kKeyTacticalTarget)
                ? blackboard.get<TacticalTarget>(
                      Blackboard::kKeyTacticalTarget)
                : TacticalTarget{
                      TacticalDuty::Formation,
                      role_position_from_blackboard(blackboard),
                      context.ball,
                      0,
                      0.5};
            return make_walk_command_avoiding(
                target.position_m, snapshot, std::nullopt,
                true, true, RoleManager::ROLE_AP);
        }

        constexpr double kPassCommitDurationS = 6.0;
        if (!state_.committed_pass.has_value() && plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Pass &&
            snapshot.server_time >= state_.pass_retry_after_s &&
            controlled_specialist_setup &&
            capabilities.supported(*plan.selected) &&
            context.ball_distance <= kSpecialistSetupMaximumBallDistanceM) {
            strategy::CooperativeAction committed = *plan.selected;
            state_.next_pass_sequence_id = static_cast<std::uint8_t>(
                state_.next_pass_sequence_id + 1U);
            if (state_.next_pass_sequence_id == 0U) {
                state_.next_pass_sequence_id = 1U;
            }
            committed.sequence_id = state_.next_pass_sequence_id;
            state_.committed_pass = committed;
            state_.pass_commit_until_s = snapshot.server_time + kPassCommitDurationS;
            state_.pass_retry_after_s = 0.0;
            state_.pass_lifecycle.start(committed, snapshot.server_time);
        }

        if (state_.committed_pass.has_value() &&
            !state_.pass_lifecycle.terminal()) {
            publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
            const auto target_direction = math::vec2_sub(
                state_.committed_pass->target_point_m, context.ball);
            if (math::norm2(target_direction) > 1.0e-6) {
                const double direction_deg = math::vector_angle_deg(target_direction);
                HighLevelCommand command = make_dribble_command(
                    context,
                    direction_deg,
                    &*state_.committed_pass,
                    state_.pass_lifecycle.release_authorized());
                if (const auto* kick = std::get_if<KickCommand>(&command);
                    kick != nullptr) {
                    state_.pass_lifecycle.mark_commanded(*kick, snapshot);
                    publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
                }
                return command;
            }
            state_.pass_lifecycle.cancel(snapshot.server_time);
            state_.pass_commit_until_s = 0.0;
        }

        if (plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Hold) {
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *plan.selected);
            return NeutralCommand{};
        }
        if (plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Move) {
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *plan.selected);
            return make_walk_command_avoiding(
                plan.selected->target_point_m, snapshot, std::nullopt,
                true, true, RoleManager::ROLE_AP);
        }
        if (plan.selected.has_value() &&
            plan.selected->category != strategy::ActionCategory::Pass &&
            controlled_specialist_setup) {
            const auto target_direction = math::vec2_sub(
                plan.selected->target_point_m, context.ball);
            if (math::norm2(target_direction) > 1.0e-6) {
                blackboard.set(
                    Blackboard::kKeySelectedCooperativeAction,
                    *plan.selected);
                const double direction_deg =
                    math::vector_angle_deg(target_direction);
                return make_dribble_command(
                    context, direction_deg, &*plan.selected);
            }
        }
        // No calm, self-owned release opportunity exists. Keep the action
        // proposal in telemetry, but execute the original Apollo contact path
        // below. This preserves ball-chase tempo under pressure instead of
        // starting a centimetre-scale setup several metres from the ball.
        context.procedural_kick_enabled = false;
    } else {
        if (state_.pass_lifecycle.active() &&
            !state_.pass_lifecycle.terminal()) {
            state_.pass_lifecycle.cancel(snapshot.server_time);
            publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
        }
        state_.pass_commit_until_s = 0.0;
    }

    if (is_our_set_play(snapshot) && state_.set_play_released) {
        return make_walk_command_avoiding(
            role_position_from_blackboard(blackboard), snapshot, std::nullopt,
            true, true, RoleManager::ROLE_AP);
    }

    const double previous_ball_distance = state_.previous_ball_distance;
    const auto result = ap_tree->tick(context);
    state_.previous_ball_distance = context.ball_distance;
    if (is_our_set_play(snapshot) && state_.dribble_ready &&
        context.ball_distance > field_geometry::kPushBallEngageDistanceM &&
        previous_ball_distance <= field_geometry::kPushBallEngageDistanceM &&
        snapshot.ball.velocity_valid &&
        math::norm2({snapshot.ball.velocity_mps[0], snapshot.ball.velocity_mps[1]}) >=
            field_geometry::kSetPlayRelayBallSpeedMps) {
        state_.set_play_released = true;
        state_.dribble_ready = false;
        role_manager.mark_self_set_play_pushed(snapshot.player_number, snapshot);
    }
    return result.command.value_or(NeutralCommand{});
}

bool SimpleRoleBehavior::matches(const Blackboard& blackboard) const {
    return match_role(blackboard, role_id_);
}

void SimpleRoleBehavior::reset_state() const {
    relay_state_ = {};
    receive_intent_.reset();
    receive_intent_until_s_ = 0.0;
}

HighLevelCommand SimpleRoleBehavior::make_command(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard) const {
    if (snapshot.play_mode == world::PlayMode::PlayOn) {
        if (const auto* intent = latest_pass_intent(snapshot); intent != nullptr) {
            if (is_terminal_pass_state(intent->state)) {
                if (receive_intent_.has_value() &&
                    receive_intent_->passer_player_number ==
                        intent->passer_player_number &&
                    receive_intent_->sequence_id == intent->sequence_id) {
                    receive_intent_.reset();
                    receive_intent_until_s_ = 0.0;
                }
            } else {
                receive_intent_ = *intent;
                receive_intent_until_s_ = snapshot.server_time + std::clamp(
                    intent->predicted_ball_time_s + 1.5, 2.0, 6.0);
            }
        }
        if (receive_intent_.has_value() && snapshot.ball.position_valid) {
            const double ball_speed_mps = snapshot.ball.velocity_valid
                ? math::norm2({
                      snapshot.ball.velocity_mps[0],
                      snapshot.ball.velocity_mps[1]})
                : 0.0;
            const std::array<double, 2> self{
                snapshot.self.position_m[0], snapshot.self.position_m[1]};
            const std::array<double, 2> ball{
                snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
            if (math::planar_dist(self, ball) <= 0.65 &&
                ball_speed_mps <= 1.5) {
                // TeamCommManager emits the matching Received acknowledgement
                // from the same physical evidence. Locally release the run so
                // the next role assignment can hand possession to AP.
                receive_intent_.reset();
                receive_intent_until_s_ = 0.0;
            }
        }
        if (receive_intent_.has_value() &&
            snapshot.server_time <= receive_intent_until_s_) {
            const auto receive_target = receive_intercept_target(
                snapshot, *receive_intent_);
            blackboard.set(
                Blackboard::kKeyTacticalTarget,
                TacticalTarget{
                    TacticalDuty::Receive,
                    receive_target,
                    std::array<double, 2>{
                        snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
                    0,
                    0.9});
            return make_walk_command_avoiding(
                receive_target, snapshot, std::nullopt,
                true, true, role_id_, false);
        }
        if (snapshot.server_time > receive_intent_until_s_) {
            receive_intent_.reset();
            receive_intent_until_s_ = 0.0;
        }
    } else {
        receive_intent_.reset();
        receive_intent_until_s_ = 0.0;
    }
    if (role_id_ == RoleManager::ROLE_ST && is_our_set_play(snapshot)) {
        if (!snapshot.ball.velocity_valid) {
            relay_state_ = {};
        } else {
            const double ball_speed = math::norm2(
                {snapshot.ball.velocity_mps[0], snapshot.ball.velocity_mps[1]});
            if (ball_speed >= field_geometry::kSetPlayRelayBallSpeedMps) {
                APDecisionContext relay_context{
                    snapshot,
                    relay_state_,
                    false,
                    {snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
                    {snapshot.self.position_m[0], snapshot.self.position_m[1]},
                    0.0};
                relay_context.ball_distance = math::planar_dist(relay_context.ball, relay_context.self);
                return make_dribble_command(relay_context, 0.0);
            }
        }
    } else {
        relay_state_ = {};
    }

    std::optional<double> opponent_x_threshold;
    if (defensive_opponent_clip_) {
        opponent_x_threshold = snapshot.ball.position_m[0] - 1.0;
    }
    const TacticalTarget tactical_target = blackboard.exists(
            Blackboard::kKeyTacticalTarget)
        ? blackboard.get<TacticalTarget>(Blackboard::kKeyTacticalTarget)
        : TacticalTarget{
              TacticalDuty::Formation,
              role_position_from_blackboard(blackboard),
              std::nullopt,
              0,
              0.0};
    blackboard.set(Blackboard::kKeyTacticalTarget, tactical_target);
    if (tactical_target.duty != TacticalDuty::Formation) {
        opponent_x_threshold.reset();
    }
    WalkCommand command = make_walk_command_avoiding(
        tactical_target.position_m, snapshot, opponent_x_threshold);
    const std::array<double, 2> self{
        snapshot.self.position_m[0], snapshot.self.position_m[1]};
    if (tactical_target.face_point_m.has_value() &&
        math::planar_dist(self, tactical_target.position_m) <=
            kWalkStopRadiusM) {
        command.orientation_deg = orientation_to_point_from_self(
            snapshot, *tactical_target.face_point_m);
        command.orientation_absolute = true;
    }
    return command;
}

bool GKBehavior::matches(const Blackboard& blackboard) const {
    return match_role(blackboard, RoleManager::ROLE_GK);
}

HighLevelCommand GKBehavior::make_command(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard) const {
    return make_command(snapshot, blackboard, false);
}

void GKBehavior::apply_execution_feedback(
    const ExecutionFeedback& feedback) const {
    if (feedback.request_kind == MotionRequestKind::Kick &&
        is_failure(feedback.status)) {
        clearance_state_ = {};
    }
}

HighLevelCommand GKBehavior::make_command(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard,
    bool enable_targeted_kick) const {
    GKDecisionContext context{
        snapshot,
        {snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
        {snapshot.self.position_m[0], snapshot.self.position_m[1]},
        0.0};
    context.ball_distance = math::planar_dist(context.ball, context.self);

    if (is_gk_our_goal_kick(context)) {
        blackboard.set(
            Blackboard::kKeyTacticalTarget,
            TacticalTarget{
                TacticalDuty::Pressure, context.ball, context.ball, 0, 0.8});
        APDecisionContext clearance_context{
            snapshot,
            clearance_state_,
            false,
            context.ball,
            context.self,
            context.ball_distance};
        if (const auto* restart = restart_decision_from_blackboard(blackboard);
            restart != nullptr) {
            synchronize_restart_contact_state(clearance_state_, *restart);
            if (!restart->plan.has_value() || !restart->self_is_taker ||
                restart->self_locked_out ||
                restart->phase == RestartPhase::Complete) {
                return make_walk_command_avoiding(
                    role_position_from_blackboard(blackboard), snapshot,
                    std::nullopt, true, true, RoleManager::ROLE_GK, false);
            }
            return make_dribble_command(
                clearance_context,
                restart->plan->contact_direction_deg,
                nullptr,
                restart->execution_authorized,
                RoleManager::ROLE_GK,
                &*restart->plan);
        }
        return make_dribble_command(
            clearance_context, 0.0, nullptr, true, RoleManager::ROLE_GK);
    }

    const TacticalTarget tactical_target = blackboard.exists(
            Blackboard::kKeyTacticalTarget)
        ? blackboard.get<TacticalTarget>(Blackboard::kKeyTacticalTarget)
        : TacticalTarget{
              TacticalDuty::Formation,
              role_position_from_blackboard(blackboard),
              std::nullopt,
              0,
              0.0};
    blackboard.set(Blackboard::kKeyTacticalTarget, tactical_target);

    // A successful smother must hand the ball back to open play. The same
    // action planner and exact capability contract used by the active player
    // select a forward safety clear; no procedural/learned capability means
    // this branch remains a walk/hold instead of inventing contact.
    if (snapshot.play_mode == world::PlayMode::PlayOn &&
        tactical_target.duty == TacticalDuty::GoalkeeperSmother &&
        snapshot.ball.position_valid) {
        APDecisionContext clearance_context{
            snapshot,
            clearance_state_,
            enable_targeted_kick,
            context.ball,
            context.self,
            context.ball_distance};
        if (clearance_state_.kick_active_until_s > snapshot.server_time) {
            return make_dribble_command(
                clearance_context, 0.0, nullptr, true,
                RoleManager::ROLE_GK);
        }

        // Inside the last-line body-block radius, walking or turning to set
        // up a clear is worse than remaining upright. A real match emergency
        // ForwardContact moved the ball only 0.14 m, toppled the keeper, and
        // conceded. Hold the blocking pose until the ball leaves this radius
        // or a contact action was already committed above.
        constexpr double kGoalkeeperBodyBlockDistanceM = 0.45;
        if (context.ball[0] <=
                -field_geometry::kActualHalfLengthM + 1.0 &&
            context.ball_distance <= kGoalkeeperBodyBlockDistanceM) {
            return NeutralCommand{};
        }

        const strategy::ActionCapabilityRegistry capabilities(
            enable_targeted_kick);
        const strategy::TacticalState tactical_state = blackboard.exists(
                Blackboard::kKeyTeamPlan)
            ? blackboard.get<TeamPlan>(
                  Blackboard::kKeyTeamPlan).tactical_state
            : strategy::build_tactical_state(snapshot);
        const strategy::PlanningResult plan = action_planner_.plan(
            snapshot, capabilities, false, tactical_state);
        blackboard.set(Blackboard::kKeyStrategyPlan, plan);
        // First close down the goal-bound ball. Starting the centimetre-scale
        // clear setup from metres away made the keeper stop defending while
        // an opponent walked the ball over the line.
        constexpr double kGoalkeeperClearEngageDistanceM = 0.55;
        if (plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Clear &&
            capabilities.supported(*plan.selected) &&
            context.ball_distance <= kGoalkeeperClearEngageDistanceM) {
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *plan.selected);
            const auto target_direction = math::vec2_sub(
                plan.selected->target_point_m, context.ball);
            if (math::norm2(target_direction) > 1.0e-6) {
                return make_dribble_command(
                    clearance_context,
                    math::vector_angle_deg(target_direction),
                    &*plan.selected,
                    true,
                    RoleManager::ROLE_GK);
            }
        }
    } else {
        clearance_state_ = {};
    }

    WalkCommand command = make_walk_command_avoiding(
        tactical_target.position_m, snapshot, std::nullopt,
        true, true, RoleManager::ROLE_GK, false);
    if (tactical_target.face_point_m.has_value() &&
        math::planar_dist(context.self, tactical_target.position_m) <=
            kWalkStopRadiusM) {
        command.orientation_deg = orientation_to_point_from_self(
            snapshot, *tactical_target.face_point_m);
        command.orientation_absolute = true;
    }
    return command;
}

std::optional<HighLevelCommand> RoleBehaviorSet::select(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard,
    RoleManager& role_manager,
    bool enable_pass_strategy,
    bool enable_targeted_kick) const {
    // AP is the only behavior that needs RoleManager (to latch the set-play
    // push); dispatch it directly and let the other behaviors share the
    // 2-param base interface.
    if (ap_.matches(blackboard)) {
        return ap_.make_command(
            snapshot, blackboard, role_manager, enable_pass_strategy,
            enable_targeted_kick);
    }
    if (gk_.matches(blackboard)) {
        return gk_.make_command(
            snapshot, blackboard, enable_targeted_kick);
    }
    const std::array<const RoleBehavior*, 5> behaviors{
        &cbm_,
        &st_,
        &cbl_,
        &cbr_,
        &cdm_,
    };

    for (const auto* behavior : behaviors) {
        if (behavior->matches(blackboard)) {
            return behavior->make_command(snapshot, blackboard);
        }
    }
    return std::nullopt;
}

void RoleBehaviorSet::reset() const {
    ap_.reset_state();
    cbm_.reset_state();
    st_.reset_state();
    cbl_.reset_state();
    cbr_.reset_state();
    cdm_.reset_state();
    gk_.reset_state();
}

void RoleBehaviorSet::apply_execution_feedback(
    const ExecutionFeedback& feedback) const {
    ap_.apply_execution_feedback(feedback);
    gk_.apply_execution_feedback(feedback);
}

}  // namespace decision
