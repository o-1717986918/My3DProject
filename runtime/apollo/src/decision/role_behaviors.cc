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
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace decision {

namespace {

constexpr double kWalkMaxSpeedM = 3.0;
constexpr double kWalkBrakeDecelMps2 = 1.0;
constexpr double kWalkStopRadiusM = 0.15;
constexpr double kWalkHeadingSlowStartDeg = 35.0;
constexpr double kWalkHeadingStopDeg = 95.0;
// At or below this distance to the target a ball-facing player strafes (holds
// ball-facing, no heading-error speed penalty) to fine-tune its slot; beyond it
// the player turns to face its travel direction and runs forward at full speed.
constexpr double kStrafeMaxDistM = 2.5;
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
// Five consecutive 50 Hz neutral cycles are sufficient for the standalone
// runner because it repeats the measured body/joint velocity guards at begin.
// A longer hold lets residual translational momentum carry the robot through
// the narrow server-calibrated ball slot before release.
constexpr double kProceduralKickSetupStableHoldS = 0.08;
constexpr double kKickSetupMaximumPlanarSpeedMps = 0.20;
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
constexpr double kRejectedPassRetryDelayS = 0.5;

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
            0.1,
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

    // Orientation vs. travel policy. Facing the ball keeps a player ready to
    // react, but holding ball-facing while travelling elsewhere means it can only
    // strafe/backpedal — and the heading-error speed penalty throttles that to
    // ~10% (~0.3 m/s). So split by travel distance:
    //   - strafe mode (close to the target, or a forced sideways mover such as
    //     the keeper): face the ball and drop the heading penalty, so short
    //     lateral adjustments run at the walk's natural lateral cap;
    //   - fast-travel mode (far from the target): face the travel direction and keep
    //     the penalty, so the player turns and walks forward at full speed, then
    //     re-faces the ball as it arrives (dist falls into strafe range).
    // A player travelling toward the ball (the AP) has travel ~= ball direction,
    // so the two modes coincide and its behavior is unchanged. The walk runner
    // still clamps |vy| in every mode, so strafing stays within its safe cap.
    const bool strafe = suppress_heading_slowdown ||
                        (orient_to_ball && dist <= kStrafeMaxDistM);
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
        const bool completed_pass = context.state.active_kick_command.has_value() &&
            context.state.active_kick_command->mode == KickMode::TargetedPass;
        context.state.kick_active_until_s = 0.0;
        context.state.dribble_ready = false;
        context.state.kick_setup_stable_since_s = 0.0;
        context.state.active_kick_command.reset();
        if (completed_pass) {
            context.state.committed_pass.reset();
            context.state.pass_commit_until_s = 0.0;
        }
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
        cooperative_action == nullptr && restart_plan == nullptr &&
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
    const bool use_procedural_strong_kick =
        use_procedural_shot || use_procedural_clear;
    const bool use_procedural_kick =
        use_procedural_dribble || use_procedural_strong_kick;

    double contact_behind_m = kKickContactBehindM;
    double command_behind_m = kDribbleCommandBehindM;
    double release_ball_local_y_m = kForwardContactBallLocalYM;
    double command_ball_local_y_m = kForwardContactBallLocalYM;
    double longitudinal_tolerance_m = kKickSetupLongitudinalToleranceM;
    double lateral_tolerance_m = kKickSetupLateralToleranceM;
    double procedural_max_orientation_error_deg =
        decision::kick_contract::kProceduralDribbleMaximumTargetAngleDeg;
    if (use_procedural_clear) {
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
    const bool setup_ready =
        std::abs(behind_distance - contact_behind_m) <=
            longitudinal_tolerance_m &&
        std::abs(release_lateral_error) <= lateral_tolerance_m &&
        (!use_procedural_kick ||
         orientation_error_deg <=
             procedural_max_orientation_error_deg);

    if (!context.state.dribble_ready && setup_ready) {
        context.state.dribble_ready = true;
    }

    const bool push_position_valid =
        along_direction <= kDribbleMaxAheadM &&
        lateral_offset <= kDribbleMaxLateralOffsetM &&
        context.ball_distance <= field_geometry::kPushBallEngageDistanceM;
    if (context.state.dribble_ready &&
        (!push_position_valid || !setup_ready)) {
        context.state.dribble_ready = false;
        context.state.kick_setup_stable_since_s = 0.0;
    }

    if (!context.state.dribble_ready) {
        // The generic walk-to-point controller brakes on a circular stop
        // radius. On a diagonal approach it can therefore stop too close to
        // the ball while still outside the kick table's lateral envelope. In
        // the near field, close the longitudinal and lateral errors
        // independently in the requested kick frame. The world model keeps a
        // bounded ball track through the expected torso occlusion, so this
        // controller can finish lining up after direct vision disappears.
        if (!needs_side_step && context.snapshot.ball.position_valid &&
            context.ball_distance <= kDribblePrecisionEntryDistanceM) {
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
        WalkCommand approach_command = make_walk_command_avoiding(
            approach_target, context.snapshot, std::nullopt, true, true,
            motion_role_id, true, false);
        approach_command.orientation_deg = absolute_direction_deg;
        approach_command.orientation_absolute = true;
        return approach_command;
    }

    const double planar_speed_mps = math::norm2({
        context.snapshot.self.lin_vel_b[0],
        context.snapshot.self.lin_vel_b[1],
    });
    const bool contact_state_stable =
        context.snapshot.ball.position_valid && setup_ready &&
        orientation_error_deg <= (use_procedural_kick
            ? procedural_max_orientation_error_deg
            : kKickMaxOrientationErrorDeg) &&
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
            (use_procedural_kick
                ? kProceduralKickSetupStableHoldS
                : kKickSetupStableHoldS);
    const bool legal_kick =
        (context.snapshot.play_mode == world::PlayMode::PlayOn ||
         context.snapshot.play_mode_group == world::PlayModeGroup::OurKick) &&
        context.snapshot.ball.position_valid &&
        now >= context.state.next_kick_allowed_s &&
        context.ball_distance >= kKickMinBallDistanceM &&
        context.ball_distance <= kKickMaxBallDistanceM &&
        orientation_error_deg <= (use_procedural_kick
            ? procedural_max_orientation_error_deg
            : kKickMaxOrientationErrorDeg) &&
        contact_state_confirmed;
    // Readiness is a hard release gate. Once dribble_ready is latched the
    // robot can drift slightly inside the valid contact envelope; gating only
    // on setup_ready here would otherwise permit a later unacknowledged kick.
    if (!allow_kick) {
        // A passer that is waiting for Ready must not keep driving through the
        // ball. Hold a small, explicit behind-ball standoff so gait residuals
        // cannot create an unannounced dribble and invalidate the proposal.
        const std::array<double, 2> hold_target{
            context.ball[0] - direction[0] * (kKickContactBehindM + 0.03),
            context.ball[1] - direction[1] * (kKickContactBehindM + 0.03),
        };
        WalkCommand hold_command = make_walk_command_avoiding(
            hold_target, context.snapshot, std::nullopt, true, true,
            motion_role_id, true, false);
        hold_command.orientation_deg = absolute_direction_deg;
        hold_command.orientation_absolute = true;
        hold_command.role_id = motion_role_id;
        return hold_command;
    }
    if (legal_kick) {
        context.state.kick_active_until_s = now + kKickDurationS;
        context.state.next_kick_allowed_s =
            context.state.kick_active_until_s + kKickCooldownS;
        KickCommand kick_command;
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
            kick_command.target_point_m = std::array<double, 2>{
                context.ball[0] + direction[0] * 0.55,
                context.ball[1] + direction[1] * 0.55,
            };
            kick_command.requested_ball_speed_mps =
                decision::kick_contract::kProceduralDribbleRequestedSpeedMps;
            kick_command.mode = KickMode::DribbleTouch;
        }
        if (restart_plan != nullptr) {
            kick_command.restart_epoch = restart_plan->epoch;
            kick_command.restart_revision = restart_plan->revision;
        }
        context.state.active_kick_command = kick_command;
        return kick_command;
    }

    if (context.state.dribble_ready) {
        // Alignment phase: once inside the trained contact envelope, stop
        // translating and turn in place. Continuing toward push_target here
        // drove through the ball before the 2-degree release gate could close.
        // The procedural anchor was calibrated from a high-gain neutral
        // stance. Holding it here also removes zero-command gait sway, making
        // the 0.25 s release debounce physically achievable without relaxing
        // the validated ball-slot or body-speed contracts.
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

bool matching_ready_intent(
    const world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& pass) {
    for (const auto& intent : snapshot.team_comm_snapshot.pass_intents) {
        if (intent.state == comm::PassIntentState::Ready &&
            intent.sender_player_number == pass.target_player_number &&
            intent.passer_player_number == snapshot.player_number &&
            intent.receiver_player_number == pass.target_player_number &&
            intent.sequence_id == pass.sequence_id) {
            return true;
        }
    }
    return false;
}

std::optional<strategy::CooperativeAction> make_goal_shot_candidate(
    const world::WorldSnapshot& snapshot) {
    if (snapshot.play_mode != world::PlayMode::PlayOn ||
        !snapshot.ball.position_valid) {
        return std::nullopt;
    }
    strategy::CooperativeAction shot;
    const auto server_tick = static_cast<std::uint32_t>(std::max(
        0.0, std::round(snapshot.server_time * 50.0)));
    shot.action_id =
        (static_cast<std::uint32_t>(snapshot.player_number) << 24U) |
        (server_tick & 0x00ffffffU);
    shot.category = strategy::ActionCategory::Shoot;
    shot.actor_player_number = snapshot.player_number;
    shot.start_ball_point_m = {
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    shot.target_point_m = field_geometry::actual_their_goal_center_target();
    shot.requested_ball_speed_mps =
        decision::kick_contract::kProceduralShotRequestedSpeedMps;
    shot.confidence = 1.0;
    const double distance_m = math::planar_dist(
        shot.start_ball_point_m, shot.target_point_m);
    if (distance_m <
            decision::kick_contract::kProceduralShotMinimumTargetDistanceM ||
        distance_m >
            decision::kick_contract::kProceduralShotMaximumTargetDistanceM) {
        return std::nullopt;
    }
    return shot;
}

std::optional<strategy::CooperativeAction> make_safety_clear_candidate(
    const world::WorldSnapshot& snapshot) {
    constexpr double kDefensiveClearDepthM = 10.0;
    constexpr double kClearTargetDistanceM = 6.0;
    if (snapshot.play_mode != world::PlayMode::PlayOn ||
        !snapshot.ball.position_valid ||
        snapshot.ball.position_m[0] >
            -field_geometry::kActualHalfLengthM + kDefensiveClearDepthM) {
        return std::nullopt;
    }
    strategy::CooperativeAction clear;
    const auto server_tick = static_cast<std::uint32_t>(std::max(
        0.0, std::round(snapshot.server_time * 50.0)));
    clear.action_id =
        (static_cast<std::uint32_t>(snapshot.player_number) << 24U) |
        (server_tick & 0x00ffffffU);
    clear.category = strategy::ActionCategory::Clear;
    clear.actor_player_number = snapshot.player_number;
    clear.start_ball_point_m = {
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const auto clear_direction = math::vec2_unit_or(
        math::vec2_sub(
            field_geometry::actual_their_goal_center_target(),
            clear.start_ball_point_m),
        {1.0, 0.0});
    clear.target_point_m = math::vec2_add(
        clear.start_ball_point_m,
        math::vec2_scale(clear_direction, kClearTargetDistanceM));
    clear.requested_ball_speed_mps =
        decision::kick_contract::kProceduralClearRequestedSpeedMps;
    clear.confidence = 1.0;
    return clear;
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
    const auto target_delta = math::vec2_sub(pass.target_point_m, ball);
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double relative_target_angle_deg = math::normalize_deg(
        math::vector_angle_deg(target_delta) - self_yaw_deg);
    return std::abs(pass.target_point_m[0]) < field_geometry::kActualHalfLengthM - 0.5 &&
           std::abs(pass.target_point_m[1]) < field_geometry::kActualHalfWidthM - 0.5 &&
           math::planar_dist(ball, pass.start_ball_point_m) <= 0.75 &&
           capabilities.executable(current_request, relative_target_angle_deg);
}

bool receiver_is_ready(
    const world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& pass) {
    return matching_ready_intent(snapshot, pass);
}

const comm::PassIntentRecord* incoming_pass_intent(
    const world::WorldSnapshot& snapshot) {
    const comm::PassIntentRecord* best = nullptr;
    for (const auto& intent : snapshot.team_comm_snapshot.pass_intents) {
        if (intent.state != comm::PassIntentState::Proposed ||
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
    if (feedback.request_kind != MotionRequestKind::Kick ||
        !is_failure(feedback.status) ||
        !feedback.cooperative_action_id.has_value() ||
        !feedback.sequence_id.has_value() ||
        !state_.committed_pass.has_value() ||
        !state_.active_kick_command.has_value()) {
        return;
    }

    const auto& committed = *state_.committed_pass;
    const auto& active_kick = *state_.active_kick_command;
    const std::uint32_t action_id = *feedback.cooperative_action_id;
    const std::uint8_t sequence_id = *feedback.sequence_id;
    if (active_kick.mode != KickMode::TargetedPass ||
        committed.action_id != action_id ||
        committed.sequence_id != sequence_id ||
        active_kick.action_id != action_id ||
        active_kick.sequence_id != sequence_id) {
        return;
    }

    state_.committed_pass.reset();
    state_.active_kick_command.reset();
    state_.pass_commit_until_s = 0.0;
    state_.kick_active_until_s = 0.0;
    state_.dribble_ready = false;
    state_.kick_setup_stable_since_s = 0.0;
    state_.pass_retry_after_s = std::max(
        state_.pass_retry_after_s,
        feedback.server_time + kRejectedPassRetryDelayS);
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
    if (enable_targeted_kick && !state_.committed_pass.has_value() &&
        context.ball_distance <= 2.5) {
        const auto clear = make_safety_clear_candidate(snapshot);
        if (clear.has_value()) {
            const auto target_direction = math::vec2_sub(
                clear->target_point_m, context.ball);
            const double direction_deg = math::vector_angle_deg(target_direction);
            const double self_yaw_deg =
                world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                    snapshot.self.orientation_wxyz);
            const double relative_target_angle_deg = math::normalize_deg(
                direction_deg - self_yaw_deg);
            if (capabilities.executable(*clear, relative_target_angle_deg)) {
                blackboard.set(
                    Blackboard::kKeySelectedCooperativeAction, *clear);
            }
            return make_dribble_command(context, direction_deg, &*clear);
        }
        const auto shot = make_goal_shot_candidate(snapshot);
        if (shot.has_value()) {
            const auto target_direction = math::vec2_sub(
                shot->target_point_m, context.ball);
            const double direction_deg = math::vector_angle_deg(target_direction);
            const double self_yaw_deg =
                world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                    snapshot.self.orientation_wxyz);
            const double relative_target_angle_deg = math::normalize_deg(
                direction_deg - self_yaw_deg);
            if (capabilities.executable(*shot, relative_target_angle_deg)) {
                blackboard.set(
                    Blackboard::kKeySelectedCooperativeAction, *shot);
            }
            // The candidate's distance and speed already lie inside the
            // declared shot envelope. Approach and align even before the
            // angle check passes; make_dribble_command repeats the exact
            // one-degree release gate before it can emit KickMode::Shot.
            return make_dribble_command(context, direction_deg, &*shot);
        }
    }
    if (enable_pass_strategy && enable_targeted_kick &&
        snapshot.play_mode == world::PlayMode::PlayOn) {
        strategy::PlanningResult plan = action_planner_.plan(snapshot);
        blackboard.set(Blackboard::kKeyStrategyPlan, plan);

        if (state_.committed_pass.has_value() &&
            (snapshot.server_time >= state_.pass_commit_until_s ||
             !pass_commit_is_valid(
                 snapshot, *state_.committed_pass, capabilities))) {
            state_.committed_pass.reset();
            state_.pass_commit_until_s = 0.0;
        }

        constexpr double kPassPlanningEngageDistanceM = 2.5;
        constexpr double kPassCommitDurationS = 6.0;
        double relative_target_angle_deg = 0.0;
        if (plan.selected.has_value()) {
            const auto target_delta = math::vec2_sub(
                plan.selected->target_point_m, context.ball);
            const double target_heading_deg = math::vector_angle_deg(target_delta);
            const double self_yaw_deg =
                world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                    snapshot.self.orientation_wxyz);
            relative_target_angle_deg =
                math::normalize_deg(target_heading_deg - self_yaw_deg);
        }
        if (!state_.committed_pass.has_value() && plan.selected.has_value() &&
            snapshot.server_time >= state_.pass_retry_after_s &&
            capabilities.executable(*plan.selected, relative_target_angle_deg) &&
            context.ball_distance <= kPassPlanningEngageDistanceM) {
            strategy::CooperativeAction committed = *plan.selected;
            state_.next_pass_sequence_id = static_cast<std::uint8_t>(
                state_.next_pass_sequence_id + 1U);
            if (state_.next_pass_sequence_id == 0U) {
                state_.next_pass_sequence_id = 1U;
            }
            committed.sequence_id = state_.next_pass_sequence_id;
            state_.committed_pass = committed;
            state_.pass_commit_until_s = snapshot.server_time + kPassCommitDurationS;
        }

        if (state_.committed_pass.has_value()) {
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *state_.committed_pass);
            const auto target_direction = math::vec2_sub(
                state_.committed_pass->target_point_m, context.ball);
            if (math::norm2(target_direction) > 1.0e-6) {
                const double direction_deg = math::vector_angle_deg(target_direction);
                return make_dribble_command(
                    context,
                    direction_deg,
                    &*state_.committed_pass,
                    receiver_is_ready(snapshot, *state_.committed_pass));
            }
            state_.committed_pass.reset();
            state_.pass_commit_until_s = 0.0;
        }
    } else {
        state_.committed_pass.reset();
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

HighLevelCommand SimpleRoleBehavior::make_command(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard) const {
    if (snapshot.play_mode == world::PlayMode::PlayOn) {
        if (const auto* intent = incoming_pass_intent(snapshot); intent != nullptr) {
            blackboard.set(
                Blackboard::kKeyTacticalTarget,
                TacticalTarget{
                    TacticalDuty::Receive,
                    {intent->target_x_m, intent->target_y_m},
                    std::array<double, 2>{
                        snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
                    0,
                    0.9});
            return make_walk_command_avoiding(
                {intent->target_x_m, intent->target_y_m}, snapshot, std::nullopt,
                true, true, role_id_, true);
        }
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
        math::planar_dist(self, tactical_target.position_m) <= kStrafeMaxDistM) {
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
                    std::nullopt, true, true, RoleManager::ROLE_GK, true);
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
    clearance_state_ = {};

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
    WalkCommand command = make_walk_command_avoiding(
        tactical_target.position_m, snapshot, std::nullopt,
        true, true, RoleManager::ROLE_GK, true);
    if (tactical_target.face_point_m.has_value() &&
        math::planar_dist(context.self, tactical_target.position_m) <=
            kStrafeMaxDistM) {
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
    const std::array<const RoleBehavior*, 6> behaviors{
        &cbm_,
        &st_,
        &cbl_,
        &cbr_,
        &cdm_,
        &gk_,
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
}

}  // namespace decision
