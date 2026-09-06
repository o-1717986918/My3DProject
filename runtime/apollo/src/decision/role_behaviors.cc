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
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace decision {

namespace {

constexpr double kWalkMaxSpeedM = 3.0;
constexpr double kWalkBrakeDecelMps2 = 1.0;
constexpr double kWalkStopRadiusM = 0.15;
constexpr double kWalkHeadingSlowStartDeg = 18.0;
// The currently deployed walk is substantially more reliable in its forward
// domain than while translating sideways. Above this heading error, compose a
// pure turn with a later forward walk instead of asking the policy to strafe or
// backpedal. This is a runtime fallback, not a learned omnidirectional claim.
constexpr double kWalkHeadingStopDeg = 32.0;
constexpr double kEmergencyChallengePushPastBallM = 0.70;
// An opponent inside this corridor makes centimetre-scale setup or a detour
// around the ball strictly lower priority than immediate physical pressure.
constexpr double kUrgentOpponentBallDistanceM = 1.40;
constexpr double kUrgentSelfBallDistanceM = 1.10;
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
constexpr double kDribbleCommandBehindM = 0.34;
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
// A restart has no opponent-pressure reason to sprint through the final setup
// corridor.  The regular precision gain can request 0.85 m/s while only 25 cm
// behind the slot; measured residual momentum then moves the body roughly
// 20 cm after braking and starts play through an accidental walk contact.
constexpr double kRestartMaxPrecisionForwardSpeedMps = 0.25;
constexpr double kRestartMaxPrecisionReverseSpeedMps = 0.20;
// Do not ask the deployed forward-dominant walk policy to strafe across a
// large precision error. First face and walk toward the canonical setup point;
// only the final bounded correction retains fixed kick orientation. This is
// the same turn-then-forward composition used by ordinary navigation.
constexpr double kKickCoarseRelocateLateralErrorM = 0.08;
constexpr double kKickCoarseTravelTurnThresholdDeg = 20.0;
constexpr double kKickFinalTurnThresholdDeg = 12.0;
// Procedural anchors describe the ball in the robot body frame. Their wider
// target-angle fields are admission envelopes, not permission to release a
// body-fixed trajectory while still several degrees away from the requested
// direction. At 5.6 deg, a ball that is perfectly centred in the target frame
// shifts by about 3.1 cm in the body frame and is rejected by the runner. Keep
// strategy admission broad, but finish static-trajectory alignment to 1 deg.
constexpr double kProceduralStaticReleaseYawToleranceDeg = 1.0;
constexpr double kDribbleSideDistanceM = 0.8;
constexpr double kDribbleSideClearanceM = 0.55;
constexpr double kDribbleSideStepBehindThresholdM = 0.1;
constexpr double kDribbleMaxLateralOffsetM = 0.08;
constexpr double kDribbleMaxAheadM = 0.15;
// Pressure possession must not inherit the precision action's centimetre-scale
// setup. These values restore Apollo's proven continuous walk-through-ball
// envelope as a separate, observable control path. A calm, explicitly selected
// Dribble/Pass/Shoot/Clear action still uses the stricter contract below.
constexpr double kPressurePushSetupDistanceM = 0.60;
constexpr double kPressurePushSetupToleranceM = 0.25;
constexpr double kPressurePushMaximumLateralOffsetM = 0.40;
constexpr int kPressurePushSetupModeKey = 100;
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
// A moving dribble touch gets one decision cycle: natural-match telemetry
// showed that a two-cycle wait lets residual walk momentum carry an otherwise
// valid contact pose back out of the release slot.  Shot/clear remain static
// strong contacts and keep their calibrated two-cycle hold.  Range-pass keeps
// immediate release until its expanded pose envelope is independently
// evaluated.
constexpr double kProceduralDribbleSetupStableHoldS = 0.02;
constexpr double kProceduralKickSetupStableHoldS = 0.04;
// The fixed-2 m residual/learned executor was trained from captured gait
// phases.  It must see that phase instead of first being filtered through the
// static procedural trajectory's leg-velocity gate and long neutral hold.
// Keep one decision-cycle pose confirmation so a single noisy ball sample
// cannot release the action; the runner still checks its own ball/yaw domain.
constexpr double kPhaseConditionedKickSetupStableHoldS = 0.02;
// The deployed walk commonly retains 0.22--0.35 m/s of measured torso motion
// after entering its neutral command. Requiring less than 0.20 m/s starved
// every contact in a complete comparison match. The procedural runner repeats
// this same bound before it captures the current pose.
constexpr double kKickMinBallDistanceM = 0.30;
constexpr double kKickMaxBallDistanceM = 0.41;
// Server zero-command sway is about two centimetres peak-to-peak. A 3 cm
// release band remains well inside the residual runner's 9 cm contact
// envelope while allowing the stable-hold timer to survive one gait cycle.
// The zero-command policy oscillates around roughly two degrees on the server.
// Three degrees remains comfortably below the ten-degree action promotion gate
// while admitting a continuous debounce window for the stable fallback.
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
constexpr double kKickPreSettlePaddingM = 0.055;
constexpr double kKickPreSettleMaximumDistanceM = 0.45;
constexpr double kKickPreSettleEntrySpeedMps = 0.25;
constexpr double kKickPreSettleExitSpeedMps = 0.20;
constexpr double kKickPreSettleStableHoldS = 0.10;
// Preserve the original Apollo walk-through-ball behavior as an explicit,
// observable last resort. It is available only after a sustained near-ball
// setup attempt and inside this wider contact corridor.
// The original Apollo contact path keeps a short recovery window under
// pressure. A selected precision action receives enough time to finish its
// bounded longitudinal/lateral correction; otherwise the fallback would win
// before the walking actor could cover the observed 10--22 cm lateral error.
// A targeted pass uses the same longer window because a fixed forward contact
// is not semantically equivalent to the agreed pass.
constexpr double kForwardContactFastFallbackDelayS = 0.45;
constexpr double kPrecisionActionFallbackDelayS = 1.20;
constexpr double kPrecisionActionProgressGraceS = 0.50;
constexpr double kPrecisionActionHardFallbackDelayS = 1.80;
// A strong kick often has to compose turn -> walk -> final turn in the lateral
// edge of the penalty area. Preserve a genuinely improving attempt for one
// additional gait phase; the ordinary stalled-progress fallback above remains
// unchanged and still prevents waiting on a dead setup.
constexpr double kStrongKickHardFallbackDelayS = 2.60;
constexpr double kKickSetupMeaningfulProgressM = 0.01;
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
// A straight goal-line aim from a wide late attack repeatedly carried the ball
// out beside the post.  Inside this zone, first spend forward progress on a
// steep inward carry.  Once the ball reaches the goal corridor, the ordinary
// low-turn goal-mouth selector takes over.
constexpr double kFinalThirdCutInDepthM = 8.0;
constexpr double kFinalThirdCutInLateralMarginM = 1.0;
constexpr double kFinalThirdCutInAdvanceM = 1.5;
constexpr double kFinalThirdCutInTargetYM = 1.0;

// A static strong shot is useful only when the current approach can finish
// before the ball or an opponent removes the opportunity.  This is an
// admission estimate, deliberately separate from the centimetre-scale release
// contract below.  The learned fixed-distance pass consumes live gait phase
// and never goes through this static-action estimate.
constexpr double kStaticShotMaximumObservedBallSpeedMps = 0.60;
constexpr double kStaticShotMaximumInitialLateralErrorM = 0.35;
constexpr double kStaticShotMaximumInitialYawErrorDeg = 60.0;
constexpr double kStaticShotEstimatedSetupSpeedMps = 0.75;
constexpr double kStaticShotEstimatedTurnRateDegS = 120.0;
constexpr double kStaticShotFixedAcquisitionTimeS = 0.18;
constexpr double kStaticShotOpponentReserveS = 0.10;
// The deterministic short-touch trajectory assumes an almost stationary ball.
// Natural v28 play admitted several touches just as the ball accelerated to
// roughly 1.9--2.4 m/s; the actor then spent its whole commitment chasing a
// release pose that no longer existed.  Reject only that static action and
// immediately retain the continuous pressure path.
constexpr double kStaticDribbleMaximumObservedBallSpeedMps = 0.45;
constexpr double kStaticDribbleMaximumBallDisplacementM = 0.12;

bool needs_final_third_cut_in(const std::array<double, 2>& ball) {
    return field_geometry::kActualHalfLengthM - ball[0] <=
               kFinalThirdCutInDepthM &&
        std::abs(ball[1]) >
            field_geometry::kGoalHalfWidthM +
                kFinalThirdCutInLateralMarginM;
}

struct ProceduralTransitionState {
    bool ready{false};
    double maximum_tilt_rate_deg_s{0.0};
    double maximum_leg_velocity_deg_s{0.0};
};

ProceduralTransitionState procedural_transition_state(
    const world::WorldSnapshot& snapshot,
    double planar_speed_mps) {
    static constexpr std::array<std::string_view, 12> kLegJointNames{{
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    }};
    ProceduralTransitionState result;
    result.maximum_tilt_rate_deg_s = std::max(
        std::abs(snapshot.self.gyro_deg_s[0]),
        std::abs(snapshot.self.gyro_deg_s[1]));
    bool complete_joint_state = true;
    bool leg_positions_safe = true;
    for (const std::string_view name : kLegJointNames) {
        const auto position = snapshot.self.joint_positions_deg.find(
            std::string(name));
        const auto velocity = snapshot.self.joint_velocities_deg_s.find(
            std::string(name));
        if (position == snapshot.self.joint_positions_deg.end() ||
            velocity == snapshot.self.joint_velocities_deg_s.end() ||
            !std::isfinite(position->second) ||
            !std::isfinite(velocity->second)) {
            complete_joint_state = false;
            continue;
        }
        leg_positions_safe = leg_positions_safe &&
            std::abs(position->second) <=
                decision::kick_contract::kProceduralMaximumStartLegPositionDeg;
        result.maximum_leg_velocity_deg_s = std::max(
            result.maximum_leg_velocity_deg_s,
            std::abs(velocity->second));
    }
    result.ready =
        complete_joint_state && leg_positions_safe &&
        planar_speed_mps <=
            decision::kick_contract::kProceduralMaximumStartPlanarSpeedMps &&
        result.maximum_tilt_rate_deg_s <=
            decision::kick_contract::kProceduralMaximumStartTiltRateDegS &&
        result.maximum_leg_velocity_deg_s <=
            decision::kick_contract::kProceduralMaximumStartLegVelocityDegS;
    return result;
}

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
        state.kick_setup_started_s = 0.0;
        state.kick_setup_best_pose_error_m = -1.0;
        state.kick_setup_last_progress_s = 0.0;
        state.kick_setup_mode_key = -1;
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
    const std::array<double, 2> legal_target_position_m =
        role_id.has_value() &&
            (*role_id == RoleManager::ROLE_GK ||
             *role_id == RoleManager::ROLE_CBL)
        ? target_position_m
        : field_geometry::keep_field_player_outside_our_goalie_area(
              target_position_m);
    const std::array<double, 2> self{snapshot.self.position_m[0], snapshot.self.position_m[1]};
    const double dist = math::planar_dist(self, legal_target_position_m);

    if (dist < field_geometry::kNearTargetM) {
        WalkCommand command = make_walk_command(legal_target_position_m);
        command.orientation_deg = orient_to_ball
            ? orientation_to_ball_from_self(snapshot)
            : std::nullopt;
        command.orientation_absolute = true;
        command.role_id = role_id;
        return command;
    }

    const auto plan = plan_walk(
        self,
        legal_target_position_m,
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
    bool learned_kick_enabled{false};
    std::array<double, 2> ball{0.0, 0.0};
    std::array<double, 2> self{0.0, 0.0};
    double ball_distance{0.0};
    bool urgent_contest{false};
};

double nearest_fresh_opponent_ball_distance_m(
    const world::WorldSnapshot& snapshot) {
    if (!snapshot.ball.position_valid) {
        return std::numeric_limits<double>::infinity();
    }
    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    double nearest = std::numeric_limits<double>::infinity();
    const auto consider = [&](const world::PlayerObservation& opponent) {
        const bool fresh = opponent.seen ||
            (opponent.last_seen_time >= 0.0 &&
             snapshot.server_time - opponent.last_seen_time <= 0.75);
        if (!fresh || opponent.fallen) return;
        nearest = std::min(
            nearest,
            math::planar_dist(
                ball,
                {opponent.position_m[0], opponent.position_m[1]}));
    };
    for (const auto& opponent : snapshot.opponents) consider(opponent);
    for (const auto& opponent : snapshot.shared_opponents) consider(opponent);
    return nearest;
}

bool urgent_ball_contest(
    const world::WorldSnapshot& snapshot,
    double self_ball_distance_m) {
    return snapshot.play_mode == world::PlayMode::PlayOn &&
        snapshot.ball.position_valid &&
        (snapshot.ball.visible || snapshot.ball.position_age_s <= 0.75) &&
        self_ball_distance_m <= kUrgentSelfBallDistanceM &&
        nearest_fresh_opponent_ball_distance_m(snapshot) <=
            kUrgentOpponentBallDistanceM;
}

bool static_shot_setup_feasible(
    const world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& action,
    const strategy::TacticalState& tactical_state) {
    if (action.category != strategy::ActionCategory::Shoot ||
        !snapshot.ball.position_valid) {
        return false;
    }

    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const std::array<double, 2> self{
        snapshot.self.position_m[0], snapshot.self.position_m[1]};
    const auto target_vector = math::vec2_sub(action.target_point_m, ball);
    const double target_distance_m = math::norm2(target_vector);
    if (!std::isfinite(target_distance_m) ||
        target_distance_m <
            decision::kick_contract::kProceduralShotMinimumTargetDistanceM ||
        target_distance_m >
            decision::kick_contract::kProceduralShotMaximumTargetDistanceM) {
        return false;
    }

    const auto direction = math::vec2_scale(
        target_vector, 1.0 / target_distance_m);
    const std::array<double, 2> perpendicular{-direction[1], direction[0]};
    const auto self_from_ball = math::vec2_sub(self, ball);
    const double behind_distance_m = -(
        self_from_ball[0] * direction[0] +
        self_from_ball[1] * direction[1]);
    const double lateral_error_m =
        self_from_ball[0] * perpendicular[0] +
        self_from_ball[1] * perpendicular[1] +
        kProceduralShotCommandBallLocalYM;
    const double longitudinal_error_m =
        behind_distance_m - kProceduralStrongKickCommandBehindM;
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double target_yaw_deg = math::vector_angle_deg(direction);
    const double yaw_error_deg = std::abs(
        math::normalize_deg(target_yaw_deg - self_yaw_deg));

    if (std::abs(lateral_error_m) >
            kStaticShotMaximumInitialLateralErrorM ||
        yaw_error_deg > kStaticShotMaximumInitialYawErrorDeg) {
        return false;
    }

    if (snapshot.ball.velocity_valid) {
        const double ball_speed_mps = math::norm2({
            snapshot.ball.velocity_mps[0], snapshot.ball.velocity_mps[1]});
        if (!std::isfinite(ball_speed_mps) ||
            ball_speed_mps > kStaticShotMaximumObservedBallSpeedMps) {
            return false;
        }
    }

    const double translation_error_m = std::hypot(
        longitudinal_error_m, lateral_error_m);
    const double estimated_setup_time_s =
        translation_error_m / kStaticShotEstimatedSetupSpeedMps +
        yaw_error_deg / kStaticShotEstimatedTurnRateDegS +
        kStaticShotFixedAcquisitionTimeS;
    return !std::isfinite(tactical_state.nearest_opponent_ball_time_s) ||
        estimated_setup_time_s + kStaticShotOpponentReserveS <=
            tactical_state.nearest_opponent_ball_time_s;
}

bool static_dribble_setup_feasible(
    const world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& action) {
    if (action.category != strategy::ActionCategory::Dribble ||
        !snapshot.ball.position_valid) {
        return false;
    }
    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    if (math::planar_dist(ball, action.start_ball_point_m) >
        kStaticDribbleMaximumBallDisplacementM) {
        return false;
    }
    if (!snapshot.ball.velocity_valid) return true;
    const double ball_speed_mps = math::norm2({
        snapshot.ball.velocity_mps[0], snapshot.ball.velocity_mps[1]});
    return std::isfinite(ball_speed_mps) &&
        ball_speed_mps <= kStaticDribbleMaximumObservedBallSpeedMps;
}

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
        context.state.kick_setup_best_pose_error_m = -1.0;
        context.state.kick_setup_last_progress_s = 0.0;
        context.state.kick_setup_mode_key = -1;
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

    // No explicit cooperative action means this is ordinary possession or a
    // pressure recovery, not permission to spend 1.2--1.8 seconds seeking a
    // static procedural release slot. Restore the original Apollo two-phase
    // continuous push: approach a broad point behind the ball, then keep
    // walking one metre through it. This path deliberately emits WalkCommand;
    // exact procedural contact remains available only for an explicitly
    // selected Dribble/Pass/Shoot/Clear action.
    const bool use_pressure_push =
        cooperative_action == nullptr && restart_plan == nullptr &&
        context.snapshot.play_mode == world::PlayMode::PlayOn;
    if (use_pressure_push) {
        if (context.state.kick_setup_mode_key != kPressurePushSetupModeKey) {
            context.state.pressure_push_latched = false;
            context.state.dribble_ready = false;
            context.state.kick_pre_settling = false;
            context.state.kick_pre_settle_stable_since_s = 0.0;
            context.state.kick_setup_stable_since_s = 0.0;
            context.state.kick_setup_started_s = 0.0;
            context.state.kick_setup_last_update_s = 0.0;
            context.state.kick_setup_best_pose_error_m = -1.0;
            context.state.kick_setup_last_progress_s = 0.0;
            context.state.last_kick_setup_gate = -1;
        }
        context.state.kick_setup_mode_key = kPressurePushSetupModeKey;

        const std::array<double, 2> perpendicular{-direction[1], direction[0]};
        const std::array<double, 2> setup_target{
            context.ball[0] - direction[0] * kPressurePushSetupDistanceM,
            context.ball[1] - direction[1] * kPressurePushSetupDistanceM,
        };
        const std::array<double, 2> push_target{
            context.ball[0] + direction[0] * field_geometry::kPushPastBallM,
            context.ball[1] + direction[1] * field_geometry::kPushPastBallM,
        };
        const std::array<double, 2> self_from_ball{
            context.self[0] - context.ball[0],
            context.self[1] - context.ball[1],
        };
        const double along_direction =
            self_from_ball[0] * direction[0] +
            self_from_ball[1] * direction[1];
        const double signed_lateral_offset =
            self_from_ball[0] * perpendicular[0] +
            self_from_ball[1] * perpendicular[1];
        const double lateral_offset = std::abs(signed_lateral_offset);
        if (context.urgent_contest) {
            // Under a live challenge, going 0.8 m around the ball to recover a
            // textbook behind-ball pose loses the race we are trying to win.
            // Walk through the ball on the shortest line; use the desired goal
            // direction only when the body is already broadly behind it.
            const auto self_to_ball = math::vec2_sub(
                context.ball, context.self);
            const auto shortest_contact_direction = math::vec2_unit_or(
                self_to_ball, direction);
            const double goal_alignment =
                shortest_contact_direction[0] * direction[0] +
                shortest_contact_direction[1] * direction[1];
            const auto contest_direction = goal_alignment >= 0.35
                ? direction
                : shortest_contact_direction;
            const std::array<double, 2> contest_target{
                context.ball[0] + contest_direction[0] *
                    kEmergencyChallengePushPastBallM,
                context.ball[1] + contest_direction[1] *
                    kEmergencyChallengePushPastBallM,
            };
            if (context.state.last_kick_setup_gate != 104) {
                context.state.last_kick_setup_gate = 104;
                std::cerr
                    << "MY3D_KICK_SETUP player="
                    << context.snapshot.player_number
                    << " time=" << context.snapshot.server_time
                    << " mode=forward phase=urgent-contest"
                    << " action_id=0 setup_elapsed=0"
                    << " ball_distance=" << context.ball_distance
                    << " behind_error="
                    << (-along_direction - kPressurePushSetupDistanceM)
                    << " lateral_error=" << signed_lateral_offset
                    << " yaw_error_deg=" << orientation_error_deg
                    << " speed="
                    << math::norm2({
                           context.snapshot.self.lin_vel_b[0],
                           context.snapshot.self.lin_vel_b[1]})
                    << " tilt_rate_deg_s=0 leg_rate_deg_s=0\n";
            }
            context.state.pressure_push_latched = false;
            return make_walk_command_avoiding(
                contest_target, context.snapshot, std::nullopt,
                true, false, motion_role_id, false, false);
        }
        const bool needs_side_step =
            along_direction > -kDribbleSideStepBehindThresholdM &&
            lateral_offset < kDribbleSideClearanceM;
        std::array<double, 2> approach_target = setup_target;
        if (needs_side_step) {
            const double side_sign = signed_lateral_offset < 0.0 ? -1.0 : 1.0;
            approach_target = {
                context.ball[0] + perpendicular[0] * side_sign *
                    kDribbleSideDistanceM,
                context.ball[1] + perpendicular[1] * side_sign *
                    kDribbleSideDistanceM,
            };
        }
        if (!context.state.pressure_push_latched &&
            math::planar_dist(context.self, setup_target) <=
                kPressurePushSetupToleranceM) {
            context.state.pressure_push_latched = true;
        }
        const bool push_position_valid =
            along_direction <= kDribbleMaxAheadM &&
            lateral_offset <= kPressurePushMaximumLateralOffsetM &&
            context.ball_distance <= field_geometry::kPushBallEngageDistanceM;
        if (context.state.pressure_push_latched && !push_position_valid) {
            context.state.pressure_push_latched = false;
        }
        const auto trace_pressure_phase = [&](int gate, const char* phase) {
            if (context.state.last_kick_setup_gate == gate) return;
            context.state.last_kick_setup_gate = gate;
            std::cerr
                << "MY3D_KICK_SETUP player=" << context.snapshot.player_number
                << " time=" << context.snapshot.server_time
                << " mode=forward phase=" << phase
                << " action_id=0 setup_elapsed=0"
                << " ball_distance=" << context.ball_distance
                << " behind_error="
                << (-along_direction - kPressurePushSetupDistanceM)
                << " lateral_error=" << signed_lateral_offset
                << " yaw_error_deg=" << orientation_error_deg
                << " speed="
                << math::norm2({
                       context.snapshot.self.lin_vel_b[0],
                       context.snapshot.self.lin_vel_b[1]})
                << " tilt_rate_deg_s=0 leg_rate_deg_s=0\n";
        };
        if (!context.state.pressure_push_latched) {
            trace_pressure_phase(
                needs_side_step ? 101 : 102,
                needs_side_step ? "pressure-side-relocate" :
                                  "pressure-approach");
            WalkCommand approach_command = make_walk_command_avoiding(
                approach_target, context.snapshot, std::nullopt, true, true,
                motion_role_id, true, false);
            approach_command.orientation_deg = absolute_direction_deg;
            approach_command.orientation_absolute = true;
            return approach_command;
        }
        trace_pressure_phase(103, "pressure-push");
        return make_walk_command_avoiding(
            push_target, context.snapshot, std::nullopt, true, true,
            motion_role_id, false, false);
    }
    context.state.pressure_push_latched = false;

    // Setup time belongs to one semantic action, not merely to a similar
    // heading. Without this key, a newly selected Dribble/Shoot/Clear action
    // could inherit an almost-expired timer from the legacy forward-contact
    // path and fall back on its very first cycle.
    const int setup_mode_key = restart_plan != nullptr
        ? 1
        : cooperative_action != nullptr
            ? 2 + static_cast<int>(cooperative_action->category)
            : 0;
    const std::uint32_t setup_action_id = cooperative_action != nullptr
        ? cooperative_action->action_id
        : 0U;
    const bool setup_action_changed =
        context.state.kick_setup_mode_key >= 0 &&
        (context.state.kick_setup_mode_key != setup_mode_key ||
         context.state.kick_setup_action_id != setup_action_id);

    const bool setup_discontinuous =
        context.state.kick_setup_last_update_s > 0.0 &&
        now - context.state.kick_setup_last_update_s >
            kKickSetupContinuityTimeoutS;
    const bool setup_direction_changed =
        context.state.kick_setup_started_s > 0.0 &&
        std::abs(math::normalize_deg(
            absolute_direction_deg - context.state.kick_setup_direction_deg)) >
            kKickSetupDirectionResetDeg;
    const bool restart_anchor_actionable = restart_plan != nullptr &&
        restart_plan->ball_anchor_valid &&
        context.snapshot.play_mode_group == world::PlayModeGroup::OurKick;
    const bool ball_position_actionable =
        context.snapshot.ball.position_valid || restart_anchor_actionable;
    if (setup_action_changed || setup_discontinuous || setup_direction_changed ||
        !ball_position_actionable ||
        context.ball_distance > kDribblePrecisionEntryDistanceM) {
        context.state.dribble_ready = false;
        context.state.kick_setup_started_s = 0.0;
        context.state.kick_setup_stable_since_s = 0.0;
        context.state.kick_setup_best_pose_error_m = -1.0;
        context.state.kick_setup_last_progress_s = 0.0;
        context.state.kick_pre_settling = false;
        context.state.kick_pre_settle_stable_since_s = 0.0;
        context.state.last_kick_setup_gate = -1;
    }
    context.state.kick_setup_mode_key = setup_mode_key;
    context.state.kick_setup_action_id = setup_action_id;
    if (ball_position_actionable &&
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
    const bool is_targeted_pass =
        cooperative_action != nullptr &&
        cooperative_action->category == strategy::ActionCategory::Pass;
    const double current_target_distance_m = cooperative_action != nullptr
        ? math::planar_dist(cooperative_action->target_point_m, context.ball)
        : 0.0;
    const bool use_learned_transition =
        context.learned_kick_enabled && is_targeted_pass &&
        decision::kick_contract::learned_transition_pass_request_supported(
            current_target_distance_m,
            cooperative_action->requested_ball_speed_mps);
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
    // Restarts use the model-independent contact macro, but they still require
    // a stationary release.  Without the same pre-settle treatment used by a
    // procedural kick, the walk actor can coast through the ball before the
    // restart coordinator authorizes contact.
    const bool use_static_release_setup =
        use_procedural_kick || restart_plan != nullptr;
    const char* kick_mode = cooperative_action == nullptr
        ? (restart_plan == nullptr ? "forward" : "restart")
        : cooperative_action->category == strategy::ActionCategory::Pass
            ? "pass"
            : cooperative_action->category == strategy::ActionCategory::Shoot
                ? "shot"
                : cooperative_action->category == strategy::ActionCategory::Clear
                    ? "clear"
                    : "dribble";

    double contact_behind_m =
        decision::kick_contract::kForwardContactBallLocalXM;
    double command_behind_m = kDribbleCommandBehindM;
    double release_ball_local_y_m =
        decision::kick_contract::kForwardContactBallLocalYM;
    double command_ball_local_y_m =
        decision::kick_contract::kForwardContactBallLocalYM;
    double longitudinal_tolerance_m =
        decision::kick_contract::kForwardContactBallLocalXToleranceM;
    double lateral_tolerance_m =
        decision::kick_contract::kForwardContactBallLocalYToleranceM;
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
        contact_behind_m =
            decision::kick_contract::kProceduralDribbleBallLocalXM;
        command_behind_m =
            decision::kick_contract::kProceduralDribbleBallLocalXM;
        release_ball_local_y_m =
            decision::kick_contract::kProceduralDribbleBallLocalYM;
        command_ball_local_y_m =
            decision::kick_contract::kProceduralDribbleBallLocalYM;
        longitudinal_tolerance_m =
            decision::kick_contract::kProceduralDribbleBallLocalXToleranceM;
        lateral_tolerance_m =
            decision::kick_contract::kProceduralDribbleBallLocalYToleranceM;
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
    const std::array<double, 2> canonical_setup_target{
        context.ball[0] - direction[0] * command_behind_m -
            perpendicular[0] * command_ball_local_y_m,
        context.ball[1] - direction[1] * command_behind_m -
            perpendicular[1] * command_ball_local_y_m,
    };
    const auto observed_ball_local = math::rotate_2d(
        math::vec2_sub(context.ball, context.self), -self_yaw_deg);
    const bool learned_transition_position_ready =
        observed_ball_local[0] >=
            decision::kick_contract::kLearnedTransitionMinimumBallLocalXM &&
        observed_ball_local[0] <=
            decision::kick_contract::kLearnedTransitionMaximumBallLocalXM &&
        observed_ball_local[1] >=
            decision::kick_contract::kLearnedTransitionMinimumBallLocalYM &&
        observed_ball_local[1] <=
            decision::kick_contract::kLearnedTransitionMaximumBallLocalYM;
    const bool position_ready = use_learned_transition
        ? learned_transition_position_ready
        : std::abs(behind_distance - contact_behind_m) <=
              longitudinal_tolerance_m &&
          std::abs(release_lateral_error) <= lateral_tolerance_m;
    const bool latched_position_ready =
        position_ready ||
        (use_procedural_dribble && context.state.dribble_ready &&
         std::abs(behind_distance - contact_behind_m) <=
             longitudinal_tolerance_m &&
         std::abs(release_lateral_error) <=
             lateral_tolerance_m);
    const double required_orientation_error_deg = use_learned_transition
        ? decision::kick_contract::kLearnedTransitionMaximumTargetAngleDeg
        : use_procedural_kick
            ? procedural_max_orientation_error_deg
            : decision::kick_contract::kForwardContactMaximumTargetAngleDeg;
    const double release_orientation_tolerance_deg = use_learned_transition
        ? required_orientation_error_deg
        : use_procedural_kick
            ? std::min(
                  required_orientation_error_deg,
                  kProceduralStaticReleaseYawToleranceDeg)
            : required_orientation_error_deg;
    const double planar_speed_mps = math::norm2({
        context.snapshot.self.lin_vel_b[0],
        context.snapshot.self.lin_vel_b[1],
    });
    const ProceduralTransitionState transition_state =
        procedural_transition_state(context.snapshot, planar_speed_mps);
    const auto trace_setup_gate = [&](int gate, const char* phase) {
        if (context.state.last_kick_setup_gate == gate) return;
        context.state.last_kick_setup_gate = gate;
        std::cerr
            << "MY3D_KICK_SETUP player=" << context.snapshot.player_number
            << " time=" << context.snapshot.server_time
            << " mode=" << kick_mode
            << " phase=" << phase
            << " action_id=" << setup_action_id
            << " setup_elapsed="
            << (context.state.kick_setup_started_s > 0.0
                    ? now - context.state.kick_setup_started_s
                    : 0.0)
            << " ball_distance=" << context.ball_distance
            << " behind_error=" << behind_distance - contact_behind_m
            << " lateral_error=" << release_lateral_error
            << " yaw_error_deg=" << orientation_error_deg
            << " speed=" << planar_speed_mps
            << " tilt_rate_deg_s="
            << transition_state.maximum_tilt_rate_deg_s
            << " leg_rate_deg_s="
            << transition_state.maximum_leg_velocity_deg_s
            << '\n';
    };
    const bool fallback_contact_pose =
        ball_position_actionable &&
        context.ball_distance >= kForwardContactFallbackMinimumBehindM &&
        context.ball_distance <= kForwardContactFallbackMaximumBehindM &&
        behind_distance >= kForwardContactFallbackMinimumBehindM &&
        behind_distance <= kForwardContactFallbackMaximumBehindM &&
        lateral_offset <= kForwardContactFallbackMaximumLateralM &&
        orientation_error_deg <= kForwardContactFallbackMaximumYawErrorDeg &&
        planar_speed_mps <= kForwardContactFallbackMaximumPlanarSpeedMps;
    const bool use_phase_conditioned_transition =
        is_targeted_pass && !use_procedural_range_pass;
    const bool static_pass_fallback_eligible =
        !is_targeted_pass ||
        (decision::kick_contract::parameterized_pass_request_supported(
             current_target_distance_m,
             cooperative_action->requested_ball_speed_mps) &&
         orientation_error_deg <=
             decision::kick_contract::kParameterizedPassMaximumTargetAngleDeg);
    const bool precision_action = is_targeted_pass || use_procedural_kick;
    const bool strong_kick_action = use_procedural_shot || use_procedural_clear;
    // Treat a centimetre of combined translation/yaw-equivalent reduction as
    // meaningful progress. A moving setup gets a short grace period after the
    // minimum window, but a hard bound prevents indefinite dithering.
    const double setup_pose_error_m =
        std::abs(behind_distance - contact_behind_m) +
        std::abs(release_lateral_error) +
        0.005 * orientation_error_deg;
    if (context.state.kick_setup_best_pose_error_m < 0.0 ||
        setup_pose_error_m + kKickSetupMeaningfulProgressM <
            context.state.kick_setup_best_pose_error_m) {
        context.state.kick_setup_best_pose_error_m = setup_pose_error_m;
        context.state.kick_setup_last_progress_s = now;
    }
    const double setup_elapsed_s = context.state.kick_setup_started_s > 0.0
        ? now - context.state.kick_setup_started_s
        : 0.0;
    const bool precision_stalled =
        context.state.kick_setup_last_progress_s <= 0.0 ||
        now - context.state.kick_setup_last_progress_s >=
            kPrecisionActionProgressGraceS;
    const bool fallback_due =
        context.state.kick_setup_started_s > 0.0 &&
        (precision_action
            ? (setup_elapsed_s >=
                   (strong_kick_action
                       ? kStrongKickHardFallbackDelayS
                       : kPrecisionActionHardFallbackDelayS) ||
               (setup_elapsed_s >= kPrecisionActionFallbackDelayS &&
                precision_stalled))
            : setup_elapsed_s >= kForwardContactFastFallbackDelayS);
    const bool fallback_allowed =
        // Never let a timeout pre-empt an exact procedural release slot. The
        // controlled 0.45 s run reached 14.7 mm longitudinal and 0.5 mm
        // lateral error, but the old ordering emitted fallback before the
        // readiness latch could enter its short neutral debounce.
        allow_kick && fallback_due && fallback_contact_pose &&
        static_pass_fallback_eligible &&
        !(use_procedural_kick && latched_position_ready) &&
        now >= context.state.next_kick_allowed_s;
    const auto make_controlled_brake_command = [&]() {
        // Keep the learned walk actor in control while removing translational
        // demand.  Jumping directly from a 0.5--0.7 m/s gait into the neutral
        // keyframe does not dissipate momentum: the v1 shot replay coasted
        // roughly half a metre through the ball and fell while every decision
        // cycle was already commanding Neutral.  A zero-velocity Walk keeps
        // phase-continuous balance until the measured torso speed is low
        // enough for the static neutral/kick actor to take over.  Do not add a
        // final-yaw target here; braking and turning at the same time caused
        // the same replay to overshoot from 17 to 24 degrees.
        WalkCommand brake_command;
        brake_command.target_2d_m = {0.0, 0.0};
        brake_command.target_absolute = false;
        brake_command.orientation_deg = std::nullopt;
        brake_command.orientation_absolute = false;
        brake_command.role_id = motion_role_id;
        return brake_command;
    };
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

    const bool push_position_valid =
        along_direction <= kDribbleMaxAheadM &&
        lateral_offset <= kDribbleMaxLateralOffsetM &&
        context.ball_distance <= field_geometry::kPushBallEngageDistanceM;
    if (context.state.dribble_ready &&
        (!push_position_valid || !latched_position_ready)) {
        context.state.dribble_ready = false;
        context.state.kick_setup_stable_since_s = 0.0;
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

    if (!context.state.dribble_ready) {
        if (fallback_allowed) {
            return make_fallback_command();
        }
        if (use_static_release_setup && context.state.kick_pre_settling) {
            if (planar_speed_mps > kKickPreSettleExitSpeedMps) {
                context.state.kick_pre_settle_stable_since_s = 0.0;
                trace_setup_gate(9, "pre-settle");
                return make_controlled_brake_command();
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
        if (use_static_release_setup && near_release_slot &&
            planar_speed_mps > kKickPreSettleEntrySpeedMps) {
            // The previous controller waited for the centimetre-scale slot
            // before braking.  Natural play then crossed the slot at roughly
            // 0.7--0.8 m/s and never satisfied the release debounce. Start a
            // phase-continuous gait brake in a bounded pre-slot corridor,
            // then enter neutral capture once the body is slow enough.
            context.state.kick_pre_settling = true;
            context.state.kick_pre_settle_stable_since_s = 0.0;
            trace_setup_gate(9, "pre-settle");
            return make_controlled_brake_command();
        }
        // Do not latch an exact static release before the pre-settle state
        // above has actively removed approach momentum.  The v1/v2 shot
        // replays reached the centimetre slot at 0.28--0.62 m/s; the old
        // eager latch skipped braking, jumped straight to Neutral, crossed
        // the ball and fell before the procedural runner could start.
        if (position_ready) {
            context.state.dribble_ready = true;
        }
    }

    if (!context.state.dribble_ready) {
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
        if (!needs_side_step && ball_position_actionable &&
            context.ball_distance <= kDribblePrecisionEntryDistanceM) {
            if (std::abs(command_lateral_error) >
                kKickCoarseRelocateLateralErrorM) {
                trace_setup_gate(10, "precision-turn-walk");
                const double travel_heading_deg = math::vector_angle_deg({
                    canonical_setup_target[0] - context.self[0],
                    canonical_setup_target[1] - context.self[1],
                });
                const double travel_heading_error_deg = std::abs(
                    math::normalize_deg(travel_heading_deg - self_yaw_deg));
                if (travel_heading_error_deg >
                    kKickCoarseTravelTurnThresholdDeg) {
                    WalkCommand turn_command = make_walk_command(context.self);
                    turn_command.orientation_deg = travel_heading_deg;
                    turn_command.orientation_absolute = true;
                    turn_command.orientation_gain = kKickSetupOrientationGain;
                    turn_command.role_id = motion_role_id;
                    return turn_command;
                }
                // Do not reuse generic point navigation here. Its 0.30 m
                // near-target stop is appropriate for formation movement but
                // stranded the v19 shot roughly ten times farther from the
                // centimetre-scale release slot than the action permits.
                // Once facing the setup point, keep a bounded forward crawl
                // until lateral error enters the independent fine controller.
                const double relocation_distance_m = math::planar_dist(
                    context.self, canonical_setup_target);
                const double maximum_relocation_speed_mps =
                    restart_plan != nullptr
                        ? kRestartMaxPrecisionForwardSpeedMps
                        : kDribbleMaxForwardSetupSpeedMps;
                const double relocation_speed_mps = std::clamp(
                    2.5 * relocation_distance_m,
                    0.12,
                    maximum_relocation_speed_mps);
                WalkCommand relocation_command;
                relocation_command.target_2d_m = {
                    relocation_speed_mps, 0.0};
                relocation_command.target_absolute = false;
                relocation_command.orientation_deg = travel_heading_deg;
                relocation_command.orientation_absolute = true;
                relocation_command.orientation_gain =
                    kKickSetupOrientationGain;
                relocation_command.role_id = motion_role_id;
                return relocation_command;
            }
            const double final_turn_threshold_deg = use_procedural_kick
                ? std::min(
                      kKickFinalTurnThresholdDeg,
                      release_orientation_tolerance_deg)
                : kKickFinalTurnThresholdDeg;
            if (orientation_error_deg > final_turn_threshold_deg) {
                trace_setup_gate(11, "precision-face-target");
                WalkCommand turn_command = make_walk_command(context.self);
                turn_command.orientation_deg = absolute_direction_deg;
                turn_command.orientation_absolute = true;
                turn_command.orientation_gain = kKickSetupOrientationGain;
                turn_command.role_id = motion_role_id;
                return turn_command;
            }
            trace_setup_gate(2, "precision-position");
            const double maximum_forward_setup_speed_mps =
                restart_plan != nullptr
                    ? kRestartMaxPrecisionForwardSpeedMps
                    : kDribbleMaxForwardSetupSpeedMps;
            const double maximum_reverse_setup_speed_mps =
                restart_plan != nullptr
                    ? kRestartMaxPrecisionReverseSpeedMps
                    : kDribbleMaxReverseSetupSpeedMps;
            const double forward_speed = std::clamp(
                kDribbleLongitudinalGain *
                    (behind_distance - command_behind_m),
                -maximum_reverse_setup_speed_mps,
                maximum_forward_setup_speed_mps);
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
    // Face the actual approach waypoint while it is still more than the
    // precision-entry distance away. Overwriting this with the eventual kick
    // heading asked the forward-dominant walk to strafe: the v20 goalkeeper
    // spent 15 seconds moving around, rather than toward, its goal-kick ball.
    // Final kick orientation is restored by the near-field controller above.
    return make_walk_command_avoiding(
        approach_target, context.snapshot, std::nullopt, true, false,
        motion_role_id, false, false);
}

    const bool contact_state_stable =
        ball_position_actionable && latched_position_ready &&
        orientation_error_deg <= release_orientation_tolerance_deg &&
        (!use_procedural_kick || transition_state.ready) &&
        planar_speed_mps <= (use_learned_transition
            ? decision::kick_contract::
                  kLearnedTransitionMaximumStartPlanarSpeedMps
            : decision::kick_contract::kProceduralMaximumStartPlanarSpeedMps);
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
                : use_phase_conditioned_transition
                    ? kPhaseConditionedKickSetupStableHoldS
                : use_procedural_dribble
                    ? kProceduralDribbleSetupStableHoldS
                    : use_procedural_kick
                        ? kProceduralKickSetupStableHoldS
                        : kKickSetupStableHoldS);
    const bool legal_kick =
        (context.snapshot.play_mode == world::PlayMode::PlayOn ||
         context.snapshot.play_mode_group == world::PlayModeGroup::OurKick) &&
        ball_position_actionable &&
        now >= context.state.next_kick_allowed_s &&
        context.ball_distance >= kKickMinBallDistanceM &&
        context.ball_distance <= kKickMaxBallDistanceM &&
        orientation_error_deg <= release_orientation_tolerance_deg &&
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
        if (restart_plan != nullptr &&
            orientation_error_deg <= required_orientation_error_deg) {
            // The frozen anchor is already in the release corridor.  Neutral
            // removes residual gait momentum while coordination catches up;
            // a zero-target Walk can retain enough sway to brush the ball and
            // start play without an attributable contact command.
            return NeutralCommand{};
        }
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
        kick_command.allow_forward_contact_fallback =
            fallback_due && static_pass_fallback_eligible;
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
        if (orientation_error_deg > release_orientation_tolerance_deg) {
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
        std::array<double, 2> their_goal =
            field_geometry::actual_their_goal_center_target();
        if (needs_final_third_cut_in(context.ball)) {
            // A direct line to even the near post still spends too much of a
            // one-metre pressure push in +x when the ball is several metres
            // outside the goal mouth.  Aim only 1.5 m farther forward and
            // well inside the post, producing a steep but still advancing
            // cut-in.  This is deliberately the continuous pressure actor:
            // the current exact short touch has not earned the right to stop
            // a live wide attack for a centimetre-scale setup.
            their_goal = {
                std::min(
                    field_geometry::kActualHalfLengthM,
                    context.ball[0] + kFinalThirdCutInAdvanceM),
                std::copysign(kFinalThirdCutInTargetYM, context.ball[1])};
        } else {
        // In the final six metres, driving every ball back toward the goal
        // centre creates a large and unnecessary turn near either post.  A
        // natural 7v7 goal-line sequence had the ball at y=0.67 m and the
        // body facing +24 deg; centre aim demanded -58 deg even though the
        // +1 m goal-mouth lane was open. Select the safe in-goal aim point
        // with the smallest current-body turn. Farther out, centre aim still
        // prevents a long carry from drifting toward a post/corner.
        constexpr double kFinalThirdGoalAimDepthM = 6.0;
        if (their_goal[0] - context.ball[0] <= kFinalThirdGoalAimDepthM) {
            constexpr std::array<double, 3> kSafeGoalAimYM{
                {-1.0, 0.0, 1.0}};
            const double self_yaw_deg =
                world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                    context.snapshot.self.orientation_wxyz);
            double best_turn_deg = std::numeric_limits<double>::infinity();
            for (const double target_y_m : kSafeGoalAimYM) {
                const std::array<double, 2> candidate{
                    their_goal[0], target_y_m};
                const auto candidate_direction =
                    math::vec2_sub(candidate, context.ball);
                if (math::norm2(candidate_direction) <= 1.0e-6) continue;
                const double turn_deg = std::abs(math::normalize_deg(
                    math::vector_angle_deg(candidate_direction) -
                    self_yaw_deg));
                if (turn_deg < best_turn_deg) {
                    best_turn_deg = turn_deg;
                    their_goal = candidate;
                }
            }
        }
        }
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
        !is_terminal(feedback.status) ||
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
    // A completed pass motion is only the start of physical-outcome
    // tracking; keep its command identity until the receiver-zone/timeout
    // lifecycle resolves. Local dribble/shot/clear actions have no such
    // downstream handshake and may release their setup commitment here.
    if (feedback.status == ExecutionStatus::Completed &&
        active_kick.mode == KickMode::TargetedPass) {
        return;
    }

    const bool matching_pass =
        active_kick.mode == KickMode::TargetedPass &&
        state_.committed_pass.has_value() &&
        state_.committed_pass->action_id == active_kick.action_id &&
        state_.committed_pass->sequence_id == active_kick.sequence_id;
    const bool matching_local_action =
        state_.committed_local_action.has_value() &&
        state_.committed_local_action->action_id == active_kick.action_id;
    state_.active_kick_command.reset();
    state_.kick_active_until_s = 0.0;
    state_.dribble_ready = false;
    state_.kick_pre_settling = false;
    state_.kick_pre_settle_stable_since_s = 0.0;
    state_.kick_setup_stable_since_s = 0.0;
    state_.kick_setup_started_s = 0.0;
    state_.kick_setup_best_pose_error_m = -1.0;
    state_.kick_setup_last_progress_s = 0.0;
    state_.kick_setup_mode_key = -1;
    if (matching_local_action) {
        state_.committed_local_action.reset();
        state_.local_action_commit_until_s = 0.0;
    }
    if (matching_pass && is_failure(feedback.status)) {
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
    bool enable_targeted_kick,
    bool enable_learned_kick) const {
    static const APNodePtr ap_tree = bt::command<APDecisionContext>(make_ap_push_ball_to_goal);

    if (!is_our_set_play(snapshot)) {
        state_.set_play_released = false;
    }
    if (state_.committed_local_action.has_value() &&
        state_.active_kick_command.has_value() &&
        state_.active_kick_command->mode != KickMode::TargetedPass &&
        state_.kick_active_until_s > 0.0 &&
        snapshot.server_time >= state_.kick_active_until_s) {
        state_.committed_local_action.reset();
        state_.local_action_commit_until_s = 0.0;
    }
    if (snapshot.play_mode != world::PlayMode::PlayOn) {
        state_.committed_local_action.reset();
        state_.local_action_commit_until_s = 0.0;
    }

    APDecisionContext context{
        snapshot,
        state_,
        enable_targeted_kick,
        enable_learned_kick,
        {snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
        {snapshot.self.position_m[0], snapshot.self.position_m[1]},
        0.0};
    context.ball_distance = math::planar_dist(context.ball, context.self);
    context.urgent_contest = urgent_ball_contest(
        snapshot, context.ball_distance);
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
            if (!snapshot.ball.position_valid &&
                restart->plan->ball_anchor_valid) {
                context.ball = restart->plan->ball_anchor_m;
                context.ball_distance = math::planar_dist(
                    context.ball, context.self);
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

    // Ball-action state and ball-search state are intentionally different.
    // An invalid position must never enter the precision/contact controller,
    // but returning to formation after every 0.20 s camera occlusion gives up
    // pressure too easily. TeamTactics may therefore assign SearchBall to the
    // AP for a bounded recent-last-seen window. Honor that movement-only target
    // here; once it expires the same path honestly returns to formation.
    if (snapshot.play_mode == world::PlayMode::PlayOn &&
        !snapshot.ball.position_valid) {
        state_.pressure_push_latched = false;
        state_.dribble_ready = false;
        state_.kick_pre_settling = false;
        state_.kick_pre_settle_stable_since_s = 0.0;
        state_.kick_setup_stable_since_s = 0.0;
        state_.kick_setup_started_s = 0.0;
        state_.kick_setup_best_pose_error_m = -1.0;
        state_.kick_setup_last_progress_s = 0.0;
        state_.kick_setup_mode_key = -1;
        state_.committed_local_action.reset();
        state_.local_action_commit_until_s = 0.0;
        state_.active_kick_command.reset();
        state_.kick_active_until_s = 0.0;
        if (state_.pass_lifecycle.active() &&
            !state_.pass_lifecycle.terminal()) {
            state_.pass_lifecycle.cancel(snapshot.server_time);
            publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
        }
        state_.committed_pass.reset();
        state_.pass_commit_until_s = 0.0;

        const TacticalTarget target = blackboard.get<TacticalTarget>(
            Blackboard::kKeyTacticalTarget);
        return make_walk_command_avoiding(
            target.position_m,
            snapshot,
            std::nullopt,
            true,
            false,
            RoleManager::ROLE_AP,
            false,
            true);
    }

    const TacticalTarget assigned_target = blackboard.get<TacticalTarget>(
        Blackboard::kKeyTacticalTarget);
    const bool active_kick_motion =
        state_.active_kick_command.has_value() &&
        snapshot.server_time < state_.kick_active_until_s;
    if (assigned_target.duty == TacticalDuty::Cover &&
        !context.urgent_contest && !active_kick_motion) {
        // TeamTactics uses Cover only for an active goalkeeper smother or a
        // configured protect-lead phase. The old AP behavior claimed to
        // preserve that target but unconditionally executed ball pressure.
        // Make the team-level assignment real while retaining the urgent
        // near-ball override above it.
        state_.pressure_push_latched = false;
        state_.committed_local_action.reset();
        state_.local_action_commit_until_s = 0.0;
        if (state_.pass_lifecycle.active() &&
            !state_.pass_lifecycle.terminal()) {
            state_.pass_lifecycle.cancel(snapshot.server_time);
            publish_pass_lifecycle(state_.pass_lifecycle, blackboard);
        }
        state_.committed_pass.reset();
        state_.pass_commit_until_s = 0.0;
        return make_walk_command_avoiding(
            assigned_target.position_m, snapshot, std::nullopt,
            true, true, RoleManager::ROLE_AP);
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
        // Admit the full motion-layer transition envelope. Speeds above the
        // neutral-release band enter the explicit pre-settle branch; rejecting
        // them here prevented that branch from doing the braking it owns.
        constexpr double kSpecialistSetupMaximumSpeedMps =
            decision::kick_contract::kProceduralMaximumStartPlanarSpeedMps;
        constexpr double kSpecialistMinimumOpponentEtaS = 1.0;
        constexpr double kLocalActionAbortRaceMarginS = 0.25;
        const bool opponent_clearly_wins_ball =
            tactical_state.possession == strategy::PossessionOwner::Theirs &&
            tactical_state.nearest_opponent_ball_time_s +
                    kLocalActionAbortRaceMarginS <
                tactical_state.nearest_teammate_ball_time_s;
        const bool controlled_specialist_setup =
            !context.urgent_contest &&
            context.ball_distance <= kSpecialistSetupMaximumBallDistanceM &&
            planar_speed_mps <= kSpecialistSetupMaximumSpeedMps &&
            tactical_state.possession == strategy::PossessionOwner::Ours &&
            tactical_state.ball_owner_is_teammate &&
            tactical_state.ball_owner_player_number == snapshot.player_number &&
            tactical_state.nearest_opponent_ball_time_s >=
                kSpecialistMinimumOpponentEtaS;
        const bool force_continuous_cut_in =
            needs_final_third_cut_in(context.ball);
        // Admission, commitment retention, and physical release are distinct
        // contracts.  A calm self-owned state is required to propose a pass,
        // but the walk controller can briefly cross the admission speed while
        // braking in the ball slot.  Cancelling the proposal on that transient
        // prevents the receiver's 0.30 s Ready dwell from ever completing.
        // Retain only while the ball track is tactically fresh, still near the
        // passer, and no other player has clearly won it.  The executor-specific
        // body speed, pose, yaw, and model-domain gates still own release.
        const bool another_teammate_owns_ball =
            tactical_state.ball_owner_is_teammate &&
            tactical_state.ball_owner_player_number > 0 &&
            tactical_state.ball_owner_player_number != snapshot.player_number;
        // Team communication plus the receiver's physical Ready dwell can
        // exceed the ordinary tactical TTL while the ball is hidden by the
        // passer's torso.  Use only a short subset of the WorldState near-field
        // lease here; the general 3.5 s track remains far too old to authorize
        // a precision release.
        constexpr double kPassNearContactRetentionS = 1.5;
        const bool retained_pass_track_fresh =
            snapshot.ball.visible || snapshot.ball.position_age_s <= 0.75 ||
            (snapshot.ball.near_contact_track &&
             snapshot.ball.position_age_s <= kPassNearContactRetentionS);
        const bool retained_pass_setup =
            snapshot.ball.position_valid &&
            retained_pass_track_fresh &&
            context.ball_distance <= kSpecialistSetupMaximumBallDistanceM &&
            !another_teammate_owns_ball &&
            !opponent_clearly_wins_ball;
        // FCP's released competition baseline starts its complete kick
        // behavior unless an opponent is clearly closer to the ball. Apply
        // that evidence only to urgent Shoot/Clear actions: a static short
        // dribble or coordinated pass still needs calm self-owned possession,
        // while a goal chance or defensive emergency may begin positioning
        // from the full precision-entry distance.
        const bool urgent_local_setup_available =
            !context.urgent_contest &&
            context.ball_distance <= kDribblePrecisionEntryDistanceM &&
            planar_speed_mps <= kSpecialistSetupMaximumSpeedMps &&
            (snapshot.ball.visible || snapshot.ball.position_age_s <= 0.75) &&
            tactical_state.phase != strategy::TacticalPhase::Unknown &&
            !opponent_clearly_wins_ball;
        const auto local_action_setup_available =
            [&](const strategy::CooperativeAction& action) {
                if (action.category == strategy::ActionCategory::Shoot) {
                    return urgent_local_setup_available &&
                        static_shot_setup_feasible(
                            snapshot, action, tactical_state);
                }
                if (action.category == strategy::ActionCategory::Clear) {
                    return urgent_local_setup_available;
                }
                if (action.category != strategy::ActionCategory::Dribble) {
                    return controlled_specialist_setup;
                }
                if (force_continuous_cut_in) return false;
                const auto direction = math::vec2_sub(
                    action.target_point_m, context.ball);
                if (math::norm2(direction) <= 1.0e-6) return false;
                const double self_yaw_deg =
                    world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                        snapshot.self.orientation_wxyz);
                // Setup-aware dribble generation asks for at most 30 degrees.
                // Keep a small perception margin here, but do not admit the
                // retained direct-goal proposal of a backwards-facing player.
                constexpr double kDribbleAdmissionMaximumTurnDeg = 35.0;
                return controlled_specialist_setup &&
                    static_dribble_setup_feasible(snapshot, action) &&
                    std::abs(math::normalize_deg(
                        math::vector_angle_deg(direction) - self_yaw_deg)) <=
                        kDribbleAdmissionMaximumTurnDeg;
            };
        constexpr double kLocalActionCommitDurationS = 2.0;
        constexpr double kStrongKickCommitDurationS = 3.0;
        const bool local_action_motion_active =
            state_.active_kick_command.has_value() &&
            state_.active_kick_command->mode != KickMode::TargetedPass &&
            snapshot.server_time < state_.kick_active_until_s;
        const bool committed_static_setup_lost =
            state_.committed_local_action.has_value() &&
            ((state_.committed_local_action->category ==
                  strategy::ActionCategory::Shoot &&
              !static_shot_setup_feasible(
                  snapshot, *state_.committed_local_action, tactical_state)) ||
             (state_.committed_local_action->category ==
                  strategy::ActionCategory::Dribble &&
              !static_dribble_setup_feasible(
                  snapshot, *state_.committed_local_action)));
        if (state_.committed_local_action.has_value() &&
            !local_action_motion_active &&
            (snapshot.server_time >= state_.local_action_commit_until_s ||
             !snapshot.ball.position_valid ||
             context.ball_distance > kDribblePrecisionEntryDistanceM ||
             (force_continuous_cut_in &&
              state_.committed_local_action->category ==
                  strategy::ActionCategory::Dribble) ||
             committed_static_setup_lost ||
             opponent_clearly_wins_ball ||
             context.urgent_contest)) {
            state_.committed_local_action.reset();
            state_.local_action_commit_until_s = 0.0;
        }
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
             !retained_pass_setup ||
             force_continuous_cut_in ||
             context.urgent_contest ||
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

        const auto is_local_ball_action = [](strategy::ActionCategory category) {
            return category == strategy::ActionCategory::Dribble ||
                   category == strategy::ActionCategory::Shoot ||
                   category == strategy::ActionCategory::Clear;
        };
        if (!state_.committed_local_action.has_value() &&
            (!state_.committed_pass.has_value() ||
             state_.pass_lifecycle.terminal()) &&
            plan.selected.has_value() &&
            is_local_ball_action(plan.selected->category) &&
            local_action_setup_available(*plan.selected) &&
            capabilities.supported(*plan.selected)) {
            state_.committed_local_action = *plan.selected;
            state_.local_action_commit_until_s =
                snapshot.server_time +
                ((plan.selected->category == strategy::ActionCategory::Shoot ||
                  plan.selected->category == strategy::ActionCategory::Clear)
                     ? kStrongKickCommitDurationS
                     : kLocalActionCommitDurationS);
        }
        if (state_.committed_local_action.has_value()) {
            // Role assignment precedes action selection each cycle. A short
            // rolling lease, refreshed only while this commitment remains
            // live, prevents a second near-equal player from taking AP and
            // abandoning a partially completed precision setup.
            constexpr double kActionApLeaseDurationS = 0.35;
            role_manager.retain_self_as_ap_for_action(
                snapshot.player_number, snapshot, kActionApLeaseDurationS);
            plan.selected = *state_.committed_local_action;
            blackboard.set(Blackboard::kKeyStrategyPlan, plan);
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *state_.committed_local_action);
            const auto target_direction = math::vec2_sub(
                state_.committed_local_action->target_point_m, context.ball);
            if (math::norm2(target_direction) > 1.0e-6) {
                return make_dribble_command(
                    context,
                    math::vector_angle_deg(target_direction),
                    &*state_.committed_local_action);
            }
            state_.committed_local_action.reset();
            state_.local_action_commit_until_s = 0.0;
        }

        constexpr double kPassCommitDurationS = 6.0;
        if (!state_.committed_pass.has_value() && plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Pass &&
            snapshot.server_time >= state_.pass_retry_after_s &&
            !force_continuous_cut_in &&
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
            constexpr double kActionApLeaseDurationS = 0.35;
            role_manager.retain_self_as_ap_for_action(
                snapshot.player_number, snapshot, kActionApLeaseDurationS);
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
            // Hold is the evaluator's reference action, not a useful
            // open-play motor primitive for the active player.  Natural-match
            // traces exposed long Neutral runs at the touchline and even when
            // an opponent's ball ETA was 0.25 s: the reference utility had
            // beaten a boundary-risky dribble, so the only pressure player
            // simply watched the opponent take the ball.  Preserve Hold in
            // telemetry for planner diagnosis, then fall through to the
            // continuous pressure controller below.
        }
        if (plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Move &&
            !context.urgent_contest) {
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *plan.selected);
            // Move is useful planner telemetry, but walking *to* the ball stops
            // at the navigation radius and can strand the only pressure player
            // beside it.  Let the AP's motion-level approach/push controller
            // own translation so its target continues through the ball.
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

    if (snapshot.play_mode == world::PlayMode::PlayOn &&
        snapshot.ball.position_valid &&
        (assigned_target.duty == TacticalDuty::Formation ||
         context.urgent_contest)) {
        blackboard.set(
            Blackboard::kKeyTacticalTarget,
            TacticalTarget{
                TacticalDuty::Pressure,
                field_geometry::keep_field_player_outside_our_goalie_area(
                    context.ball),
                context.ball,
                0,
                1.0});
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
            if (!snapshot.ball.position_valid &&
                restart->plan->ball_anchor_valid) {
                clearance_context.ball = restart->plan->ball_anchor_m;
                clearance_context.ball_distance = math::planar_dist(
                    clearance_context.ball, clearance_context.self);
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

    // If the ball has reached the keeper's side or back inside the final
    // metre, a turn-first walk target can pull the body out of the goal path.
    // This occurred after a successful smother approach in the 2026-09-06
    // comparison: the cached ball was 0.49 m away and only 0.06 m behind the
    // torso plane, yet a duty re-evaluation commanded a retreat and the keeper
    // fell. Preserve the last-line body block until the ball moves forward or
    // away. This guard is independent of the tactical duty because noisy yaw
    // reach estimates may legitimately switch Smother back to Hold.
    const double goalkeeper_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double goalkeeper_yaw_rad = math::deg_to_rad(goalkeeper_yaw_deg);
    const auto ball_from_keeper = math::vec2_sub(context.ball, context.self);
    const double ball_forward_m =
        ball_from_keeper[0] * std::cos(goalkeeper_yaw_rad) +
        ball_from_keeper[1] * std::sin(goalkeeper_yaw_rad);
    constexpr double kGoalkeeperLastLineGuardDistanceM = 0.65;
    constexpr double kGoalkeeperLastLineForwardPlaneM = 0.10;
    if (snapshot.play_mode == world::PlayMode::PlayOn &&
        snapshot.ball.position_valid &&
        context.ball[0] <=
            -field_geometry::kActualHalfLengthM + 1.0 &&
        context.ball_distance <= kGoalkeeperLastLineGuardDistanceM &&
        ball_forward_m <= kGoalkeeperLastLineForwardPlaneM) {
        clearance_state_ = {};
        return NeutralCommand{};
    }

    // A successful smother must hand the ball back to open play. The same
    // action planner and exact capability contract used by the active player
    // select a forward safety clear; no procedural/learned capability means
    // this branch remains a walk/hold instead of inventing contact. Preserve
    // an already-admitted clear across a one-cycle Smother -> Hold estimate
    // change: otherwise the moving ball produces a new quantized target and
    // action id on every re-entry, resetting the precision setup indefinitely.
    const bool goalkeeper_ball_track_fresh = snapshot.ball.visible ||
        snapshot.ball.position_age_s <= 0.75;
    const bool committed_goalkeeper_clear =
        clearance_state_.committed_local_action.has_value() &&
        snapshot.ball.position_valid &&
        goalkeeper_ball_track_fresh &&
        context.ball_distance <= kDribblePrecisionEntryDistanceM;
    const bool active_goalkeeper_clear =
        clearance_state_.active_kick_command.has_value() &&
        clearance_state_.active_kick_command->mode == KickMode::Clear &&
        clearance_state_.kick_active_until_s > snapshot.server_time;
    if (snapshot.play_mode == world::PlayMode::PlayOn &&
        snapshot.ball.position_valid &&
        (tactical_target.duty == TacticalDuty::GoalkeeperSmother ||
         committed_goalkeeper_clear || active_goalkeeper_clear)) {
        APDecisionContext clearance_context{
            snapshot,
            clearance_state_,
            enable_targeted_kick,
            false,
            context.ball,
            context.self,
            context.ball_distance};
        // The composed strong-kick setup has a 2.6 s hard bound. Its semantic
        // commitment must outlive that controller bound or a live setup would
        // be discarded just before its final turn/release phase.
        constexpr double kGoalkeeperClearCommitDurationS = 3.0;
        const bool clear_motion_active =
            clearance_state_.active_kick_command.has_value() &&
            clearance_state_.kick_active_until_s > snapshot.server_time;
        if (clearance_state_.committed_local_action.has_value() &&
            !clear_motion_active &&
            (snapshot.server_time >=
                 clearance_state_.local_action_commit_until_s ||
             !snapshot.ball.position_valid ||
             !goalkeeper_ball_track_fresh ||
             context.ball_distance > kDribblePrecisionEntryDistanceM)) {
            clearance_state_.committed_local_action.reset();
            clearance_state_.local_action_commit_until_s = 0.0;
        }
        if (clear_motion_active) {
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
        bool goalkeeper_clear_setup_reachable = false;
        if (plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Clear) {
            const auto clear_delta = math::vec2_sub(
                plan.selected->target_point_m, context.ball);
            if (math::norm2(clear_delta) > 1.0e-6) {
                const auto clear_direction = math::vec2_unit_or(
                    clear_delta, {1.0, 0.0});
                const std::array<double, 2> clear_lateral{
                    -clear_direction[1], clear_direction[0]};
                const auto keeper_from_ball = math::vec2_sub(
                    context.self, context.ball);
                const double behind_ball_m = -(
                    keeper_from_ball[0] * clear_direction[0] +
                    keeper_from_ball[1] * clear_direction[1]);
                const double lateral_from_ball_m =
                    keeper_from_ball[0] * clear_lateral[0] +
                    keeper_from_ball[1] * clear_lateral[1];
                const double clear_heading_error_deg = std::abs(
                    math::normalize_deg(
                        math::vector_angle_deg(clear_delta) -
                        goalkeeper_yaw_deg));
                // A keeper already standing over the ball cannot safely spend
                // three seconds circling to the far side of it. In the v13
                // match that produced 22 alternating side-relocate/turn phases
                // from a 77--89 degree heading error and no contact. Admit the
                // strong clear only from a reachable goal-side approach cone;
                // otherwise keep executing the smother target and protect the
                // line until a later frame offers a real release pose.
                constexpr double kGoalkeeperClearMinimumBehindM = 0.15;
                constexpr double kGoalkeeperClearMaximumBehindM = 0.65;
                constexpr double kGoalkeeperClearMaximumLateralM = 0.25;
                constexpr double kGoalkeeperClearMaximumHeadingErrorDeg = 45.0;
                goalkeeper_clear_setup_reachable =
                    behind_ball_m >= kGoalkeeperClearMinimumBehindM &&
                    behind_ball_m <= kGoalkeeperClearMaximumBehindM &&
                    std::abs(lateral_from_ball_m) <=
                        kGoalkeeperClearMaximumLateralM &&
                    clear_heading_error_deg <=
                        kGoalkeeperClearMaximumHeadingErrorDeg;
            }
        }
        // First close down the goal-bound ball. Starting the centimetre-scale
        // clear setup from metres away made the keeper stop defending while
        // an opponent walked the ball over the line.
        constexpr double kGoalkeeperClearEngageDistanceM = 0.55;
        if (!clearance_state_.committed_local_action.has_value() &&
            plan.selected.has_value() &&
            plan.selected->category == strategy::ActionCategory::Clear &&
            capabilities.supported(*plan.selected) &&
            goalkeeper_ball_track_fresh &&
            goalkeeper_clear_setup_reachable &&
            context.ball_distance <= kGoalkeeperClearEngageDistanceM) {
            clearance_state_.committed_local_action = *plan.selected;
            clearance_state_.local_action_commit_until_s =
                snapshot.server_time + kGoalkeeperClearCommitDurationS;
        }
        if (clearance_state_.committed_local_action.has_value()) {
            blackboard.set(
                Blackboard::kKeySelectedCooperativeAction,
                *clearance_state_.committed_local_action);
            const auto target_direction = math::vec2_sub(
                clearance_state_.committed_local_action->target_point_m,
                context.ball);
            if (math::norm2(target_direction) > 1.0e-6) {
                return make_dribble_command(
                    clearance_context,
                    math::vector_angle_deg(target_direction),
                    &*clearance_state_.committed_local_action,
                    true,
                    RoleManager::ROLE_GK);
            }
            clearance_state_.committed_local_action.reset();
            clearance_state_.local_action_commit_until_s = 0.0;
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
    bool enable_targeted_kick,
    bool enable_learned_kick) const {
    // AP is the only behavior that needs RoleManager (to latch the set-play
    // push); dispatch it directly and let the other behaviors share the
    // 2-param base interface.
    if (ap_.matches(blackboard)) {
        return ap_.make_command(
            snapshot, blackboard, role_manager, enable_pass_strategy,
            enable_targeted_kick, enable_learned_kick);
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
