// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/action_planner.h"

#include "src/decision/kick_contract.h"
#include "src/math/math_utils.h"
#include "src/server/server_constants.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace strategy {

namespace {

std::uint32_t stable_local_action_id(
    int actor,
    ActionCategory category,
    const Position2& target) {
    const auto qx = static_cast<std::int32_t>(std::lround(target[0] * 100.0));
    const auto qy = static_cast<std::int32_t>(std::lround(target[1] * 100.0));
    std::uint32_t hash = 2166136261U;
    const auto mix = [&hash](std::uint32_t value) {
        hash ^= value;
        hash *= 16777619U;
    };
    mix(static_cast<std::uint32_t>(actor));
    mix(static_cast<std::uint32_t>(category));
    mix(static_cast<std::uint32_t>(qx));
    mix(static_cast<std::uint32_t>(qy));
    return hash;
}

CooperativeAction make_local_action(
    const world::WorldSnapshot& snapshot,
    ActionCategory category,
    const Position2& target,
    double requested_speed_mps) {
    CooperativeAction action;
    action.action_id = stable_local_action_id(
        snapshot.player_number, category, target);
    action.category = category;
    action.actor_player_number = snapshot.player_number;
    action.start_ball_point_m = {
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    action.target_point_m = target;
    action.requested_ball_speed_mps = requested_speed_mps;
    action.confidence = 1.0;
    return action;
}

std::vector<CooperativeAction> local_ball_actions(
    const world::WorldSnapshot& snapshot) {
    std::vector<CooperativeAction> actions;
    if (snapshot.play_mode != world::PlayMode::PlayOn ||
        !snapshot.ball.position_valid) {
        return actions;
    }
    const Position2 ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const Position2 goal{server_constants::kFieldHalfLengthM, 0.0};
    const Position2 goal_direction = math::vec2_unit_or(
        math::vec2_sub(goal, ball), {1.0, 0.0});
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);

    actions.push_back(make_local_action(
        snapshot,
        ActionCategory::Hold,
        {snapshot.self.position_m[0], snapshot.self.position_m[1]},
        0.0));
    actions.push_back(make_local_action(
        snapshot,
        ActionCategory::Move,
        ball,
        1.0));

    // An exact touch aimed straight at goal can demand a 70--130 degree turn
    // while the robot stands over the ball. Natural-match traces showed that
    // such commitments consumed several seconds without one release. When a
    // forward-progressing intermediate heading exists, ask for at most one
    // 30 degree turn and let subsequent touches bend the carry toward goal.
    // If the body points too far away for that intermediate touch to advance
    // the ball, retain the direct-goal proposal; the decision-layer admission
    // guard will leave it to the continuous pressure controller instead.
    constexpr double kDribbleTouchDistanceM = 0.55;
    constexpr double kDribbleMaximumSetupTurnDeg = 30.0;
    constexpr double kDribbleMaximumResidualGoalErrorDeg = 75.0;
    const double goal_heading_deg = math::vector_angle_deg(goal_direction);
    const double goal_heading_error_deg = math::normalize_deg(
        goal_heading_deg - self_yaw_deg);
    double dribble_heading_deg = goal_heading_deg;
    const double bounded_heading_deg = self_yaw_deg + std::clamp(
        goal_heading_error_deg,
        -kDribbleMaximumSetupTurnDeg,
        kDribbleMaximumSetupTurnDeg);
    if (std::abs(math::normalize_deg(
            goal_heading_deg - bounded_heading_deg)) <=
        kDribbleMaximumResidualGoalErrorDeg) {
        dribble_heading_deg = bounded_heading_deg;
    }
    const double dribble_heading_rad = math::deg_to_rad(dribble_heading_deg);
    const Position2 dribble_direction{
        std::cos(dribble_heading_rad), std::sin(dribble_heading_rad)};
    actions.push_back(make_local_action(
        snapshot,
        ActionCategory::Dribble,
        math::vec2_add(
            ball,
            math::vec2_scale(
                dribble_direction,
                kDribbleTouchDistanceM)),
        decision::kick_contract::kProceduralDribbleRequestedSpeedMps));

    // A centre-only target made a near-post opportunity need an unnecessary
    // 30--40 degree setup turn. Sample three points safely inside the 1.83 m
    // half-width and select the executable one that costs the current body the
    // least rotation. A small centre bias keeps the most tolerant aim whenever
    // its turn cost is already comparable.
    constexpr std::array<double, 3> kSafeGoalAimYM{{-1.0, 0.0, 1.0}};
    std::optional<Position2> shot_target;
    double best_shot_setup_cost = std::numeric_limits<double>::infinity();
    for (const double target_y_m : kSafeGoalAimYM) {
        const Position2 candidate{server_constants::kFieldHalfLengthM, target_y_m};
        const double distance_m = math::planar_dist(ball, candidate);
        if (distance_m <
                decision::kick_contract::kProceduralShotMinimumTargetDistanceM ||
            distance_m >
                decision::kick_contract::kProceduralShotMaximumTargetDistanceM) {
            continue;
        }
        const Position2 direction = math::vec2_sub(candidate, ball);
        const double heading_error_deg = std::abs(math::normalize_deg(
            math::vector_angle_deg(direction) - self_yaw_deg));
        constexpr double kGoalCentreBiasDegPerM = 4.0;
        const double setup_cost = heading_error_deg +
            kGoalCentreBiasDegPerM * std::abs(target_y_m);
        if (setup_cost < best_shot_setup_cost - 1.0e-9 ||
            (std::abs(setup_cost - best_shot_setup_cost) <= 1.0e-9 &&
             (!shot_target.has_value() ||
              std::abs(target_y_m) < std::abs((*shot_target)[1])))) {
            best_shot_setup_cost = setup_cost;
            shot_target = candidate;
        }
    }
    if (shot_target.has_value()) {
        actions.push_back(make_local_action(
            snapshot,
            ActionCategory::Shoot,
            *shot_target,
            decision::kick_contract::kProceduralShotRequestedSpeedMps));
    }

    constexpr double kDefensiveClearDepthM = 10.0;
    constexpr double kClearTargetDistanceM = 6.0;
    if (ball[0] <=
        -server_constants::kFieldHalfLengthM + kDefensiveClearDepthM) {
        actions.push_back(make_local_action(
            snapshot,
            ActionCategory::Clear,
            math::vec2_add(
                ball,
                math::vec2_scale(goal_direction, kClearTargetDistanceM)),
            decision::kick_contract::kProceduralClearRequestedSpeedMps));
    }
    return actions;
}

}  // namespace

ActionPlanner::ActionPlanner() = default;

ActionPlanner::ActionPlanner(Parameters parameters)
    : parameters_(parameters) {}

PlanningResult ActionPlanner::plan(const world::WorldSnapshot& snapshot) const {
    return plan(snapshot, ActionCapabilityRegistry(true), true);
}

PlanningResult ActionPlanner::plan(
    const world::WorldSnapshot& snapshot,
    const ActionCapabilityRegistry& capabilities,
    bool enable_passes) const {
    return plan(
        snapshot, capabilities, enable_passes,
        build_tactical_state(snapshot));
}

PlanningResult ActionPlanner::plan(
    const world::WorldSnapshot& snapshot,
    const ActionCapabilityRegistry& capabilities,
    bool enable_passes,
    const TacticalState& tactical_state) const {
    PlanningResult result;
    result.tactical_state = tactical_state;
    if (enable_passes) {
        CandidateGenerationResult generated = pass_generator_.generate(snapshot);
        result.rejections = std::move(generated.rejections);
        for (auto& candidate : generated.candidates) {
            if (capabilities.supported(candidate)) {
                result.candidates.push_back(std::move(candidate));
            } else {
                result.rejections.push_back({
                    candidate.pass_type,
                    candidate.target_player_number,
                    candidate.target_point_m,
                    RejectionReason::CapabilityUnavailable});
            }
        }
    }
    for (auto& candidate : local_ball_actions(snapshot)) {
        if (capabilities.supported(candidate)) {
            result.candidates.push_back(std::move(candidate));
        } else {
            result.rejections.push_back({
                PassType::None,
                0,
                candidate.target_point_m,
                RejectionReason::CapabilityUnavailable});
        }
    }

    for (auto& candidate : result.candidates) {
        candidate.utility = field_evaluator_.evaluate(
            candidate, snapshot, result.tactical_state);
    }
    std::sort(
        result.candidates.begin(), result.candidates.end(),
        [](const CooperativeAction& lhs, const CooperativeAction& rhs) {
            if (std::abs(lhs.utility - rhs.utility) > 1.0e-9) {
                return lhs.utility > rhs.utility;
            }
            if (lhs.category != rhs.category) {
                return lhs.category < rhs.category;
            }
            if (lhs.target_player_number != rhs.target_player_number) {
                return lhs.target_player_number < rhs.target_player_number;
            }
            if (lhs.pass_type != rhs.pass_type) {
                return lhs.pass_type < rhs.pass_type;
            }
            return lhs.action_id < rhs.action_id;
        });

    if (!result.candidates.empty() &&
        result.candidates.front().utility >= parameters_.minimum_action_utility) {
        result.selected = result.candidates.front();
    } else if (!result.candidates.empty()) {
        const auto& best = result.candidates.front();
        result.rejections.push_back({
            best.pass_type, best.target_player_number, best.target_point_m,
            RejectionReason::BelowUtilityFloor});
    }
    return result;
}

}  // namespace strategy
