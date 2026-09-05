// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/action_planner.h"

#include "src/decision/kick_contract.h"
#include "src/math/math_utils.h"
#include "src/server/server_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

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

    constexpr double kDribbleTouchDistanceM = 0.55;
    actions.push_back(make_local_action(
        snapshot,
        ActionCategory::Dribble,
        math::vec2_add(
            ball,
            math::vec2_scale(
                goal_direction,
                kDribbleTouchDistanceM)),
        decision::kick_contract::kProceduralDribbleRequestedSpeedMps));

    const double goal_distance_m = math::planar_dist(ball, goal);
    if (goal_distance_m >=
            decision::kick_contract::kProceduralShotMinimumTargetDistanceM &&
        goal_distance_m <=
            decision::kick_contract::kProceduralShotMaximumTargetDistanceM) {
        actions.push_back(make_local_action(
            snapshot,
            ActionCategory::Shoot,
            goal,
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
