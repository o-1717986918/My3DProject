// SPDX-License-Identifier: GPL-3.0-or-later
// Candidate structure follows the Cyrus2D strict-pass pattern, with all 2D
// cycle/decay and point-player assumptions replaced by conservative 3D time.

#include "src/strategy/pass_candidate_generator.h"

#include "src/math/math_utils.h"
#include "src/server/server_constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace strategy {

namespace {

bool fresh_observation(
    const world::PlayerObservation& player,
    double current_time_s,
    double maximum_age_s) {
    return player.seen ||
        (player.last_seen_time >= 0.0 &&
         current_time_s - player.last_seen_time <= maximum_age_s);
}

std::uint32_t stable_action_id(
    int actor,
    int receiver,
    PassType pass_type,
    const Position2& target) {
    const auto qx = static_cast<std::int32_t>(std::lround(target[0] * 4.0));
    const auto qy = static_cast<std::int32_t>(std::lround(target[1] * 4.0));
    std::uint32_t hash = 2166136261U;
    auto mix = [&](std::uint32_t value) {
        hash ^= value;
        hash *= 16777619U;
    };
    mix(static_cast<std::uint32_t>(actor));
    mix(static_cast<std::uint32_t>(receiver));
    mix(static_cast<std::uint32_t>(pass_type));
    mix(static_cast<std::uint32_t>(qx));
    mix(static_cast<std::uint32_t>(qy));
    return hash;
}

}  // namespace

PassCandidateGenerator::PassCandidateGenerator()
    : opponent_model_(ReachTimeModel::Parameters{
          1.35, 180.0, 0.05, 0.75, 0.05}) {}

PassCandidateGenerator::PassCandidateGenerator(Parameters parameters)
    : parameters_(parameters),
      ball_model_(BallTrajectoryModel::Parameters{
          parameters.requested_ball_speed_mps, 0.08, 0.20}),
      opponent_model_(ReachTimeModel::Parameters{
          1.35, 180.0, 0.05, 0.75, 0.05}) {}

CandidateGenerationResult PassCandidateGenerator::generate(
    const world::WorldSnapshot& snapshot) const {
    CandidateGenerationResult result;
    if (snapshot.play_mode != world::PlayMode::PlayOn) {
        result.rejections.push_back({
            PassType::None, 0, {0.0, 0.0}, RejectionReason::NotOpenPlay});
        return result;
    }
    if (!snapshot.ball.visible) {
        result.rejections.push_back({
            PassType::None, 0, {0.0, 0.0}, RejectionReason::BallNotVisible});
        return result;
    }

    for (const auto& teammate : snapshot.teammates) {
        if (teammate.player_number <= 0 ||
            teammate.player_number == snapshot.player_number) {
            continue;
        }
        const Position2 receiver_position{
            teammate.position_m[0], teammate.position_m[1]};
        if (!fresh_observation(
                teammate, snapshot.server_time,
                parameters_.maximum_observation_age_s)) {
            result.rejections.push_back({
                PassType::Direct, teammate.player_number, receiver_position,
                RejectionReason::TeammateStale});
            continue;
        }
        if (teammate.fallen) {
            result.rejections.push_back({
                PassType::Direct, teammate.player_number, receiver_position,
                RejectionReason::TeammateFallen});
            continue;
        }

        CooperativeAction direct;
        const RejectionReason direct_reason = evaluate_candidate(
            snapshot, teammate.player_number, receiver_position, teammate.fallen,
            PassType::Direct, receiver_position, &direct);
        if (direct_reason == RejectionReason::None) {
            result.candidates.push_back(direct);
        } else {
            result.rejections.push_back({
                PassType::Direct, teammate.player_number, receiver_position,
                direct_reason});
        }

        if (!parameters_.enable_leading_pass) continue;
        const Position2 leading_target{
            receiver_position[0] + parameters_.leading_offset_m,
            receiver_position[1]};
        CooperativeAction leading;
        const RejectionReason leading_reason = evaluate_candidate(
            snapshot, teammate.player_number, receiver_position, teammate.fallen,
            PassType::Leading, leading_target, &leading);
        if (leading_reason == RejectionReason::None) {
            result.candidates.push_back(leading);
        } else {
            result.rejections.push_back({
                PassType::Leading, teammate.player_number, leading_target,
                leading_reason});
        }
    }
    return result;
}

RejectionReason PassCandidateGenerator::evaluate_candidate(
    const world::WorldSnapshot& snapshot,
    int receiver_player_number,
    const Position2& receiver_position_m,
    bool receiver_fallen,
    PassType pass_type,
    const Position2& target_point_m,
    CooperativeAction* output) const {
    if (output == nullptr || receiver_player_number <= 0 ||
        receiver_player_number == snapshot.player_number) {
        return RejectionReason::ReceiverInvalid;
    }
    if (receiver_fallen) return RejectionReason::TeammateFallen;
    if (std::abs(target_point_m[0]) >
            server_constants::kFieldHalfLengthM - parameters_.field_margin_m ||
        std::abs(target_point_m[1]) >
            server_constants::kFieldHalfWidthM - parameters_.field_margin_m) {
        return RejectionReason::OutOfField;
    }

    const Position2 ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const double distance = math::planar_dist(ball, target_point_m);
    if (distance < parameters_.minimum_pass_distance_m) {
        return RejectionReason::TooNear;
    }
    if (distance > parameters_.maximum_pass_distance_m) {
        return RejectionReason::TooFar;
    }
    if (ball[0] < parameters_.dangerous_backpass_x_m &&
        target_point_m[0] < ball[0] - 0.5) {
        return RejectionReason::UnsafeBackPass;
    }

    const double ball_time = ball_model_.travel_time_s(
        distance, parameters_.requested_ball_speed_mps);
    if (!std::isfinite(ball_time)) {
        return RejectionReason::BallCannotReach;
    }
    const double receiver_time = receiver_model_.estimate_s(
        receiver_position_m, target_point_m, std::nullopt, receiver_fallen);
    if (!std::isfinite(receiver_time) ||
        receiver_time + parameters_.minimum_receiver_lead_s > ball_time) {
        return RejectionReason::ReceiverLate;
    }

    double minimum_margin = std::numeric_limits<double>::infinity();
    const double opponent_time = earliest_opponent_time_s(
        snapshot, ball, target_point_m, parameters_.requested_ball_speed_mps,
        &minimum_margin);
    if (minimum_margin < parameters_.minimum_interception_margin_s) {
        return RejectionReason::OpponentFirst;
    }

    output->action_id = stable_action_id(
        snapshot.player_number, receiver_player_number, pass_type, target_point_m);
    output->category = ActionCategory::Pass;
    output->pass_type = pass_type;
    output->actor_player_number = snapshot.player_number;
    output->target_player_number = receiver_player_number;
    output->start_ball_point_m = ball;
    output->target_point_m = target_point_m;
    output->requested_ball_speed_mps = parameters_.requested_ball_speed_mps;
    output->predicted_ball_time_s = ball_time;
    output->predicted_receiver_time_s = receiver_time;
    output->predicted_opponent_time_s = opponent_time;
    output->interception_margin_s = std::isfinite(minimum_margin)
        ? minimum_margin
        : 10.0;
    output->confidence = std::clamp(output->interception_margin_s / 2.0, 0.0, 1.0);
    return RejectionReason::None;
}

double PassCandidateGenerator::earliest_opponent_time_s(
    const world::WorldSnapshot& snapshot,
    const Position2& ball_position_m,
    const Position2& target_point_m,
    double initial_ball_speed_mps,
    double* minimum_margin_s) const {
    constexpr int kSamples = 20;
    double earliest = std::numeric_limits<double>::infinity();
    double min_margin = std::numeric_limits<double>::infinity();

    auto consider = [&](const world::PlayerObservation& opponent) {
        if (!fresh_observation(
                opponent, snapshot.server_time,
                parameters_.maximum_observation_age_s)) {
            return;
        }
        const Position2 opponent_position{
            opponent.position_m[0], opponent.position_m[1]};
        for (int i = 1; i <= kSamples; ++i) {
            const double ratio = static_cast<double>(i) / kSamples;
            const Position2 point{
                ball_position_m[0] +
                    (target_point_m[0] - ball_position_m[0]) * ratio,
                ball_position_m[1] +
                    (target_point_m[1] - ball_position_m[1]) * ratio};
            const double path_distance = math::planar_dist(ball_position_m, point);
            const double ball_time = ball_model_.travel_time_s(
                path_distance, initial_ball_speed_mps);
            if (!std::isfinite(ball_time)) continue;
            const double opponent_time = opponent_model_.estimate_s(
                opponent_position, point, std::nullopt, opponent.fallen);
            earliest = std::min(earliest, opponent_time);
            min_margin = std::min(min_margin, opponent_time - ball_time);
        }
    };

    for (const auto& opponent : snapshot.opponents) consider(opponent);
    for (const auto& opponent : snapshot.shared_opponents) consider(opponent);
    if (minimum_margin_s != nullptr) *minimum_margin_s = min_margin;
    return earliest;
}

}  // namespace strategy
