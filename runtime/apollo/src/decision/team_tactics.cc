// SPDX-License-Identifier: GPL-3.0-or-later
// Tactical concepts are independently implemented from patterns audited in
// Cyrus2D/HELIOS; all timing and targets use the Apollo 3D world contract.

#include "src/decision/team_tactics.h"

#include "src/decision/role_manager.h"
#include "src/math/math_utils.h"
#include "src/strategy/reach_time_model.h"
#include "src/strategy/tactical_state.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace decision {

namespace {

using field_geometry::Position2;

struct KnownOpponent {
    int player_number{0};
    Position2 position_m{0.0, 0.0};
};

Position2 clamp_field_player(Position2 point) {
    constexpr double margin = field_geometry::kFormationFieldMarginM;
    point[0] = std::clamp(
        point[0],
        -field_geometry::kActualHalfLengthM +
            field_geometry::kGoalieAreaDepthM + 0.8,
        field_geometry::kActualHalfLengthM - margin);
    point[1] = std::clamp(
        point[1],
        -field_geometry::kActualHalfWidthM + margin,
        field_geometry::kActualHalfWidthM - margin);
    return point;
}

Position2 clamp_goalkeeper(Position2 point) {
    point[0] = std::clamp(
        point[0],
        -field_geometry::kActualHalfLengthM + 0.25,
        -field_geometry::kActualHalfLengthM +
            field_geometry::kGoalieAreaDepthM - 0.4);
    point[1] = std::clamp(
        point[1],
        -field_geometry::kGoalHalfWidthM + 0.25,
        field_geometry::kGoalHalfWidthM - 0.25);
    return point;
}

bool fresh(const world::PlayerObservation& player, double now) {
    return !player.fallen &&
        (player.seen ||
         (player.last_seen_time >= 0.0 && now - player.last_seen_time <= 2.0));
}

void append_opponent(
    std::vector<KnownOpponent>& opponents,
    const world::PlayerObservation& candidate,
    double now) {
    if (!fresh(candidate, now)) return;
    const Position2 position{candidate.position_m[0], candidate.position_m[1]};
    for (const auto& existing : opponents) {
        const bool same_known_player = candidate.player_number > 0 &&
            existing.player_number == candidate.player_number;
        const bool both_anonymous = candidate.player_number <= 0 &&
            existing.player_number <= 0;
        const bool anonymous_duplicate = candidate.player_number <= 0 ||
            existing.player_number <= 0;
        if ((same_known_player || both_anonymous || anonymous_duplicate) &&
            math::planar_dist(existing.position_m, position) < 0.75) {
            return;
        }
    }
    opponents.push_back({candidate.player_number, position});
}

std::vector<KnownOpponent> known_opponents(const world::WorldSnapshot& snapshot) {
    std::vector<KnownOpponent> opponents;
    opponents.reserve(snapshot.opponents.size() + snapshot.shared_opponents.size());
    for (const auto& opponent : snapshot.opponents) {
        append_opponent(opponents, opponent, snapshot.server_time);
    }
    for (const auto& opponent : snapshot.shared_opponents) {
        append_opponent(opponents, opponent, snapshot.server_time);
    }
    std::sort(
        opponents.begin(), opponents.end(),
        [](const KnownOpponent& lhs, const KnownOpponent& rhs) {
            if (lhs.player_number != rhs.player_number) {
                return lhs.player_number < rhs.player_number;
            }
            if (lhs.position_m[0] != rhs.position_m[0]) {
                return lhs.position_m[0] < rhs.position_m[0];
            }
            return lhs.position_m[1] < rhs.position_m[1];
        });
    return opponents;
}

double nearest_opponent_distance(
    const Position2& point,
    const std::vector<KnownOpponent>& opponents) {
    double distance = std::numeric_limits<double>::infinity();
    for (const auto& opponent : opponents) {
        distance = std::min(distance, math::planar_dist(point, opponent.position_m));
    }
    return distance;
}

double lane_clearance(
    const Position2& ball,
    const Position2& target,
    const std::vector<KnownOpponent>& opponents) {
    double clearance = std::numeric_limits<double>::infinity();
    for (const auto& opponent : opponents) {
        clearance = std::min(
            clearance,
            math::point_segment_distance(opponent.position_m, ball, target));
    }
    return clearance;
}

double nearest_teammate_distance(
    const Position2& point,
    const world::WorldSnapshot& snapshot,
    int excluded_player_number) {
    double distance = std::numeric_limits<double>::infinity();
    for (const auto& teammate : snapshot.teammates) {
        if (teammate.player_number == excluded_player_number ||
            !fresh(teammate, snapshot.server_time)) {
            continue;
        }
        distance = std::min(distance, math::planar_dist(
            point, {teammate.position_m[0], teammate.position_m[1]}));
    }
    return distance;
}

TacticalTarget plan_support(
    const Position2& ball,
    const Position2& formation,
    int role_id,
    int player_number,
    const std::vector<KnownOpponent>& opponents,
    const world::WorldSnapshot& snapshot,
    strategy::TacticalRiskMode risk_mode) {
    const Position2 goal{field_geometry::kActualHalfLengthM, 0.0};
    const Position2 forward = math::vec2_unit_or(math::vec2_sub(goal, ball), {1.0, 0.0});
    const Position2 lateral = math::perpendicular_left(forward);
    const bool protect_lead = risk_mode == strategy::TacticalRiskMode::ProtectLead;
    const bool chase_goal = risk_mode == strategy::TacticalRiskMode::ChaseGoal;
    const double depth = role_id == RoleManager::ROLE_ST
        ? (protect_lead ? 3.2 : (chase_goal ? 5.5 : 4.5))
        : -1.8;
    const double width = role_id == RoleManager::ROLE_ST
        ? (protect_lead ? 2.5 : (chase_goal ? 3.5 : 3.0))
        : 4.0;
    std::vector<double> opponent_x;
    opponent_x.reserve(opponents.size());
    for (const auto& opponent : opponents) opponent_x.push_back(opponent.position_m[0]);
    std::sort(opponent_x.begin(), opponent_x.end(), std::greater<double>());
    const std::optional<double> offside_limit = opponent_x.size() >= 2U
        ? std::optional<double>{std::max(ball[0], opponent_x[1]) - 0.4}
        : std::nullopt;
    auto legalize_support = [&](Position2 point) {
        if (offside_limit.has_value()) point[0] = std::min(point[0], *offside_limit);
        return clamp_field_player(point);
    };
    const std::array<Position2, 4> candidates{{
        legalize_support(math::vec2_add(
            math::vec2_add(ball, math::vec2_scale(forward, depth)),
            math::vec2_scale(lateral, width))),
        legalize_support(math::vec2_add(
            math::vec2_add(ball, math::vec2_scale(forward, depth)),
            math::vec2_scale(lateral, -width))),
        legalize_support(math::vec2_add(
            math::vec2_add(ball, math::vec2_scale(forward, depth + 1.5)),
            math::vec2_scale(lateral, width * 0.55))),
        legalize_support(math::vec2_add(
            math::vec2_add(ball, math::vec2_scale(forward, depth + 1.5)),
            math::vec2_scale(lateral, -width * 0.55))),
    }};

    Position2 best = candidates.front();
    double best_score = -std::numeric_limits<double>::infinity();
    bool found = false;
    for (const auto& candidate : candidates) {
        if (math::planar_dist(candidate, formation) > 8.0) continue;
        const double opponent_space = std::min(
            nearest_opponent_distance(candidate, opponents), 6.0);
        const double pass_lane = std::min(lane_clearance(ball, candidate, opponents), 4.0);
        const double teammate_space = std::min(
            nearest_teammate_distance(candidate, snapshot, player_number), 4.0);
        if (teammate_space < 1.5) continue;
        const double formation_cost = math::planar_dist(candidate, formation);
        const double crowding_cost = teammate_space < 1.8
            ? 8.0 * (1.8 - teammate_space)
            : 0.0;
        const double score = 1.4 * opponent_space + pass_lane +
            0.35 * teammate_space - 0.18 * formation_cost - crowding_cost;
        if (score > best_score) {
            best_score = score;
            best = candidate;
            found = true;
        }
    }

    if (!found) {
        return {TacticalDuty::Formation, clamp_field_player(formation), ball, 0, 0.35};
    }

    const double old_space = nearest_opponent_distance(formation, opponents);
    const double new_space = nearest_opponent_distance(best, opponents);
    const bool escaped_pressure = std::isfinite(old_space) && old_space < 2.2 &&
        new_space > old_space + 0.5;
    return {
        escaped_pressure ? TacticalDuty::Unmark : TacticalDuty::Support,
        best,
        ball,
        0,
        opponents.empty() ? 0.55 : 0.8};
}

struct InterceptCandidate {
    int player_number{0};
    Position2 position_m{0.0, 0.0};
    double ball_arrival_s{0.0};
    double player_arrival_s{0.0};
};

bool legal_field_intercept(const Position2& point) {
    constexpr double margin = field_geometry::kFormationFieldMarginM;
    return point[0] > -field_geometry::kActualHalfLengthM +
            field_geometry::kGoalieAreaDepthM + 0.8 &&
        point[0] < field_geometry::kActualHalfLengthM - margin &&
        std::abs(point[1]) < field_geometry::kActualHalfWidthM - margin;
}

Position2 predicted_ball_position(
    const Position2& ball,
    const Position2& velocity,
    double time_s) {
    // This is deliberately a conservative 3D-runtime approximation, not a
    // transplanted 2D ballDecay constant. It bounds the projection at a stop
    // and is only used to choose a reachable walking target.
    constexpr double kConservativeBallDecelMps2 = 0.30;
    const double speed = math::norm2(velocity);
    if (speed <= 1.0e-6) return ball;
    const double travel_time = std::min(time_s, speed / kConservativeBallDecelMps2);
    const double travel_distance = std::max(
        0.0,
        speed * travel_time -
            0.5 * kConservativeBallDecelMps2 * travel_time * travel_time);
    return math::vec2_add(
        ball,
        math::vec2_scale(velocity, travel_distance / speed));
}

std::optional<Position2> player_position(
    const world::WorldSnapshot& snapshot,
    int player_number) {
    if (player_number == snapshot.player_number) {
        if (snapshot.self.position_m[2] < world::kFallenHeightThresholdM) {
            return std::nullopt;
        }
        return Position2{snapshot.self.position_m[0], snapshot.self.position_m[1]};
    }
    for (const auto& teammate : snapshot.teammates) {
        if (teammate.player_number == player_number &&
            fresh(teammate, snapshot.server_time)) {
            return Position2{teammate.position_m[0], teammate.position_m[1]};
        }
    }
    return std::nullopt;
}

std::optional<InterceptCandidate> select_intercept_owner(
    const world::WorldSnapshot& snapshot,
    const std::vector<RoleAssignment>& assignments) {
    if (!snapshot.ball.velocity_valid) return std::nullopt;
    const Position2 ball{snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const Position2 velocity{
        snapshot.ball.velocity_mps[0], snapshot.ball.velocity_mps[1]};
    if (math::norm2(velocity) < 0.35 || velocity[0] >= -0.10) {
        return std::nullopt;
    }

    const strategy::ReachTimeModel reach_model(
        strategy::ReachTimeModel::Parameters{0.90, 120.0, 0.25, 0.25, 0.20});
    constexpr std::array<int, 5> eligible_roles{
        RoleManager::ROLE_CDM,
        RoleManager::ROLE_CBM,
        RoleManager::ROLE_CBL,
        RoleManager::ROLE_CBR,
        RoleManager::ROLE_ST,
    };
    const bool team_scope = assignments.size() > 1U;
    for (double ball_arrival_s = 0.25; ball_arrival_s <= 3.0;
         ball_arrival_s += 0.25) {
        const Position2 target = predicted_ball_position(
            ball, velocity, ball_arrival_s);
        if (!legal_field_intercept(target)) continue;

        std::optional<InterceptCandidate> best_at_target;
        for (const auto& assignment : assignments) {
            if (!team_scope && assignment.role_id != RoleManager::ROLE_CDM) {
                continue;
            }
            if (std::find(
                    eligible_roles.begin(), eligible_roles.end(),
                    assignment.role_id) == eligible_roles.end()) {
                continue;
            }
            const auto position = player_position(snapshot, assignment.player_number);
            if (!position.has_value()) continue;
            const double eta = reach_model.estimate_s(*position, target);
            if (eta + 0.15 > ball_arrival_s) continue;

            const InterceptCandidate candidate{
                assignment.player_number, target, ball_arrival_s, eta};
            if (!best_at_target.has_value() ||
                candidate.player_arrival_s < best_at_target->player_arrival_s - 1.0e-9 ||
                (std::abs(candidate.player_arrival_s -
                          best_at_target->player_arrival_s) <= 1.0e-9 &&
                 candidate.player_number < best_at_target->player_number)) {
                best_at_target = candidate;
            }
        }
        if (best_at_target.has_value()) return best_at_target;
    }
    return std::nullopt;
}

TacticalTarget plan_mark(
    const Position2& ball,
    const Position2& formation,
    int role_id,
    const std::vector<KnownOpponent>& opponents) {
    const Position2 own_goal{-field_geometry::kActualHalfLengthM, 0.0};
    std::vector<const KnownOpponent*> threats;
    for (const auto& opponent : opponents) {
        if (opponent.position_m[0] <= ball[0] + 4.0) threats.push_back(&opponent);
    }
    auto role_cost = [&](const KnownOpponent& opponent, bool left) {
        const bool wrong_side = left
            ? opponent.position_m[1] < 0.0
            : opponent.position_m[1] >= 0.0;
        return math::planar_dist(opponent.position_m, own_goal) +
            (wrong_side ? 4.0 : 0.0);
    };
    const KnownOpponent* selected = nullptr;
    if (threats.size() == 1U) {
        const bool left_owner = threats.front()->position_m[1] >= 0.0;
        if ((role_id == RoleManager::ROLE_CBL) == left_owner) {
            selected = threats.front();
        }
    } else if (threats.size() >= 2U) {
        const KnownOpponent* best_left = nullptr;
        const KnownOpponent* best_right = nullptr;
        double best_pair_cost = std::numeric_limits<double>::infinity();
        for (const auto* left : threats) {
            for (const auto* right : threats) {
                if (left == right) continue;
                const double cost = role_cost(*left, true) + role_cost(*right, false);
                if (cost < best_pair_cost) {
                    best_pair_cost = cost;
                    best_left = left;
                    best_right = right;
                }
            }
        }
        selected = role_id == RoleManager::ROLE_CBL ? best_left : best_right;
    }
    if (selected == nullptr) {
        return {TacticalDuty::Cover, formation, ball, 0, 0.45};
    }
    const Position2 goal_side = math::vec2_unit_or(
        math::vec2_sub(own_goal, selected->position_m), {-1.0, 0.0});
    const Position2 target = clamp_field_player(math::vec2_add(
        selected->position_m, math::vec2_scale(goal_side, 1.3)));
    return {
        TacticalDuty::Mark,
        target,
        ball,
        selected->player_number,
        0.8};
}

TacticalTarget plan_goalkeeper(
    const world::WorldSnapshot& snapshot,
    const Position2& ball,
    const Position2& goalkeeper_position,
    std::optional<double> goalkeeper_yaw_deg = std::nullopt) {
    constexpr double hold_x =
        -field_geometry::kActualHalfLengthM + field_geometry::kGkHoldDepthM;
    if (snapshot.ball.velocity_valid && snapshot.ball.velocity_mps[0] < -0.20) {
        const double time_to_line =
            (hold_x - ball[0]) / snapshot.ball.velocity_mps[0];
        if (time_to_line >= 0.0 && time_to_line <= 4.0) {
            const double crossing_y =
                ball[1] + snapshot.ball.velocity_mps[1] * time_to_line;
            if (std::abs(crossing_y) <= field_geometry::kGoalHalfWidthM + 0.15) {
                const Position2 target = clamp_goalkeeper({hold_x, crossing_y});
                const strategy::ReachTimeModel reach_model(
                    strategy::ReachTimeModel::Parameters{
                        1.10, 120.0, 0.20, 0.15, 0.10});
                if (reach_model.estimate_s(
                        goalkeeper_position, target, goalkeeper_yaw_deg) +
                        0.10 <= time_to_line) {
                    return {
                        TacticalDuty::GoalkeeperIntercept,
                        target,
                        ball,
                        0,
                        std::clamp(1.0 - time_to_line / 5.0, 0.4, 1.0)};
                }
            }
        }
    }
    const double angle_cover_y = std::clamp(
        ball[1] * 0.10,
        -field_geometry::kGoalHalfWidthM + 0.35,
        field_geometry::kGoalHalfWidthM - 0.35);
    return {
        TacticalDuty::GoalkeeperHold,
        clamp_goalkeeper({hold_x, angle_cover_y}),
        ball,
        0,
        snapshot.ball.position_valid ? 0.7 : 0.3};
}

}  // namespace

const TeamTacticalAssignment* TeamPlan::for_player(int player_number) const {
    const auto it = std::find_if(
        assignments.begin(), assignments.end(),
        [player_number](const TeamTacticalAssignment& assignment) {
            return assignment.player_number == player_number;
        });
    return it == assignments.end() ? nullptr : &*it;
}

const TeamTacticalAssignment* TeamPlan::for_role(int role_id) const {
    const auto it = std::find_if(
        assignments.begin(), assignments.end(),
        [role_id](const TeamTacticalAssignment& assignment) {
            return assignment.role_id == role_id;
        });
    return it == assignments.end() ? nullptr : &*it;
}

TeamPlan TeamTactics::plan_all(
    const world::WorldSnapshot& snapshot,
    const std::vector<RoleAssignment>& role_assignments) const {
    TeamPlan result;
    result.assignments.reserve(role_assignments.size());
    for (const auto& assignment : role_assignments) {
        const Position2 formation = assignment.role_id == RoleManager::ROLE_GK
            ? clamp_goalkeeper(assignment.role_position_m)
            : clamp_field_player(assignment.role_position_m);
        result.assignments.push_back({
            assignment.player_number,
            assignment.role_id,
            TacticalTarget{
                TacticalDuty::Formation, formation, std::nullopt, 0, 0.25},
        });
    }
    std::sort(
        result.assignments.begin(), result.assignments.end(),
        [](const TeamTacticalAssignment& lhs,
           const TeamTacticalAssignment& rhs) {
            return lhs.player_number < rhs.player_number;
        });

    const auto clear_support_latches = [this]() {
        for (auto& latch : support_latches_) latch = {};
    };
    if (snapshot.play_mode != world::PlayMode::PlayOn ||
        !snapshot.ball.position_valid ||
        (!snapshot.ball.visible && snapshot.ball.position_age_s > 0.75)) {
        clear_support_latches();
        return result;
    }

    const Position2 ball{snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const auto opponents = known_opponents(snapshot);
    const strategy::TacticalState state = strategy::build_tactical_state(snapshot);
    const auto intercept_owner = state.phase == strategy::TacticalPhase::Attack
        ? std::optional<InterceptCandidate>{}
        : select_intercept_owner(snapshot, role_assignments);
    const Position2 own_goal{-field_geometry::kActualHalfLengthM, 0.0};
    const Position2 goal_direction = math::vec2_unit_or(
        math::vec2_sub(own_goal, ball), {-1.0, 0.0});

    for (auto& assignment : result.assignments) {
        const int role_id = assignment.role_id;
        const Position2 formation = assignment.target.position_m;
        if (role_id == RoleManager::ROLE_GK) {
            const Position2 keeper_position =
                player_position(snapshot, assignment.player_number)
                    .value_or(formation);
            assignment.target = plan_goalkeeper(
                snapshot, ball, keeper_position);
            continue;
        }
        if (role_id == RoleManager::ROLE_AP) {
            if (state.risk_mode == strategy::TacticalRiskMode::ProtectLead &&
                state.phase != strategy::TacticalPhase::Attack) {
                assignment.target = {
                    TacticalDuty::Cover,
                    clamp_field_player(math::vec2_add(
                        ball, math::vec2_scale(goal_direction, 2.8))),
                    ball,
                    0,
                    0.85};
            } else {
                assignment.target = {
                    TacticalDuty::Pressure, ball, ball, 0, 1.0};
            }
            continue;
        }

        if (state.phase == strategy::TacticalPhase::Attack) {
            if (role_id == RoleManager::ROLE_ST ||
                role_id == RoleManager::ROLE_CBM) {
                TacticalTarget support = plan_support(
                    ball, formation, role_id, assignment.player_number,
                    opponents, snapshot, state.risk_mode);
                const auto index = static_cast<std::size_t>(
                    assignment.player_number);
                if (index < support_latches_.size()) {
                    auto& latch = support_latches_[index];
                    if (latch.target.has_value() &&
                        latch.role_id == role_id &&
                        snapshot.server_time < latch.until_s &&
                        support.duty != TacticalDuty::Formation) {
                        TacticalTarget held = *latch.target;
                        held.duty = support.duty;
                        held.face_point_m = ball;
                        assignment.target = held;
                        continue;
                    }
                    if (support.duty == TacticalDuty::Formation) {
                        latch = {};
                    } else {
                        latch.target = support;
                        latch.until_s = snapshot.server_time + 0.5;
                        latch.role_id = role_id;
                    }
                }
                assignment.target = support;
                continue;
            }
            if (role_id == RoleManager::ROLE_CDM) {
                assignment.target = {
                    TacticalDuty::Cover,
                    clamp_field_player(math::vec2_add(
                        ball, math::vec2_scale(goal_direction, 6.0))),
                    ball,
                    0,
                    0.75};
                continue;
            }
            assignment.target = {
                TacticalDuty::Cover, formation, ball, 0, 0.6};
            continue;
        }

        if (intercept_owner.has_value() &&
            assignment.player_number == intercept_owner->player_number) {
            assignment.target = {
                TacticalDuty::Intercept,
                intercept_owner->position_m,
                ball,
                0,
                std::clamp(
                    1.0 - intercept_owner->ball_arrival_s / 4.0,
                    0.35,
                    0.9)};
            continue;
        }
        if (role_id == RoleManager::ROLE_CBL ||
            role_id == RoleManager::ROLE_CBR) {
            assignment.target = plan_mark(
                ball, formation, role_id, opponents);
            continue;
        }
        if (role_id == RoleManager::ROLE_CDM) {
            assignment.target = {
                TacticalDuty::Cover,
                clamp_field_player(math::vec2_add(
                    ball, math::vec2_scale(goal_direction, 4.0))),
                ball,
                0,
                0.8};
            continue;
        }
        if (role_id == RoleManager::ROLE_CBM) {
            assignment.target = {
                TacticalDuty::BlockLane,
                clamp_field_player(math::vec2_add(
                    ball, math::vec2_scale(goal_direction, 2.5))),
                ball,
                0,
                0.7};
            continue;
        }
        assignment.target = {
            TacticalDuty::Outlet, formation, ball, 0, 0.55};
    }

    if (state.phase != strategy::TacticalPhase::Attack) {
        clear_support_latches();
    }
    return result;
}

TacticalTarget TeamTactics::plan(
    const world::WorldSnapshot& snapshot,
    int role_id,
    const Position2& formation_target_m) const {
    const TeamPlan result = plan_all(
        snapshot,
        {RoleAssignment{
            snapshot.player_number, role_id, formation_target_m}});
    const auto* assignment = result.for_player(snapshot.player_number);
    return assignment != nullptr
        ? assignment->target
        : TacticalTarget{
              TacticalDuty::Formation,
              role_id == RoleManager::ROLE_GK
                  ? clamp_goalkeeper(formation_target_m)
                  : clamp_field_player(formation_target_m),
              std::nullopt,
              0,
              0.0};
}

std::string_view to_string(TacticalDuty duty) {
    switch (duty) {
        case TacticalDuty::Formation: return "Formation";
        case TacticalDuty::Support: return "Support";
        case TacticalDuty::Unmark: return "Unmark";
        case TacticalDuty::Outlet: return "Outlet";
        case TacticalDuty::Pressure: return "Pressure";
        case TacticalDuty::Cover: return "Cover";
        case TacticalDuty::Mark: return "Mark";
        case TacticalDuty::BlockLane: return "BlockLane";
        case TacticalDuty::Intercept: return "Intercept";
        case TacticalDuty::GoalkeeperHold: return "GoalkeeperHold";
        case TacticalDuty::GoalkeeperIntercept: return "GoalkeeperIntercept";
        case TacticalDuty::Receive: return "Receive";
    }
    return "Formation";
}

}  // namespace decision
