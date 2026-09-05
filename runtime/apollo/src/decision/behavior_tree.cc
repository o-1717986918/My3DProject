// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/behavior_tree.h"

#include "src/decision/behavior_nodes.h"
#include "src/decision/field_geometry.h"
#include "src/decision/role_behaviors.h"
#include "src/math/math_utils.h"
#include "src/strategy/tactical_state.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace decision {

namespace {

struct BehaviorContext {
    const world::WorldSnapshot& snapshot;
    Blackboard& blackboard;
    RoleManager& role_manager;
    TeamTactics& team_tactics;
    RestartCoordinator& restart_coordinator;
    RoleBehaviorSet& role_behaviors;
    world::PlayMode& previous_play_mode;
    double& kickoff_hold_until_s;
    bool enable_pass_strategy;
    bool enable_targeted_kick;
    bool enable_team_tactics;
    const std::optional<ExecutionFeedback>& execution_feedback;
};

using NodeResult = bt::NodeResult<BehaviorContext>;
using NodePtr = bt::NodePtr<BehaviorContext>;

bool is_fallen(const world::WorldSnapshot& snapshot) {
    return snapshot.self.position_m[2] < world::kFallenHeightThresholdM;
}

bool has_teammates(const BehaviorContext& context) {
    return !context.snapshot.teammates.empty();
}

bool needs_beam(const BehaviorContext& context) {
    return (context.snapshot.play_mode_group == world::PlayModeGroup::ActiveBeam ||
            context.snapshot.play_mode_group == world::PlayModeGroup::PassiveBeam) &&
           !context.snapshot.has_beamed;
}

bool is_set_play_mode(const BehaviorContext& context) {
    return context.snapshot.play_mode_group == world::PlayModeGroup::OurKick ||
           context.snapshot.play_mode_group == world::PlayModeGroup::TheirKick;
}

bool is_game_over(const BehaviorContext& context) {
    return context.snapshot.play_mode == world::PlayMode::GameOver;
}

// Humanoid robots walk forward only, so while farther than kFaceTargetRadiusM
// from the destination they must face the direction of travel; once on station
// they face a fixed point (the ball, or the goal to pre-aim a kick). Returns
// std::nullopt when the facing target is degenerate (coincides with the
// reference point), leaving the orientation unset.
constexpr double kFaceTargetRadiusM = 0.5;

std::optional<double> walk_facing_orientation_deg(
    const std::array<double, 2>& self,
    const std::array<double, 2>& destination,
    const std::array<double, 2>& face_when_close) {
    const bool far = math::planar_dist(self, destination) > kFaceTargetRadiusM;
    const std::array<double, 2> from = far ? self : destination;
    const std::array<double, 2> to = far ? destination : face_when_close;
    if (math::sq_dist2(to, from) < 1e-6) {
        return std::nullopt;
    }
    return math::vector_angle_deg(math::vec2_sub(to, from));
}

WalkCommand walk_to_current_role_position(BehaviorContext& context) {
    const int role_id = current_role_from_blackboard(context.blackboard);
    const auto role_pos = context.blackboard.exists(Blackboard::kKeyRolePos)
        ? context.blackboard.get<std::array<double, 2>>(Blackboard::kKeyRolePos)
        : std::array<double, 2>{0.0, 0.0};
    const std::array<double, 2> ball_position{
        context.snapshot.ball.position_m[0],
        context.snapshot.ball.position_m[1],
    };
    const std::array<double, 2> self_position{
        context.snapshot.self.position_m[0],
        context.snapshot.self.position_m[1],
    };
    const auto legal_role_pos =
        field_geometry::legalize_set_play_target(role_pos, ball_position, context.snapshot.play_mode);
    context.blackboard.set(Blackboard::kKeyRolePos, legal_role_pos);

    WalkCommand command;
    command.target_2d_m = legal_role_pos;
    command.target_absolute = true;
    command.role_id = role_id;

    command.orientation_deg =
        walk_facing_orientation_deg(self_position, legal_role_pos, ball_position);
    command.orientation_absolute = true;
    return command;
}

HighLevelCommand make_beam_command(BehaviorContext& context) {
    context.role_behaviors.reset();
    context.restart_coordinator.reset();
    context.team_tactics.reset();
    // The role manager has not yet run at beam time, so we cannot map this
    // player to a formation slot. Use the player-number pose table, with a
    // deeper front-player pose only when the next kickoff belongs to the other
    // team and the server will enforce center-circle clearance immediately.
    const auto pose =
        context.snapshot.play_mode_group == world::PlayModeGroup::PassiveBeam
            ? field_geometry::player_defensive_kickoff_beam_pose(context.snapshot.player_number)
            : field_geometry::player_beam_pose(context.snapshot.player_number);
    return BeamCommand{pose[0], pose[1], pose[2]};
}

HighLevelCommand make_get_up_command(BehaviorContext& context) {
    context.role_behaviors.reset();
    return GetUpCommand{};
}

std::optional<std::array<double, 2>> observed_player_position(
    const world::WorldSnapshot& snapshot,
    int player_number) {
    if (player_number == snapshot.player_number) {
        if (is_fallen(snapshot)) return std::nullopt;
        return std::array<double, 2>{
            snapshot.self.position_m[0], snapshot.self.position_m[1]};
    }
    for (const auto& teammate : snapshot.teammates) {
        const bool fresh = teammate.seen ||
            (teammate.last_seen_time >= 0.0 &&
             snapshot.server_time - teammate.last_seen_time <= 2.0);
        if (teammate.player_number == player_number && fresh &&
            !teammate.fallen) {
            return std::array<double, 2>{
                teammate.position_m[0], teammate.position_m[1]};
        }
    }
    return std::nullopt;
}

bool restart_team_positioned(
    const world::WorldSnapshot& snapshot,
    const std::vector<RoleAssignment>& assignments,
    const std::optional<RestartPlan>& plan) {
    if (!plan.has_value()) return false;
    if (!plan->requires_receiver_ready) {
        const auto taker = observed_player_position(
            snapshot, plan->taker_player_number);
        return taker.has_value() &&
            math::planar_dist(*taker, plan->ball_anchor_m) <= 1.25;
    }
    for (const auto& assignment : assignments) {
        const auto position = observed_player_position(
            snapshot, assignment.player_number);
        if (!position.has_value()) return false;

        std::array<double, 2> target = assignment.role_position_m;
        double tolerance_m = 1.5;
        if (assignment.player_number == plan->receiver_player_number) {
            target = plan->receiver_target_m;
            tolerance_m = 0.9;
        } else if (assignment.player_number == plan->taker_player_number) {
            target = plan->ball_anchor_m;
            tolerance_m = 1.25;
        }
        if (math::planar_dist(*position, target) > tolerance_m) return false;
    }
    return true;
}

bool restart_receiver_ready(
    const world::WorldSnapshot& snapshot,
    const std::optional<RestartPlan>& plan) {
    if (!plan.has_value() || !plan->requires_receiver_ready ||
        plan->receiver_player_number <= 0) {
        return plan.has_value() && !plan->requires_receiver_ready;
    }
    const auto receiver = observed_player_position(
        snapshot, plan->receiver_player_number);
    if (!receiver.has_value() ||
        math::planar_dist(*receiver, plan->receiver_target_m) > 0.75) {
        return false;
    }
    if (plan->receiver_player_number != snapshot.player_number) return true;

    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double face_error_deg = std::abs(math::normalize_deg(
        math::vector_angle_deg(math::vec2_sub(ball, *receiver)) - yaw_deg));
    const double speed_mps = math::norm2({
        snapshot.self.lin_vel_b[0], snapshot.self.lin_vel_b[1]});
    return face_error_deg <= 25.0 && speed_mps <= 0.35;
}

bool restart_taker_aligned(
    const world::WorldSnapshot& snapshot,
    const std::optional<RestartPlan>& plan) {
    if (!plan.has_value() ||
        plan->taker_player_number != snapshot.player_number ||
        !snapshot.ball.position_valid) {
        return false;
    }
    constexpr double contact_behind_m = 0.33;
    constexpr double longitudinal_tolerance_m = 0.03;
    constexpr double lateral_tolerance_m = 0.03;
    constexpr double orientation_tolerance_deg = 3.0;
    constexpr double maximum_speed_mps = 0.20;
    const double direction_rad = math::deg_to_rad(plan->contact_direction_deg);
    const std::array<double, 2> direction{
        std::cos(direction_rad), std::sin(direction_rad)};
    const std::array<double, 2> lateral{-direction[1], direction[0]};
    const std::array<double, 2> self_from_ball{
        snapshot.self.position_m[0] - snapshot.ball.position_m[0],
        snapshot.self.position_m[1] - snapshot.ball.position_m[1]};
    const double behind_m = -(
        self_from_ball[0] * direction[0] +
        self_from_ball[1] * direction[1]);
    const double lateral_m =
        self_from_ball[0] * lateral[0] + self_from_ball[1] * lateral[1];
    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const double speed_mps = math::norm2({
        snapshot.self.lin_vel_b[0], snapshot.self.lin_vel_b[1]});
    return std::abs(behind_m - contact_behind_m) <= longitudinal_tolerance_m &&
        std::abs(lateral_m) <= lateral_tolerance_m &&
        std::abs(math::normalize_deg(plan->contact_direction_deg - yaw_deg)) <=
            orientation_tolerance_deg &&
        speed_mps <= maximum_speed_mps;
}

bool another_player_controls_released_ball(
    const world::WorldSnapshot& snapshot,
    const std::optional<RestartPlan>& plan) {
    if (!plan.has_value() || !snapshot.ball.position_valid ||
        math::planar_dist(
            {snapshot.ball.position_m[0], snapshot.ball.position_m[1]},
            plan->ball_anchor_m) < 0.35) {
        return false;
    }
    const std::array<double, 2> ball{
        snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
    for (const auto& teammate : snapshot.teammates) {
        if (teammate.player_number == plan->taker_player_number ||
            teammate.player_number <= 0 || teammate.fallen) {
            continue;
        }
        const bool fresh = teammate.seen ||
            (teammate.last_seen_time >= 0.0 &&
             snapshot.server_time - teammate.last_seen_time <= 0.5);
        if (fresh && math::planar_dist(
                {teammate.position_m[0], teammate.position_m[1]}, ball) <=
                0.55) {
            return true;
        }
    }
    return false;
}

std::optional<RestartExecutionFeedback> restart_execution_feedback(
    const std::optional<ExecutionFeedback>& feedback) {
    if (!feedback.has_value() ||
        feedback->request_kind != MotionRequestKind::Kick ||
        !feedback->restart_epoch.has_value() ||
        !feedback->restart_revision.has_value()) {
        return std::nullopt;
    }
    RestartExecutionStatus status = RestartExecutionStatus::Running;
    switch (feedback->status) {
        case ExecutionStatus::Running:
            status = RestartExecutionStatus::Running;
            break;
        case ExecutionStatus::Completed:
            status = RestartExecutionStatus::Completed;
            break;
        case ExecutionStatus::Rejected:
            status = RestartExecutionStatus::Rejected;
            break;
        case ExecutionStatus::TimedOut:
            status = RestartExecutionStatus::TimedOut;
            break;
    }
    return RestartExecutionFeedback{
        *feedback->restart_epoch, *feedback->restart_revision, status};
}

NodeResult compute_formation(BehaviorContext& context) {
    const std::array<double, 2> ball_position{
        context.snapshot.ball.position_m[0],
        context.snapshot.ball.position_m[1],
    };
    auto role_assignments = context.role_manager.assign(context.snapshot);
    for (auto& assignment : role_assignments) {
        assignment.role_position_m = field_geometry::legalize_set_play_target(
            assignment.role_position_m,
            ball_position,
            context.snapshot.play_mode);
    }

    RestartCoordinatorInput restart_input;
    restart_input.play_mode = context.snapshot.play_mode;
    restart_input.server_time_s = context.snapshot.server_time;
    restart_input.self_player_number = context.snapshot.player_number;
    restart_input.ball_position_m = ball_position;
    restart_input.ball_velocity_mps = {
        context.snapshot.ball.velocity_mps[0],
        context.snapshot.ball.velocity_mps[1]};
    restart_input.ball_position_valid = context.snapshot.ball.position_valid;
    restart_input.ball_velocity_valid = context.snapshot.ball.velocity_valid;
    restart_input.role_assignments = role_assignments;
    const auto append_restart_opponent = [&](const world::PlayerObservation& opponent) {
        const bool fresh = opponent.seen ||
            (opponent.last_seen_time >= 0.0 &&
             context.snapshot.server_time - opponent.last_seen_time <= 2.0);
        if (!fresh || opponent.fallen) return;
        const std::array<double, 2> position{
            opponent.position_m[0], opponent.position_m[1]};
        const bool duplicate = std::any_of(
            restart_input.opponent_positions_m.begin(),
            restart_input.opponent_positions_m.end(),
            [&](const std::array<double, 2>& existing) {
                return math::planar_dist(existing, position) < 0.5;
            });
        if (!duplicate) restart_input.opponent_positions_m.push_back(position);
    };
    for (const auto& opponent : context.snapshot.opponents) {
        append_restart_opponent(opponent);
    }
    for (const auto& opponent : context.snapshot.shared_opponents) {
        append_restart_opponent(opponent);
    }
    std::sort(
        restart_input.opponent_positions_m.begin(),
        restart_input.opponent_positions_m.end());
    restart_input.team_positioned = restart_team_positioned(
        context.snapshot, role_assignments,
        context.restart_coordinator.plan());
    restart_input.receiver_ready = restart_receiver_ready(
        context.snapshot, context.restart_coordinator.plan());
    restart_input.taker_aligned = restart_taker_aligned(
        context.snapshot, context.restart_coordinator.plan());
    restart_input.another_player_touched_ball =
        another_player_controls_released_ball(
            context.snapshot, context.restart_coordinator.plan());
    restart_input.execution_feedback = restart_execution_feedback(
        context.execution_feedback);
    const RestartCoordinationDecision restart_decision =
        context.restart_coordinator.update(restart_input);

    const bool restart_active = restart_decision.plan.has_value() &&
        restart_decision.phase != RestartPhase::Idle &&
        restart_decision.phase != RestartPhase::Complete;
    if (restart_active) {
        for (auto& assignment : role_assignments) {
            if (assignment.player_number ==
                restart_decision.plan->receiver_player_number) {
                assignment.role_position_m =
                    restart_decision.plan->receiver_target_m;
            }
        }
    }

    TeamPlan team_plan;
    if (context.enable_team_tactics) {
        team_plan = context.team_tactics.plan_all(
            context.snapshot, role_assignments);
    } else {
        // A controlled ablation keeps the upstream role/formation and all
        // restart legality, while removing only the new open-play duty layer.
        // This makes current-vs-base evidence attributable instead of forcing
        // an all-or-nothing binary comparison.
        team_plan.tactical_state =
            strategy::build_tactical_state(context.snapshot);
        team_plan.source_server_time_s = context.snapshot.server_time;
        team_plan.fresh = context.snapshot.ball.position_valid &&
            (context.snapshot.ball.visible ||
             context.snapshot.ball.position_age_s <= 0.75);
        for (const auto& role_assignment : role_assignments) {
            team_plan.assignments.push_back({
                role_assignment.player_number,
                role_assignment.role_id,
                TacticalTarget{
                    TacticalDuty::Formation,
                    role_assignment.role_position_m,
                    ball_position,
                    0,
                    0.25},
            });
        }
    }
    if (restart_active) {
        const auto& restart_plan = *restart_decision.plan;
        for (auto& assignment : team_plan.assignments) {
            if (assignment.player_number == restart_plan.receiver_player_number) {
                assignment.target = {
                    TacticalDuty::Receive,
                    restart_plan.receiver_target_m,
                    restart_plan.ball_anchor_m,
                    0,
                    0.8};
            } else if (assignment.player_number ==
                       restart_plan.taker_player_number) {
                assignment.target = {
                    TacticalDuty::Pressure,
                    restart_plan.ball_anchor_m,
                    restart_plan.ball_anchor_m,
                    0,
                    0.9};
            }
        }
    }
    const auto* own_assignment = team_plan.for_player(
        context.snapshot.player_number);
    if (own_assignment == nullptr) return NodeResult::failure();

    const auto role_it = std::find_if(
        role_assignments.begin(), role_assignments.end(),
        [&](const RoleAssignment& assignment) {
            return assignment.player_number == context.snapshot.player_number;
        });
    if (role_it == role_assignments.end()) return NodeResult::failure();

    std::array<double, 2> own_role_position = role_it->role_position_m;
    TacticalTarget own_target = own_assignment->target;
    if (restart_decision.self_locked_out &&
        restart_decision.plan.has_value()) {
        const double direction_rad = math::deg_to_rad(
            restart_decision.plan->contact_direction_deg);
        own_role_position = {
            restart_decision.plan->ball_anchor_m[0] -
                2.5 * std::cos(direction_rad),
            restart_decision.plan->ball_anchor_m[1] -
                2.5 * std::sin(direction_rad)};
        own_role_position[0] = std::clamp(
            own_role_position[0],
            -field_geometry::kActualHalfLengthM +
                field_geometry::kFormationFieldMarginM,
            field_geometry::kActualHalfLengthM -
                field_geometry::kFormationFieldMarginM);
        own_role_position[1] = std::clamp(
            own_role_position[1],
            -field_geometry::kActualHalfWidthM +
                field_geometry::kFormationFieldMarginM,
            field_geometry::kActualHalfWidthM -
                field_geometry::kFormationFieldMarginM);
        own_target = {
            TacticalDuty::Outlet,
            own_role_position,
            ball_position,
            0,
            0.9};
    }

    context.blackboard.set(
        Blackboard::kKeyCurrentRole, own_assignment->role_id);
    context.blackboard.set(
        Blackboard::kKeyRolePos, own_role_position);
    context.blackboard.set(
        Blackboard::kKeyTacticalTarget, own_target);
    context.blackboard.set(
        Blackboard::kKeyTacticalRiskMode,
        team_plan.tactical_state.risk_mode);
    context.blackboard.set(Blackboard::kKeyTeamPlan, team_plan);
    context.blackboard.set(
        Blackboard::kKeyRoleAssignments, role_assignments);
    context.blackboard.set(
        Blackboard::kKeyRestartDecision, restart_decision);
    return NodeResult::success_only();
}

// Deep-pair hold after an opponent kickoff. Players 6 and 7 beam to a back line
// on their kickoff (player_defensive_kickoff_beam_pose) and must keep it for the
// first kKickoffHoldAfterPlayOnS of open play rather than releasing up the pitch
// the instant the ball is in play. The blackboard is cleared every tick, so the
// TheirKickOff -> PlayOn transition is latched in persistent state here (one
// robot per process, mirroring the existing file-scope role state).
constexpr double kKickoffHoldAfterPlayOnS = 3.0;

void update_kickoff_hold_state(BehaviorContext& context) {
    if (context.previous_play_mode == world::PlayMode::TheirKickOff &&
        context.snapshot.play_mode == world::PlayMode::PlayOn) {
        context.kickoff_hold_until_s =
            context.snapshot.server_time + kKickoffHoldAfterPlayOnS;
    }
    context.previous_play_mode = context.snapshot.play_mode;
}

bool is_kickoff_hold_active(const BehaviorContext& context) {
    const int player_number = context.snapshot.player_number;
    return (player_number == 6 || player_number == 7) &&
           context.snapshot.play_mode == world::PlayMode::PlayOn &&
           context.snapshot.server_time < context.kickoff_hold_until_s;
}

bool is_restart_taker_lockout(const BehaviorContext& context) {
    return context.blackboard.exists(Blackboard::kKeyRestartDecision) &&
        context.blackboard.get<RestartCoordinationDecision>(
            Blackboard::kKeyRestartDecision).self_locked_out;
}

// During BeforeKickOff the server waits for the kicker to confirm before the
// ball is live. Any teammate who is already inside the center circle (the
// kicker at the center spot, the relay at the perpendicular slot) must hold
// position — otherwise the formation would walk them out before the kickoff
// actually starts.
bool is_before_kickoff_hold(const BehaviorContext& context) {
    if (context.snapshot.play_mode != world::PlayMode::BeforeKickOff) {
        return false;
    }
    const std::array<double, 2> ball{
        context.snapshot.ball.position_m[0],
        context.snapshot.ball.position_m[1],
    };
    const std::array<double, 2> self{
        context.snapshot.self.position_m[0],
        context.snapshot.self.position_m[1],
    };
    return math::planar_dist(self, ball) < field_geometry::kCenterCircleRadiusM;
}

HighLevelCommand make_walk_in_place_command(BehaviorContext& context) {
    WalkCommand walk_cmd;
    walk_cmd.target_2d_m = {
        context.snapshot.self.position_m[0],
        context.snapshot.self.position_m[1],
    };
    walk_cmd.target_absolute = true;
    walk_cmd.role_id = current_role_from_blackboard(context.blackboard);
    return walk_cmd;
}

HighLevelCommand make_kickoff_hold_command(BehaviorContext& context) {
    // The two baseline defenders hold their line for the opening seconds of
    // open play (after TheirKickOff) so they don't release up the pitch the
    // instant the ball is live. role_id is read from the blackboard that
    // compute_formation populated one task earlier in the sequence.
    return make_walk_in_place_command(context);
}

HighLevelCommand make_before_kickoff_hold_command(BehaviorContext& context) {
    // Step in place at the beam pose so the kicker/relay stay ready inside the
    // center circle until the server flips to OurKickOff. Moving the kicker off
    // the center spot during BeforeKickOff would not just waste motion — the
    // server's legality check may relocate them mid-wait.
    return make_walk_in_place_command(context);
}

HighLevelCommand make_set_play_command(BehaviorContext& context) {
    compute_formation(context);
    if (context.snapshot.play_mode_group == world::PlayModeGroup::TheirKick) {
        context.role_behaviors.reset();
        return walk_to_current_role_position(context);
    }

    const int role_id = current_role_from_blackboard(context.blackboard);

    if (role_id == RoleManager::ROLE_AP &&
        context.snapshot.play_mode != world::PlayMode::OurGoalKick) {
        // OurGoalKick is excluded because the GK is the designated taker there
        // and an approaching AP would crowd the keeper's clearance.
        auto selected = context.role_behaviors.select(
            context.snapshot, context.blackboard, context.role_manager,
            false, context.enable_targeted_kick);
        if (selected.has_value()) {
            return *selected;
        }
    }

    if (role_id == RoleManager::ROLE_GK) {
        auto selected = context.role_behaviors.select(
            context.snapshot, context.blackboard, context.role_manager,
            false, context.enable_targeted_kick);
        if (selected.has_value()) {
            return *selected;
        }
    }

    if (context.blackboard.exists(Blackboard::kKeyRestartDecision) &&
        context.blackboard.get<RestartCoordinationDecision>(
            Blackboard::kKeyRestartDecision).self_is_receiver) {
        auto selected = context.role_behaviors.select(
            context.snapshot, context.blackboard, context.role_manager,
            false, context.enable_targeted_kick);
        if (selected.has_value()) return *selected;
    }

    return walk_to_current_role_position(context);
}

NodeResult make_role_behavior_command(BehaviorContext& context) {
    const int role_id = current_role_from_blackboard(context.blackboard);
    auto selected = context.role_behaviors.select(
        context.snapshot, context.blackboard, context.role_manager,
        context.enable_pass_strategy, context.enable_targeted_kick);
    if (!selected.has_value()) {
        return NodeResult::failure();
    }

    if (auto* walk = std::get_if<WalkCommand>(&*selected); walk != nullptr) {
        walk->role_id = role_id;
    }
    return NodeResult::with_command(*selected);
}

HighLevelCommand make_neutral_command(BehaviorContext&) {
    return NeutralCommand{};
}

NodePtr make_top_level_tree() {
    return bt::fallback<BehaviorContext>({
        bt::sequence<BehaviorContext>({
            bt::condition<BehaviorContext>(has_teammates),
            bt::fallback<BehaviorContext>({
                bt::sequence<BehaviorContext>({
                    bt::condition<BehaviorContext>(is_game_over),
                    bt::command<BehaviorContext>(make_neutral_command),
                }),
                bt::sequence<BehaviorContext>({
                    bt::condition<BehaviorContext>(needs_beam),
                    bt::command<BehaviorContext>(make_beam_command),
                }),
                bt::sequence<BehaviorContext>({
                    bt::condition<BehaviorContext>([](const BehaviorContext& context) { return is_fallen(context.snapshot); }),
                    bt::command<BehaviorContext>(make_get_up_command),
                }),
                bt::sequence<BehaviorContext>({
                    bt::condition<BehaviorContext>(is_set_play_mode),
                    bt::command<BehaviorContext>(make_set_play_command),
                }),
                bt::sequence<BehaviorContext>({
                    bt::condition<BehaviorContext>(is_before_kickoff_hold),
                    bt::task<BehaviorContext>(compute_formation),
                    bt::command<BehaviorContext>(make_before_kickoff_hold_command),
                }),
                bt::sequence<BehaviorContext>({
                    bt::condition<BehaviorContext>(is_kickoff_hold_active),
                    bt::task<BehaviorContext>(compute_formation),
                    bt::command<BehaviorContext>(make_kickoff_hold_command),
                }),
                bt::sequence<BehaviorContext>({
                    bt::task<BehaviorContext>(compute_formation),
                    bt::fallback<BehaviorContext>({
                        bt::sequence<BehaviorContext>({
                            bt::condition<BehaviorContext>(
                                is_restart_taker_lockout),
                            bt::command<BehaviorContext>(
                                walk_to_current_role_position),
                        }),
                        bt::task<BehaviorContext>(make_role_behavior_command),
                    }),
                }),
            }),
        }),
        bt::command<BehaviorContext>(make_neutral_command),
    });
}

}  // namespace

HighLevelCommand BehaviorTree::evaluate(
    const world::WorldSnapshot& snapshot,
    Blackboard& blackboard,
    RoleManager& role_manager,
    bool enable_pass_strategy,
    bool enable_targeted_kick,
    const std::optional<ExecutionFeedback>& execution_feedback,
    bool enable_team_tactics) const {
    blackboard.clear();
    BehaviorContext context{
        snapshot,
        blackboard,
        role_manager,
        team_tactics_,
        restart_coordinator_,
        role_behaviors_,
        previous_play_mode_,
        kickoff_hold_until_s_,
        enable_pass_strategy,
        enable_targeted_kick,
        enable_team_tactics,
        execution_feedback};
    if (execution_feedback.has_value()) {
        role_behaviors_.apply_execution_feedback(*execution_feedback);
    }
    update_kickoff_hold_state(context);
    static const NodePtr top_level_tree = make_top_level_tree();
    const auto result = top_level_tree->tick(context);
    return result.command.value_or(NeutralCommand{});
}

}  // namespace decision
