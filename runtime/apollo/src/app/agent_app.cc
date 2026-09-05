// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/app/agent_app.h"

#include "src/comm/team_comm_codec.h"
#include "src/decision/role_behaviors.h"
#include "src/decision/team_tactics.h"
#include "src/math/math_utils.h"
#include "src/server/action_encoder.h"
#include "src/strategy/tactical_state.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace app {

namespace {

bool infer_is_left_team(const server::PerceptionFrame& frame, const std::string& team_name, const world::WorldState& world_state) {
    if (frame.game_state.has_value()) {
        const auto& gs = *frame.game_state;
        if (gs.team_left.has_value() && gs.team_left.value() == team_name) {
            return true;
        }
        if (gs.team_right.has_value() && gs.team_right.value() == team_name) {
            return false;
        }
    }
    return world_state.snapshot().is_left_team.value_or(true);
}

void normalize_frame_for_team(server::PerceptionFrame& frame, bool is_left_team) {
    if (is_left_team) {
        return;
    }
    if (frame.position.has_value()) {
        frame.position->xyz_m = world::FrameNormalizer::normalize_position(frame.position->xyz_m, false);
    }
    if (frame.orientation.has_value()) {
        frame.orientation->wxyz = world::FrameNormalizer::normalize_quaternion_wxyz(frame.orientation->wxyz, false);
    }
}

bool pass_ready(
    const world::WorldSnapshot& snapshot,
    const strategy::CooperativeAction& action) {
    for (const auto& intent : snapshot.team_comm_snapshot.pass_intents) {
        if (intent.state == comm::PassIntentState::Ready &&
            intent.sender_player_number == action.target_player_number &&
            intent.passer_player_number == snapshot.player_number &&
            intent.receiver_player_number == action.target_player_number &&
            intent.sequence_id == action.sequence_id) {
            return true;
        }
    }
    return false;
}

std::string_view kick_mode_name(const decision::HighLevelCommand& command) {
    const auto* kick = std::get_if<decision::KickCommand>(&command);
    if (kick == nullptr) return "None";
    switch (kick->mode) {
        case decision::KickMode::ForwardContact: return "ForwardContact";
        case decision::KickMode::DribbleTouch: return "DribbleTouch";
        case decision::KickMode::TargetedPass: return "TargetedPass";
        case decision::KickMode::Shot: return "Shot";
        case decision::KickMode::Clear: return "Clear";
    }
    return "None";
}

decision::ExecutionFeedback make_execution_feedback(
    std::uint64_t request_id,
    double server_time,
    const behavior::MotionStepResult& result,
    const decision::HighLevelCommand& command) {
    decision::ExecutionFeedback feedback;
    feedback.request_id = request_id;
    feedback.server_time = server_time;
    feedback.status = result.status;
    feedback.request_kind = result.request_kind;

    if (result.request_kind == decision::MotionRequestKind::Kick) {
        const auto* kick = std::get_if<decision::KickCommand>(&command);
        if (kick != nullptr) {
            feedback.restart_epoch = kick->restart_epoch;
            feedback.restart_revision = kick->restart_revision;
        }
        // Preserve action identity for every planned ball action, not only a
        // pass. Otherwise a rejected dribble/shot/clear remains latched until
        // its nominal duration and repeats the same rejected request.
        if (kick != nullptr && kick->action_id != 0U) {
            feedback.cooperative_action_id = kick->action_id;
            feedback.sequence_id = kick->sequence_id;
        }
    }
    return feedback;
}

}  // namespace

AgentApp::AgentApp(RuntimeConfig config)
    : config_(std::move(config)),
      world_state_(config_.team_name, config_.player_number, 7),
      decision_manager_(
          config_.enable_pass_strategy,
          config_.enable_parameterized_kick,
          config_.enable_team_tactics),
      motion_manager_(config_),
      team_comm_manager_(config_.team_name) {}

AgentApp::AgentApp(RuntimeConfig config, std::unique_ptr<server::TcpLpmClient> client)
    : config_(std::move(config)),
      world_state_(config_.team_name, config_.player_number, 7),
      decision_manager_(
          config_.enable_pass_strategy,
          config_.enable_parameterized_kick,
          config_.enable_team_tactics),
      motion_manager_(config_),
      team_comm_manager_(config_.team_name),
      client_(std::move(client)) {}

int AgentApp::run() {
    return run_for_cycles(std::numeric_limits<std::size_t>::max());
}

int AgentApp::run_for_cycles(std::size_t cycles) {
    if (cycles == 0U) {
        return 0;
    }
    if (!client_) {
        client_ = std::make_unique<server::TcpLpmClient>(config_.host, config_.port);
    }
    client_->connect();
    if (!init_sent_) {
        client_->send_message(server::ActionEncoder::make_init(config_.team_name, config_.player_number));
        init_sent_ = true;
    }

    std::size_t processed = 0;
    while (!shutdown_requested_ && processed < cycles) {
        const std::string message = client_->receive_message();
        const std::string response = process_perception_message(message);
        client_->send_message(response);
        ++processed;
    }
    return 0;
}

std::string AgentApp::process_perception_message(const std::string& message) {
    server::PerceptionFrame frame = perception_parser_.parse(message);
    const bool is_left_team = infer_is_left_team(frame, config_.team_name, world_state_);
    normalize_frame_for_team(frame, is_left_team);
    world_state_.update_from_perception(frame, robot_model_, is_left_team);

    for (const auto& mic : frame.microphone_messages) {
        try {
            const auto payload = server::decode_base64(mic.message_b64);
            if (payload.size() == comm::TeamCommCodec::kPacketSizeBytes) {
                team_comm_manager_.ingest(comm::TeamCommCodec::decode(payload), frame.server_cycle);
            }
        } catch (const std::exception&) {
        }
    }
    world_state_.set_team_comm_snapshot(team_comm_manager_.make_snapshot(frame.server_cycle));

    const world::WorldSnapshot& snapshot = world_state_.snapshot();
    const auto decision_started_at = std::chrono::steady_clock::now();
    const auto command = decision_manager_.decide(
        snapshot, pending_execution_feedback_);
    last_decision_latency_us_ = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - decision_started_at).count());
    pending_execution_feedback_.reset();
    const auto* kick_command = std::get_if<decision::KickCommand>(&command);
    const auto* selected_action =
        decision_manager_.selected_cooperative_action();
    const auto* strategy_plan = decision_manager_.strategy_plan();
    const bool reset = last_command_variant_index_ != command.index();
    last_command_variant_index_ = command.index();

    std::vector<std::string> nodes;
    if (const auto* beam = std::get_if<decision::BeamCommand>(&command)) {
        // Beam is the only absolute-coordinate output. The agent works in a
        // canonical frame (own goal at -x) obtained by a 180-degree rotation of
        // the global frame for the right team, but the server's beam contract
        // maps the right team by REFLECTION (x = abs(x) * side, y unchanged,
        // theta + pi). Those agree on x and yaw but disagree on y, so the right
        // team's canonical beam y must be negated to land where intended.
        const double beam_y = is_left_team ? beam->y_m : -beam->y_m;
        nodes.push_back(server::ActionEncoder::encode_beam(beam->x_m, beam_y, beam->yaw_deg));
        world_state_.set_has_beamed(true);
        last_active_motion_ = "Beam";
        last_execution_status_ = behavior::SkillExecutionStatus::Completed;
    } else {
        const auto motion_result = motion_manager_.step(snapshot, command, reset);
        last_active_motion_ = motion_result.active_motion;
        last_execution_status_ = motion_result.status;
        pending_execution_feedback_ = make_execution_feedback(
            next_execution_request_id_++, snapshot.server_time,
            motion_result, command);
        if (motion_result.handled) {
            const auto motor_nodes = server::ActionEncoder::encode_motor_actions(motion_result.joint_targets, robot_model_);
            nodes.insert(nodes.end(), motor_nodes.begin(), motor_nodes.end());
        }
    }

    if (team_comm_manager_.is_send_slot(config_.player_number, frame.server_cycle)) {
        std::optional<comm::OutgoingPassIntent> outgoing_pass;
        if (const auto* lifecycle = decision_manager_.outgoing_pass_intent();
            lifecycle != nullptr) {
            outgoing_pass = *lifecycle;
        }
        const auto packet = team_comm_manager_.make_packet(
            snapshot,
            decision::current_role_from_blackboard(decision_manager_.blackboard()),
            outgoing_pass);
        nodes.push_back(server::ActionEncoder::encode_spk(comm::TeamCommCodec::encode(packet)));
    }

    if (config_.status_interval_cycles > 0U && frame.server_cycle >= 0 &&
        static_cast<std::size_t>(frame.server_cycle) % config_.status_interval_cycles == 0U) {
        const auto* restart_decision = decision_manager_.blackboard().exists(
                decision::Blackboard::kKeyRestartDecision)
            ? &decision_manager_.blackboard().get<
                  decision::RestartCoordinationDecision>(
                  decision::Blackboard::kKeyRestartDecision)
            : nullptr;
        const auto* team_plan = decision_manager_.blackboard().exists(
                decision::Blackboard::kKeyTeamPlan)
            ? &decision_manager_.blackboard().get<decision::TeamPlan>(
                  decision::Blackboard::kKeyTeamPlan)
            : nullptr;
        const auto* pass_lifecycle =
            decision_manager_.outgoing_pass_intent();
        const double self_yaw_deg =
            world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                snapshot.self.orientation_wxyz);
        const std::array<double, 2> ball_delta_world{
            snapshot.ball.position_m[0] - snapshot.self.position_m[0],
            snapshot.ball.position_m[1] - snapshot.self.position_m[1],
        };
        const auto ball_local = math::rotate_2d(ball_delta_world, -self_yaw_deg);
        double kick_target_distance_m = 0.0;
        double kick_relative_angle_deg = 0.0;
        if (kick_command != nullptr && kick_command->target_point_m.has_value()) {
            const auto target_delta = math::vec2_sub(
                *kick_command->target_point_m,
                std::array<double, 2>{
                    snapshot.ball.position_m[0], snapshot.ball.position_m[1]});
            kick_target_distance_m = math::norm2(target_delta);
            kick_relative_angle_deg = math::normalize_deg(
                math::vector_angle_deg(target_delta) - self_yaw_deg);
        }
        std::cerr
            << "MY3D_STATUS"
            << " team=" << config_.team_name
            << " player=" << config_.player_number
            << " cycle=" << frame.server_cycle
            << " play_on=" << (snapshot.play_mode == world::PlayMode::PlayOn ? 1 : 0)
            << " motion=" << last_active_motion_
            << " execution=" << behavior::to_string(last_execution_status_)
            << " execution_request_id="
            << (pending_execution_feedback_.has_value()
                    ? pending_execution_feedback_->request_id
                    : 0U)
            << " execution_kind="
            << (pending_execution_feedback_.has_value()
                    ? decision::to_string(
                        pending_execution_feedback_->request_kind)
                    : "Beam")
            << " restart_phase="
            << (restart_decision != nullptr
                    ? decision::to_string(restart_decision->phase)
                    : std::string_view{"Idle"})
            << " restart_epoch="
            << (restart_decision != nullptr &&
                    restart_decision->plan.has_value()
                    ? restart_decision->plan->epoch
                    : 0U)
            << " restart_revision="
            << (restart_decision != nullptr &&
                    restart_decision->plan.has_value()
                    ? restart_decision->plan->revision
                    : 0U)
            << " restart_variant="
            << (restart_decision != nullptr &&
                    restart_decision->plan.has_value()
                    ? decision::to_string(restart_decision->plan->variant)
                    : std::string_view{"None"})
            << " restart_target_x="
            << (restart_decision != nullptr &&
                    restart_decision->plan.has_value()
                    ? restart_decision->plan->contact_target_m[0]
                    : 0.0)
            << " restart_target_y="
            << (restart_decision != nullptr &&
                    restart_decision->plan.has_value()
                    ? restart_decision->plan->contact_target_m[1]
                    : 0.0)
            << " restart_taker="
            << (restart_decision != nullptr &&
                    restart_decision->plan.has_value()
                    ? restart_decision->plan->taker_player_number
                    : 0)
            << " duty="
            << (decision_manager_.blackboard().exists(
                    decision::Blackboard::kKeyTacticalTarget)
                ? decision::to_string(
                    decision_manager_.blackboard().get<decision::TacticalTarget>(
                        decision::Blackboard::kKeyTacticalTarget).duty)
                : std::string_view{"None"})
            << " tactical_target_x="
            << (decision_manager_.blackboard().exists(
                    decision::Blackboard::kKeyTacticalTarget)
                ? decision_manager_.blackboard().get<decision::TacticalTarget>(
                    decision::Blackboard::kKeyTacticalTarget).position_m[0]
                : 0.0)
            << " tactical_target_y="
            << (decision_manager_.blackboard().exists(
                    decision::Blackboard::kKeyTacticalTarget)
                ? decision_manager_.blackboard().get<decision::TacticalTarget>(
                    decision::Blackboard::kKeyTacticalTarget).position_m[1]
                : 0.0)
            << " marked_opponent="
            << (decision_manager_.blackboard().exists(
                    decision::Blackboard::kKeyTacticalTarget)
                ? decision_manager_.blackboard().get<decision::TacticalTarget>(
                    decision::Blackboard::kKeyTacticalTarget)
                    .marked_opponent_player_number
                : 0)
            << " plan_revision=" << (team_plan != nullptr
                    ? team_plan->revision
                    : 0U)
            << " plan_fresh=" << (team_plan != nullptr && team_plan->fresh
                    ? 1
                    : 0)
            << " risk_mode="
            << (decision_manager_.blackboard().exists(
                    decision::Blackboard::kKeyTacticalRiskMode)
                ? strategy::to_string(
                    decision_manager_.blackboard().get<strategy::TacticalRiskMode>(
                        decision::Blackboard::kKeyTacticalRiskMode))
                : std::string_view{"Balanced"})
            << " kick_mode=" << kick_mode_name(command)
            << " ball_visible=" << (snapshot.ball.visible ? 1 : 0)
            << " ball_position_valid=" << (snapshot.ball.position_valid ? 1 : 0)
            << " ball_near_contact_track="
            << (snapshot.ball.near_contact_track ? 1 : 0)
            << " ball_position_age=" << snapshot.ball.position_age_s
            << " ball_velocity_valid=" << (snapshot.ball.velocity_valid ? 1 : 0)
            << " ball_x=" << snapshot.ball.position_m[0]
            << " ball_y=" << snapshot.ball.position_m[1]
            << " ball_local_x=" << ball_local[0]
            << " ball_local_y=" << ball_local[1]
            << " ball_vx=" << snapshot.ball.velocity_mps[0]
            << " ball_vy=" << snapshot.ball.velocity_mps[1]
            << " x=" << snapshot.self.position_m[0]
            << " y=" << snapshot.self.position_m[1]
            << " z=" << snapshot.self.position_m[2]
            << " self_yaw=" << self_yaw_deg
            << " kick_speed=" << (kick_command != nullptr
                    ? kick_command->requested_ball_speed_mps
                    : 0.0)
            << " kick_target_x=" << (kick_command != nullptr &&
                    kick_command->target_point_m.has_value()
                    ? (*kick_command->target_point_m)[0]
                    : 0.0)
            << " kick_target_y=" << (kick_command != nullptr &&
                    kick_command->target_point_m.has_value()
                    ? (*kick_command->target_point_m)[1]
                    : 0.0)
            << " kick_target_distance=" << kick_target_distance_m
            << " kick_relative_angle=" << kick_relative_angle_deg
            << " kick_action_id=" << (kick_command != nullptr
                    ? kick_command->action_id
                    : 0U)
            << " kick_sequence_id=" << (kick_command != nullptr
                    ? static_cast<unsigned int>(kick_command->sequence_id)
                    : 0U)
            << " kick_receiver=" << (kick_command != nullptr
                    ? kick_command->receiver_player_number.value_or(0)
                    : 0)
            << " kick_condition="
            << motion_manager_.active_kick_condition_index()
            << " procedural_kick_anchor="
            << (motion_manager_.active_procedural_kick_anchor().empty()
                    ? std::string{"None"}
                    : motion_manager_.active_procedural_kick_anchor())
            << " learned_kick_active="
            << (motion_manager_.learned_kick_active() ? 1 : 0)
            << " learned_kick_shadow_valid="
            << (motion_manager_.learned_kick_shadow_valid() ? 1 : 0)
            << " learned_kick_max_abs_action="
            << motion_manager_.learned_kick_maximum_absolute_action()
            << " own_score=" << snapshot.own_score
            << " opponent_score=" << snapshot.opponent_score
            << " strategy=" << (selected_action != nullptr
                    ? strategy::to_string(selected_action->category)
                    : std::string_view{"None"})
            << " pass_type=" << (selected_action != nullptr
                    ? strategy::to_string(selected_action->pass_type)
                    : std::string_view{"None"})
            << " action_id=" << (selected_action != nullptr
                    ? selected_action->action_id
                    : 0U)
            << " pass_seq=" << (selected_action != nullptr
                    ? static_cast<unsigned int>(selected_action->sequence_id)
                    : 0U)
            << " receiver=" << (selected_action != nullptr
                    ? selected_action->target_player_number
                    : 0)
            << " pass_ready=" << (selected_action != nullptr &&
                    pass_ready(snapshot, *selected_action) ? 1 : 0)
            << " pass_lifecycle=" << (pass_lifecycle != nullptr
                    ? comm::to_string(pass_lifecycle->state)
                    : std::string_view{"None"})
            << " pass_target_x=" << (selected_action != nullptr
                    ? selected_action->target_point_m[0]
                    : 0.0)
            << " pass_target_y=" << (selected_action != nullptr
                    ? selected_action->target_point_m[1]
                    : 0.0)
            << " pass_margin=" << (selected_action != nullptr
                    ? selected_action->interception_margin_s
                    : 0.0)
            << " pass_utility=" << (selected_action != nullptr
                    ? selected_action->utility
                    : 0.0)
            << " candidates=" << (strategy_plan != nullptr
                    ? strategy_plan->candidates.size()
                    : 0U)
            << " rejected=" << (strategy_plan != nullptr
                    ? strategy_plan->rejections.size()
                    : 0U)
            << " decision_us=" << last_decision_latency_us_
            << " phase=" << (team_plan != nullptr
                    ? strategy::to_string(team_plan->tactical_state.phase)
                    : strategy_plan != nullptr
                    ? strategy::to_string(strategy_plan->tactical_state.phase)
                    : std::string_view{"Unknown"})
            << " possession=" << (team_plan != nullptr
                    ? strategy::to_string(team_plan->tactical_state.possession)
                    : strategy_plan != nullptr
                    ? strategy::to_string(strategy_plan->tactical_state.possession)
                    : std::string_view{"Unknown"})
            << " ball_owner=" << (team_plan != nullptr
                    ? team_plan->tactical_state.ball_owner_player_number
                    : strategy_plan != nullptr
                    ? strategy_plan->tactical_state.ball_owner_player_number
                    : 0)
            << " ball_owner_team=" << (team_plan != nullptr
                    ? (team_plan->tactical_state.ball_owner_player_number <= 0
                        ? "unknown"
                        : team_plan->tactical_state.ball_owner_is_teammate
                        ? "ours"
                        : "theirs")
                    : "unknown")
            << '\n';
    }

    return server::ActionEncoder::finish_frame(nodes);
}

void AgentApp::shutdown() {
    shutdown_requested_ = true;
}

const RuntimeConfig& AgentApp::config() const {
    return config_;
}

bool AgentApp::shutdown_requested() const {
    return shutdown_requested_;
}

const std::string& AgentApp::last_active_motion() const {
    return last_active_motion_;
}

}  // namespace app
