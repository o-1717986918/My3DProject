// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/app/agent_app.h"

#include "src/comm/team_comm_codec.h"
#include "src/decision/role_behaviors.h"
#include "src/math/math_utils.h"
#include "src/server/action_encoder.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
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
        case decision::KickMode::TargetedPass: return "TargetedPass";
        case decision::KickMode::Shot: return "Shot";
        case decision::KickMode::Clear: return "Clear";
    }
    return "None";
}

}  // namespace

AgentApp::AgentApp(RuntimeConfig config)
    : config_(std::move(config)),
      world_state_(config_.team_name, config_.player_number, 7),
      decision_manager_(config_.enable_pass_strategy),
      motion_manager_(config_),
      team_comm_manager_(config_.team_name) {}

AgentApp::AgentApp(RuntimeConfig config, std::unique_ptr<server::TcpLpmClient> client)
    : config_(std::move(config)),
      world_state_(config_.team_name, config_.player_number, 7),
      decision_manager_(config_.enable_pass_strategy),
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
    const auto command = decision_manager_.decide(snapshot);
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
    } else {
        const auto motion_result = motion_manager_.step(snapshot, command, reset);
        last_active_motion_ = motion_result.active_motion;
        if (motion_result.handled) {
            const auto motor_nodes = server::ActionEncoder::encode_motor_actions(motion_result.joint_targets, robot_model_);
            nodes.insert(nodes.end(), motor_nodes.begin(), motor_nodes.end());
        }
    }

    if (team_comm_manager_.is_send_slot(config_.player_number, frame.server_cycle)) {
        std::optional<comm::OutgoingPassIntent> outgoing_pass;
        if (selected_action != nullptr &&
            selected_action->category == strategy::ActionCategory::Pass &&
            selected_action->actor_player_number == snapshot.player_number) {
            outgoing_pass = comm::OutgoingPassIntent{
                selected_action->target_player_number,
                selected_action->sequence_id,
                selected_action->target_point_m[0],
                selected_action->target_point_m[1],
                selected_action->requested_ball_speed_mps,
                selected_action->predicted_ball_time_s,
            };
        }
        const auto packet = team_comm_manager_.make_packet(
            snapshot,
            decision::current_role_from_blackboard(decision_manager_.blackboard()),
            outgoing_pass);
        nodes.push_back(server::ActionEncoder::encode_spk(comm::TeamCommCodec::encode(packet)));
    }

    if (config_.status_interval_cycles > 0U && frame.server_cycle >= 0 &&
        static_cast<std::size_t>(frame.server_cycle) % config_.status_interval_cycles == 0U) {
        std::cerr
            << "MY3D_STATUS"
            << " team=" << config_.team_name
            << " player=" << config_.player_number
            << " cycle=" << frame.server_cycle
            << " play_on=" << (snapshot.play_mode == world::PlayMode::PlayOn ? 1 : 0)
            << " motion=" << last_active_motion_
            << " kick_mode=" << kick_mode_name(command)
            << " ball_visible=" << (snapshot.ball.visible ? 1 : 0)
            << " ball_velocity_valid=" << (snapshot.ball.velocity_valid ? 1 : 0)
            << " ball_x=" << snapshot.ball.position_m[0]
            << " ball_y=" << snapshot.ball.position_m[1]
            << " ball_vx=" << snapshot.ball.velocity_mps[0]
            << " ball_vy=" << snapshot.ball.velocity_mps[1]
            << " x=" << snapshot.self.position_m[0]
            << " y=" << snapshot.self.position_m[1]
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
            << " phase=" << (strategy_plan != nullptr
                    ? strategy::to_string(strategy_plan->tactical_state.phase)
                    : std::string_view{"Unknown"})
            << " possession=" << (strategy_plan != nullptr
                    ? strategy::to_string(strategy_plan->tactical_state.possession)
                    : std::string_view{"Unknown"})
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
