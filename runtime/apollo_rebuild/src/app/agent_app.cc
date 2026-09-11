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
#include <stdexcept>
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

}  // namespace

AgentApp::AgentApp(RuntimeConfig config)
    : config_(std::move(config)),
      world_state_(config_.team_name, config_.player_number, 7),
      motion_manager_(config_),
      team_comm_manager_(config_.team_name) {}

AgentApp::AgentApp(RuntimeConfig config, std::unique_ptr<server::TcpLpmClient> client)
    : config_(std::move(config)),
      world_state_(config_.team_name, config_.player_number, 7),
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

    ++processed_frames_;
    if (config_.status_interval > 0 &&
        processed_frames_ % config_.status_interval == 0) {
        const std::array<double, 2> self{
            snapshot.self.position_m[0], snapshot.self.position_m[1]};
        const std::array<double, 2> ball{
            snapshot.ball.position_m[0], snapshot.ball.position_m[1]};
        const double self_yaw_deg =
            world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
                snapshot.self.orientation_wxyz);
        double walk_target_norm = -1.0;
        double walk_target_x = 0.0;
        double walk_target_y = 0.0;
        double walk_orientation_deg = 0.0;
        int walk_target_absolute = -1;
        int walk_orientation_present = 0;
        int walk_orientation_absolute = -1;
        if (const auto* walk = std::get_if<decision::WalkCommand>(&command)) {
            walk_target_norm = math::norm2(walk->target_2d_m);
            walk_target_x = walk->target_2d_m[0];
            walk_target_y = walk->target_2d_m[1];
            walk_target_absolute = walk->target_absolute ? 1 : 0;
            if (walk->orientation_deg.has_value()) {
                walk_orientation_present = 1;
                walk_orientation_deg = walk->orientation_deg.value();
                walk_orientation_absolute = walk->orientation_absolute ? 1 : 0;
            }
        }
        std::cerr
            << "APOLLO_REBUILD_STATUS"
            << " t=" << snapshot.server_time
            << " player=" << snapshot.player_number
            << " mode=" << static_cast<int>(snapshot.play_mode)
            << " role=" << decision::current_role_from_blackboard(
                   decision_manager_.blackboard())
            << " ball_dist=" << math::planar_dist(self, ball)
            << " ball_visible=" << (snapshot.ball.visible ? 1 : 0)
            << " ball_x=" << snapshot.ball.position_m[0]
            << " ball_y=" << snapshot.ball.position_m[1]
            << " x=" << snapshot.self.position_m[0]
            << " y=" << snapshot.self.position_m[1]
            << " z=" << snapshot.self.position_m[2]
            << " yaw_deg=" << self_yaw_deg
            << " self_speed=" << math::norm2({
                   snapshot.self.lin_vel_b[0], snapshot.self.lin_vel_b[1]})
            << " ball_speed=" << (snapshot.ball.velocity_valid
                   ? math::norm2({
                         snapshot.ball.velocity_mps[0],
                         snapshot.ball.velocity_mps[1]})
                   : -1.0)
            << " motion=" << last_active_motion_
            << " walk_target_norm=" << walk_target_norm
            << " walk_target_x=" << walk_target_x
            << " walk_target_y=" << walk_target_y
            << " walk_target_absolute=" << walk_target_absolute
            << " walk_orientation_present=" << walk_orientation_present
            << " walk_orientation_deg=" << walk_orientation_deg
            << " walk_orientation_absolute=" << walk_orientation_absolute
            << '\n';
    }

    if (team_comm_manager_.is_send_slot(config_.player_number, frame.server_cycle)) {
        const auto packet = team_comm_manager_.make_packet(
            snapshot,
            decision::current_role_from_blackboard(decision_manager_.blackboard()));
        nodes.push_back(server::ActionEncoder::encode_spk(comm::TeamCommCodec::encode(packet)));
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
