// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/behavior/walk_runner.h"

#include "src/decision/role_manager.h"
#include "src/math/math_utils.h"
#include "src/behavior/policy_common.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace behavior {

namespace {

using math::deg_to_rad;
using math::kPi;
using math::norm2;
using math::quaternion_conjugate;
using math::rotate_2d;
using math::rotate_vec_by_quaternion;
using math::vector_angle_deg;

std::optional<robot::T1RobotModel::HeadTargets> update_head_tracker(
    behavior::WalkRunner::HeadTrackerState& state,
    const world::WorldSnapshot& snapshot,
    const robot::T1RobotModel& robot_model,
    std::optional<int> role_id) {
    constexpr double lock_distance = 1.0;
    constexpr double lock_hold_time = 0.6;
    constexpr double search_sweep_deg = 70.0;
    constexpr double search_period = 3.0;
    constexpr double sweep_min_deg = -90.0;
    constexpr double sweep_max_deg = 90.0;
    constexpr double sweep_speed_deg_per_sec = 60.0;
    constexpr double sweep_pitch_deg = 0.0;

    // Only the GK (ROLE_GK = 0) tracks the ball with its head; every other
    // role sweeps the field. Roles are 0..6 in this codebase (see
    // decision::RoleManager), so there is no special "role 10" anymore.
    if (role_id.has_value() && *role_id != decision::RoleManager::ROLE_GK) {
        const double sweep_range = sweep_max_deg - sweep_min_deg;
        const double period = sweep_range / sweep_speed_deg_per_sec * 2.0;
        const double t = std::fmod(snapshot.server_time, period);
        double yaw = 0.0;
        if (t < period / 2.0) {
            const double progress = t / (period / 2.0);
            yaw = sweep_min_deg + progress * sweep_range;
        } else {
            const double progress = (t - period / 2.0) / (period / 2.0);
            yaw = sweep_max_deg - progress * sweep_range;
        }
        return robot_model.clamp_head_targets(yaw, sweep_pitch_deg);
    }

    if (snapshot.ball.visible) {
        const std::array<double, 3> ball_vec_world{
            snapshot.ball.position_m[0] - snapshot.self.position_m[0],
            snapshot.ball.position_m[1] - snapshot.self.position_m[1],
            snapshot.ball.position_m[2] - snapshot.self.position_m[2],
        };
        if (norm2({ball_vec_world[0], ball_vec_world[1]}) <= lock_distance) {
            if (!state.last_head_target_deg.has_value()) {
                return std::nullopt;
            }
            const auto target = state.last_head_target_deg.value();
            return robot_model.clamp_head_targets(target[0], target[1]);
        }

        const auto body_vec = rotate_vec_by_quaternion(ball_vec_world, quaternion_conjugate(snapshot.self.orientation_wxyz));
        const double horiz = std::sqrt(body_vec[0] * body_vec[0] + body_vec[1] * body_vec[1]);
        if (horiz < 1e-6) {
            return std::nullopt;
        }
        const double yaw = std::atan2(body_vec[1], body_vec[0]) * 180.0 / kPi;
        const double pitch = -std::atan2(body_vec[2], horiz) * 180.0 / kPi;
        state.last_ball_seen_time = snapshot.server_time;
        state.last_head_target_deg = std::array<double, 2>{yaw, pitch};
        return robot_model.clamp_head_targets(yaw, pitch);
    }

    if (!state.last_head_target_deg.has_value() || !state.last_ball_seen_time.has_value()) {
        return std::nullopt;
    }

    const double time_since_seen = snapshot.server_time - state.last_ball_seen_time.value();
    if (time_since_seen <= lock_hold_time) {
        const auto target = state.last_head_target_deg.value();
        return robot_model.clamp_head_targets(target[0], target[1]);
    }

    const double phase =
        2.0 * kPi * std::fmod(snapshot.server_time, search_period) / search_period;
    const double yaw = search_sweep_deg * std::sin(phase);
    return robot_model.clamp_head_targets(yaw, sweep_pitch_deg);
}

}  // namespace

WalkRunner::WalkRunner(
    const std::filesystem::path& model_path,
    std::optional<std::filesystem::path> fast_walk_model_path)
    : session_(model_path, OnnxModelContract{{1, 78}, {1, 23}}) {
    if (fast_walk_model_path.has_value()) {
        fast_walk_session_.emplace(
            *fast_walk_model_path,
            OnnxModelContract{{1, 80}, {1, 23}});
    }
    step_obs_dim_ = 9 + 3 * static_cast<int>(robot_model_.readable_joint_names().size());
    const auto& input_shape = session_.info().input_shape;
    history_length_ = static_cast<int>(input_shape.at(1) / step_obs_dim_);
    if (history_length_ < 1 || history_length_ * step_obs_dim_ != input_shape.at(1)) {
        throw std::runtime_error("WalkRunner input dimension is incompatible with step observation layout");
    }
    previous_action_.assign(robot_model_.readable_joint_names().size(), 0.0F);
    observation_.assign(static_cast<std::size_t>(history_length_ * step_obs_dim_), 0.0F);
    step_obs_buffer_.assign(static_cast<std::size_t>(step_obs_dim_), 0.0F);
    fast_previous_action_.assign(robot_model_.readable_joint_names().size(), 0.0F);
}

bool WalkRunner::fast_walk_supported(
    const world::WorldSnapshot& snapshot,
    const decision::WalkCommand& command,
    const std::array<float, 3>& stable_velocity_command) {
    if (!fast_walk_session_.has_value() || fast_walk_disabled_ ||
        snapshot.play_mode != world::PlayMode::PlayOn ||
        snapshot.player_number == 1) {
        fast_walk_active_ = false;
        return false;
    }
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    std::array<double, 2> local_target = command.target_2d_m;
    if (command.target_absolute) {
        const std::array<double, 2> target_delta{
            command.target_2d_m[0] - snapshot.self.position_m[0],
            command.target_2d_m[1] - snapshot.self.position_m[1],
        };
        local_target = rotate_2d(target_delta, -self_yaw_deg);
    }
    const auto gravity_body = rotate_vec_by_quaternion(
        {0.0, 0.0, -1.0}, quaternion_conjugate(snapshot.self.orientation_wxyz));
    const double tilt_deg = math::rad_to_deg(std::acos(
        std::clamp(-gravity_body[2], -1.0, 1.0)));
    const double max_gyro = std::max({
        std::abs(snapshot.self.gyro_deg_s[0]),
        std::abs(snapshot.self.gyro_deg_s[1]),
        std::abs(snapshot.self.gyro_deg_s[2]),
    });
    const double distance_m = command.target_absolute
        ? math::norm2(local_target)
        : std::numeric_limits<double>::infinity();
    constexpr double kFastWalkEntryTiltDeg = 6.0;
    constexpr double kFastWalkActiveTiltDeg = 11.0;
    constexpr double kFastWalkEntryGyroDegS = 55.0;
    constexpr double kFastWalkActiveGyroDegS = 110.0;
    constexpr double kFastWalkMinimumHeightM = 0.55;
    constexpr double kFastWalkRecoveryCooldownS = 4.0;
    const bool was_fast_walk_active = fast_walk_active_;
    int gate = 0;
    const char* reason = "active";
    if (snapshot.server_time < fast_walk_cooldown_until_s_) {
        gate = 8;
        reason = "recovery-cooldown";
    } else if (snapshot.self.position_m[2] < kFastWalkMinimumHeightM) {
        gate = 6;
        reason = "height";
    } else if (tilt_deg > (was_fast_walk_active
                               ? kFastWalkActiveTiltDeg
                               : kFastWalkEntryTiltDeg)) {
        gate = 4;
        reason = "tilt";
    } else if (max_gyro > (was_fast_walk_active
                               ? kFastWalkActiveGyroDegS
                               : kFastWalkEntryGyroDegS)) {
        gate = 5;
        reason = "gyro";
    } else if (command.target_absolute && distance_m < 2.0) {
        gate = 1;
        reason = "target-near";
    } else if (stable_velocity_command[0] < 0.95F) {
        gate = 2;
        reason = "target-not-forward";
    } else if (std::abs(stable_velocity_command[1]) > 0.04F) {
        gate = 7;
        reason = "lateral-command";
    } else if (std::abs(stable_velocity_command[2]) > 0.12F) {
        gate = 3;
        reason = "turn-rate";
    }
    if (was_fast_walk_active && gate >= 4 && gate <= 6) {
        fast_walk_cooldown_until_s_ = std::max(
            fast_walk_cooldown_until_s_,
            snapshot.server_time + kFastWalkRecoveryCooldownS);
    }
    if (gate != last_fast_walk_gate_) {
        std::cerr
            << "MY3D_FAST_WALK_GATE player=" << snapshot.player_number
            << " state=" << reason
            << " distance=" << distance_m
            << " local_x=" << local_target[0]
            << " local_y=" << local_target[1]
            << " vx=" << stable_velocity_command[0]
            << " vy=" << stable_velocity_command[1]
            << " yaw_rate=" << stable_velocity_command[2]
            << " tilt_deg=" << tilt_deg
            << " gyro_deg_s=" << max_gyro
            << " z=" << snapshot.self.position_m[2]
            << " cooldown_remaining="
            << std::max(
                   0.0,
                   fast_walk_cooldown_until_s_ - snapshot.server_time)
            << '\n';
        last_fast_walk_gate_ = gate;
    }
    fast_walk_active_ = gate == 0;
    return fast_walk_active_;
}

std::optional<robot::JointTargets> WalkRunner::step_fast_walk(
    const world::WorldSnapshot& snapshot,
    const decision::WalkCommand& command,
    const std::array<float, 3>& stable_velocity_command,
    const robot::JointTargets& stable_targets,
    bool reset) {
    if (!fast_walk_supported(snapshot, command, stable_velocity_command)) {
        fast_walk_active_ = false;
        std::fill(fast_previous_action_.begin(), fast_previous_action_.end(), 0.0F);
        fast_gait_phase_ = 0.0;
        return std::nullopt;
    }
    if (reset) {
        std::fill(fast_previous_action_.begin(), fast_previous_action_.end(), 0.0F);
        fast_gait_phase_ = 0.0;
    }

    static constexpr std::array<float, 23> nominal_training{
        0.0F, 0.0F, 0.0F, 1.4F, 0.0F, -0.4F, 0.0F, -1.4F,
        0.0F, 0.4F, 0.0F, -0.4F, 0.0F, 0.0F, 0.8F, -0.4F,
        0.0F, 0.4F, 0.0F, 0.0F, -0.8F, 0.4F, 0.0F,
    };
    static constexpr std::array<float, 23> train_to_server_sign{
        1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F,
        1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F,
        -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
    };
    constexpr float action_scale = 0.5F;
    constexpr float gait_frequency_hz = 1.75F;

    std::vector<float> observation;
    observation.reserve(80U);
    const auto& names = robot_model_.readable_joint_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto pos = snapshot.self.joint_positions_deg.find(names[i]);
        const auto vel = snapshot.self.joint_velocities_deg_s.find(names[i]);
        const float position_server = static_cast<float>(math::deg_to_rad(
            pos == snapshot.self.joint_positions_deg.end() ? 0.0 : pos->second));
        const float velocity_server = static_cast<float>(math::deg_to_rad(
            vel == snapshot.self.joint_velocities_deg_s.end() ? 0.0 : vel->second));
        observation.push_back(
            (position_server * train_to_server_sign[i] - nominal_training[i]) / 4.6F);
        observation.push_back(
            velocity_server * train_to_server_sign[i] / 110.0F);
        observation.push_back(fast_previous_action_[i] / 10.0F);
    }
    const auto imu = build_imu_obs(snapshot);
    observation.push_back(imu[0] / 50.0F);
    observation.push_back(imu[1] / 50.0F);
    observation.push_back(imu[2] / 50.0F);
    observation.push_back(1.5F);
    observation.push_back(std::clamp(stable_velocity_command[1], -0.1F, 0.1F));
    observation.push_back(std::clamp(stable_velocity_command[2], -0.2F, 0.2F));
    observation.push_back(imu[3]);
    observation.push_back(imu[4]);
    observation.push_back(imu[5]);
    const double phase_angle = 2.0 * kPi * fast_gait_phase_;
    observation.push_back(static_cast<float>(std::cos(phase_angle)));
    observation.push_back(static_cast<float>(std::sin(phase_angle)));
    for (float& value : observation) {
        value = std::clamp(std::isfinite(value) ? value : 0.0F, -10.0F, 10.0F);
    }

    try {
        auto action = fast_walk_session_->run(observation);
        if (action.size() != names.size()) {
            throw std::runtime_error("fast-walk actor output size mismatch");
        }
        robot::JointTargets targets = stable_targets;
        for (std::size_t i = 0; i < action.size(); ++i) {
            if (!std::isfinite(action[i])) {
                throw std::runtime_error("fast-walk actor returned non-finite action");
            }
            action[i] = std::clamp(action[i], -10.0F, 10.0F);
            if (i < 2U) {
                continue;
            }
            const double target_training =
                nominal_training[i] + action_scale * action[i];
            targets[i].q_deg = math::rad_to_deg(
                target_training * train_to_server_sign[i]);
            targets[i].kp = 25.0;
            targets[i].kd = 0.6;
        }
        fast_previous_action_ = std::move(action);
        fast_gait_phase_ = std::fmod(
            fast_gait_phase_ + 0.02 * gait_frequency_hz, 1.0);
        return targets;
    } catch (const std::exception& error) {
        std::cerr << "MY3D_FAST_WALK_DISABLED error=" << error.what() << '\n';
        fast_walk_disabled_ = true;
        fast_walk_active_ = false;
        std::fill(fast_previous_action_.begin(), fast_previous_action_.end(), 0.0F);
        fast_gait_phase_ = 0.0;
        return std::nullopt;
    }
}

std::array<float, 3> WalkRunner::compute_velocity_command(
    const world::WorldSnapshot& snapshot,
    const decision::WalkCommand& command) const {
    std::array<float, 3> velocity{0.0F, 0.0F, 0.0F};
    const double self_yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(snapshot.self.orientation_wxyz);
    if (command.target_absolute) {
        const std::array<double, 2> raw_target{
            command.target_2d_m[0] - snapshot.self.position_m[0],
            command.target_2d_m[1] - snapshot.self.position_m[1],
        };
        const auto local_target = rotate_2d(raw_target, -self_yaw_deg);
        velocity[0] = static_cast<float>(local_target[0]);
        velocity[1] = static_cast<float>(local_target[1]);
    } else {
        velocity[0] = static_cast<float>(command.target_2d_m[0]);
        velocity[1] = static_cast<float>(command.target_2d_m[1]);
    }

    double rel_orientation_deg = 0.0;
    if (!command.orientation_deg.has_value()) {
        const std::array<double, 2> planar{velocity[0], velocity[1]};
        if (norm2(planar) > 0.1) {
            rel_orientation_deg = vector_angle_deg(planar);
        }
    } else if (command.orientation_absolute) {
        rel_orientation_deg = math::normalize_deg(command.orientation_deg.value() - self_yaw_deg);
    } else {
        rel_orientation_deg = math::normalize_deg(command.orientation_deg.value());
    }

    const double orientation_gain = std::clamp(
        std::isfinite(command.orientation_gain) ? command.orientation_gain : 1.0,
        0.0,
        4.0);
    velocity[2] = static_cast<float>(
        kOrientationToAngVelScale * orientation_gain *
        deg_to_rad(rel_orientation_deg));
    velocity[0] = std::clamp(velocity[0], -0.5F, 1.0F);
    velocity[1] = std::clamp(velocity[1], -0.5F, 0.5F);
    velocity[2] = std::clamp(velocity[2], -0.5F, 0.5F);
    return velocity;
}

void WalkRunner::build_observation(
    const world::WorldSnapshot& snapshot,
    const std::array<float, 3>& velocity_command) {
    std::fill(step_obs_buffer_.begin(), step_obs_buffer_.end(), 0.0F);
    const auto imu = build_imu_obs(snapshot);
    step_obs_buffer_[0] = imu[0];
    step_obs_buffer_[1] = imu[1];
    step_obs_buffer_[2] = imu[2];
    step_obs_buffer_[3] = imu[3];
    step_obs_buffer_[4] = imu[4];
    step_obs_buffer_[5] = imu[5];
    step_obs_buffer_[6] = velocity_command[0];
    step_obs_buffer_[7] = velocity_command[1];
    step_obs_buffer_[8] = velocity_command[2];

    const auto& joint_names = robot_model_.readable_joint_names();
    const std::size_t offset = 9U;
    auto* pos_out = step_obs_buffer_.data() + offset;
    auto* vel_out = pos_out + joint_names.size();
    auto* prev_out = vel_out + joint_names.size();
    fill_joint_obs(pos_out, vel_out, prev_out, snapshot, robot_model_, previous_action_);
    for (std::size_t i = 0; i < 2U; ++i) {
        pos_out[i] = 0.0F;
        vel_out[i] = 0.0F;
        prev_out[i] = 0.0F;
    }
}

robot::JointTargets WalkRunner::decode_action(
    const world::WorldSnapshot& snapshot,
    const std::vector<float>& action,
    std::optional<int> role_id) {
    robot::JointTargets joint_targets = decode_action_base(action, robot_model_, kActionScale);

    const auto head_target = update_head_tracker(head_tracker_state_, snapshot, robot_model_, role_id);
    if (head_target.has_value()) {
        joint_targets[0].q_deg = head_target->yaw_deg;
        joint_targets[1].q_deg = head_target->pitch_deg;
    }

    return joint_targets;
}

WalkStepResult WalkRunner::step(
    const world::WorldSnapshot& snapshot,
    const decision::WalkCommand& command,
    bool reset,
    std::optional<int> role_id) {
    if (reset) {
        std::fill(previous_action_.begin(), previous_action_.end(), 0.0F);
    }

    const auto velocity_command = compute_velocity_command(snapshot, command);
    build_observation(snapshot, velocity_command);
    advance_history(observation_, step_obs_buffer_, step_obs_dim_, history_length_, reset);

    auto action = session_.run(observation_);
    for (float& value : action) {
        value = std::clamp(value, -5.0F, 5.0F);
    }
    previous_action_ = action;

    auto stable_targets = decode_action(snapshot, action, role_id);
    const auto fast_targets = step_fast_walk(
        snapshot, command, velocity_command, stable_targets, reset);
    return {
        observation_,
        action,
        fast_targets.has_value() ? *fast_targets : std::move(stable_targets),
        fast_targets.has_value(),
    };
}

}  // namespace behavior
