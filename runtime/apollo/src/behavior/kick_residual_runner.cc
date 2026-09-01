// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/kick_residual_runner.h"

#include "src/math/math_utils.h"
#include "src/world/frame_normalizer.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace behavior {

namespace {

constexpr double kNominalBallLocalXM = 0.32;
constexpr double kDistanceToleranceM = 0.55;
constexpr double kAngleToleranceDeg = 2.0;
constexpr double kSpeedToleranceMps = 0.20;
constexpr double kBallXMinimumM = -0.02;
constexpr double kBallXMaximumM = 0.09;
constexpr double kBallYMaximumAbsM = 0.09;

constexpr std::array<double, 23> kActionScaleRad{
    0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.20, 0.20, 0.15, 0.35, 0.25, 0.25, 0.45, 0.25,
    0.20, 0.35, 0.25, 0.25, 0.45, 0.25, 0.20,
};

constexpr std::array<double, 23> kDefaultPoseRad{
    0.0, 0.0, 0.0, -1.4, 0.0, -0.4, 0.0, 1.4,
    0.0, 0.4, 0.0, -0.2, 0.0, 0.0, 0.4, -0.2,
    0.0, -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
};

constexpr std::array<double, 23> kJointLowerRad{
    -1.57, -0.35, -3.31, -1.74, -2.27, -2.44, -3.31, -1.57,
    -2.27, 0.0, -1.57, -1.8, -0.2, -1.0, 0.0, -0.87,
    -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44,
};

constexpr std::array<double, 23> kJointUpperRad{
    1.57, 1.22, 1.22, 1.57, 2.27, 0.0, 1.22, 1.74,
    2.27, 2.44, 1.57, 1.57, 1.57, 1.0, 2.34, 0.35,
    0.44, 1.57, 0.2, 1.0, 2.34, 0.35, 0.44,
};

double smoothstep(double value) {
    const double bounded = std::clamp(value, 0.0, 1.0);
    return bounded * bounded * (3.0 - 2.0 * bounded);
}

}  // namespace

KickResidualRunner::KickResidualRunner(const std::filesystem::path& table_path) {
    const YAML::Node root = YAML::LoadFile(table_path.string());
    if (!root["schema_version"] || root["schema_version"].as<int>() != 1) {
        throw std::runtime_error("unsupported kick residual table schema");
    }
    const YAML::Node node_list = root["nodes"];
    if (!node_list || !node_list.IsSequence() || node_list.size() == 0U) {
        throw std::runtime_error("kick residual table contains no nodes");
    }
    for (const auto& encoded : node_list) {
        Node node;
        node.condition_index = encoded["condition_index"].as<int>();
        node.distance_m = encoded["distance_m"].as<double>();
        node.angle_deg = encoded["angle_deg"].as<double>();
        node.requested_speed_mps = encoded["requested_speed_mps"].as<double>();
        node.ball_x_offset_m = encoded["ball_x_offset_m"].as<double>();
        node.ball_y_offset_m = encoded["ball_y_offset_m"].as<double>();
        node.mode = encoded["mode"].as<std::string>();
        const YAML::Node parameters = encoded["parameters"];
        if (!parameters || !parameters.IsSequence() ||
            parameters.size() != kParameterCount) {
            throw std::runtime_error("kick residual node must contain 14 parameters");
        }
        for (std::size_t index = 0; index < kParameterCount; ++index) {
            node.parameters[index] = parameters[index].as<double>();
            if (!std::isfinite(node.parameters[index])) {
                throw std::runtime_error("kick residual parameter is not finite");
            }
        }
        nodes_.push_back(std::move(node));
    }
}

bool KickResidualRunner::begin(
    const world::WorldSnapshot& snapshot,
    const KickExecutionProfile& profile) {
    selected_ = nullptr;
    const bool ball_track_usable =
        snapshot.ball.visible ||
        snapshot.ball.position_age_s <= world::kBallPositionFreshLifetimeS ||
        (snapshot.ball.near_contact_track &&
         snapshot.ball.position_age_s <=
             world::kNearContactBallTrackLifetimeS);
    if (!snapshot.ball.position_valid ||
        !ball_track_usable ||
        profile.kind != KickProfileKind::ParameterizedContact ||
        profile.mode != decision::KickMode::TargetedPass ||
        !std::isfinite(profile.target_distance_m) ||
        !std::isfinite(profile.relative_target_angle_deg) ||
        !std::isfinite(profile.requested_speed_mps)) {
        return false;
    }

    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const std::array<double, 2> ball_delta_world{
        snapshot.ball.position_m[0] - snapshot.self.position_m[0],
        snapshot.ball.position_m[1] - snapshot.self.position_m[1],
    };
    const auto ball_local = math::rotate_2d(ball_delta_world, -yaw_deg);
    const double ball_x_offset_m = ball_local[0] - kNominalBallLocalXM;
    const double ball_y_offset_m = ball_local[1];
    if (ball_x_offset_m < kBallXMinimumM ||
        ball_x_offset_m > kBallXMaximumM ||
        std::abs(ball_y_offset_m) > kBallYMaximumAbsM) {
        return false;
    }

    double best_score = std::numeric_limits<double>::infinity();
    for (const Node& node : nodes_) {
        if (node.mode != "pass" ||
            std::abs(profile.target_distance_m - node.distance_m) >
                kDistanceToleranceM ||
            std::abs(profile.relative_target_angle_deg - node.angle_deg) >
                kAngleToleranceDeg ||
            std::abs(profile.requested_speed_mps - node.requested_speed_mps) >
                kSpeedToleranceMps) {
            continue;
        }
        const double distance_error =
            (profile.target_distance_m - node.distance_m) / 3.0;
        const double angle_error =
            (profile.relative_target_angle_deg - node.angle_deg) / 15.0;
        const double speed_error =
            (profile.requested_speed_mps - node.requested_speed_mps) / 2.2;
        const double ball_x_error =
            (ball_x_offset_m - node.ball_x_offset_m) / 0.09;
        const double ball_y_error =
            (ball_y_offset_m - node.ball_y_offset_m) / 0.08;
        const double score =
            4.0 * distance_error * distance_error +
            angle_error * angle_error + speed_error * speed_error +
            ball_x_error * ball_x_error + ball_y_error * ball_y_error;
        if (score < best_score) {
            best_score = score;
            selected_ = &node;
        }
    }
    return selected_ != nullptr;
}

std::array<double, KickResidualRunner::kJointCount>
KickResidualRunner::residual_at(double elapsed_s) const {
    std::array<double, kJointCount> zero{};
    if (selected_ == nullptr || !std::isfinite(elapsed_s)) {
        return zero;
    }
    constexpr std::array<double, 6> times{0.0, 0.18, 0.34, 0.54, 0.76, 1.20};
    std::array<std::array<double, kJointCount>, 6> keys{};
    const auto& p = selected_->parameters;

    for (const std::size_t key_index : {1U, 2U, 3U}) {
        keys[key_index][12] = p[0];
        keys[key_index][18] = p[0];
        keys[key_index][16] = p[1];
        keys[key_index][22] = p[1];
    }
    keys[2][17] = p[2];
    keys[2][20] = p[3];
    keys[2][21] = p[4];
    keys[3][17] = p[5];
    keys[3][20] = p[6];
    keys[3][21] = p[7];
    keys[3][19] = p[8];
    keys[3][18] += p[12];
    keys[3][22] += p[13];
    for (const std::size_t key_index : {2U, 3U}) {
        keys[key_index][10] = p[9];
        keys[key_index][2] = p[10];
        keys[key_index][6] = p[11];
    }
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        keys[4][joint] = 0.35 * keys[3][joint];
    }

    std::size_t left = 0U;
    if (elapsed_s >= times.back()) {
        left = times.size() - 2U;
    } else {
        while (left + 1U < times.size() && elapsed_s > times[left + 1U]) {
            ++left;
        }
    }
    const std::size_t right = std::min(left + 1U, times.size() - 1U);
    const double fraction = smoothstep(
        (elapsed_s - times[left]) / (times[right] - times[left]));
    std::array<double, kJointCount> residual{};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        const double raw =
            keys[left][joint] * (1.0 - fraction) + keys[right][joint] * fraction;
        const double joint_limited =
            std::clamp(
                kDefaultPoseRad[joint] + raw,
                kJointLowerRad[joint],
                kJointUpperRad[joint]) -
            kDefaultPoseRad[joint];
        residual[joint] = std::clamp(
            joint_limited / kActionScaleRad[joint], -1.0, 1.0) *
            kActionScaleRad[joint];
    }
    return residual;
}

bool KickResidualRunner::apply(
    double elapsed_s,
    robot::JointTargets& joint_targets) const {
    if (selected_ == nullptr) {
        return false;
    }
    const auto residual = residual_at(elapsed_s);
    for (auto& target : joint_targets) {
        const auto index = robot_model_.joint_order_index(target.joint_name);
        if (index.has_value()) {
            target.q_deg += math::rad_to_deg(residual.at(*index));
        }
    }
    return true;
}

std::size_t KickResidualRunner::node_count() const {
    return nodes_.size();
}

int KickResidualRunner::selected_condition_index() const {
    return selected_ != nullptr ? selected_->condition_index : -1;
}

}  // namespace behavior
