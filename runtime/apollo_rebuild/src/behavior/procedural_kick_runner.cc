// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/procedural_kick_runner.h"

#include "src/math/math_utils.h"
#include "src/decision/kick_contract.h"
#include "src/robot/t1_joint_limits.h"
#include "src/world/frame_normalizer.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace behavior {

namespace {

constexpr double kCaptureDurationS = 0.18;

constexpr std::array<double, 14> kParameterLower{
    -0.25, -0.20, -0.80, -1.00, -0.60, -1.00, -1.20,
    -0.80, -0.40, -0.30, -0.50, -0.50, -0.45, -0.35,
};
constexpr std::array<double, 14> kParameterUpper{
    0.25, 0.20, 0.80, 1.00, 0.60, 1.00, 1.20,
    0.80, 0.40, 0.30, 0.50, 0.50, 0.45, 0.35,
};

double smoothstep(double value) {
    const double bounded = std::clamp(value, 0.0, 1.0);
    return bounded * bounded * (3.0 - 2.0 * bounded);
}

std::string mode_name(decision::KickMode mode) {
    switch (mode) {
        case decision::KickMode::DribbleTouch: return "dribble";
        case decision::KickMode::TargetedPass: return "pass";
        case decision::KickMode::Shot: return "shot";
        case decision::KickMode::Clear: return "clear";
        case decision::KickMode::ForwardContact: return "forward_contact";
    }
    return "unsupported";
}

bool ball_track_usable(const world::WorldSnapshot& snapshot) {
    return snapshot.ball.visible ||
        snapshot.ball.position_age_s <= world::kBallPositionFreshLifetimeS ||
        (snapshot.ball.near_contact_track &&
         snapshot.ball.position_age_s <= world::kNearContactBallTrackLifetimeS);
}

}  // namespace

ProceduralKickRunner::ProceduralKickRunner(
    const std::filesystem::path& asset_path) {
    const YAML::Node root = YAML::LoadFile(asset_path.string());
    if (!root["schema_version"] || root["schema_version"].as<int>() != 1) {
        throw std::runtime_error("unsupported procedural kick schema");
    }
    const YAML::Node encoded_anchors = root["anchors"];
    if (!encoded_anchors || !encoded_anchors.IsSequence() ||
        encoded_anchors.size() == 0U) {
        throw std::runtime_error("procedural kick asset contains no anchors");
    }
    for (const auto& encoded : encoded_anchors) {
        Anchor anchor;
        anchor.name = encoded["name"].as<std::string>();
        anchor.mode = encoded["mode"].as<std::string>();
        anchor.target_distance_m = encoded["target_distance_m"].as<double>();
        anchor.target_distance_tolerance_m =
            encoded["target_distance_tolerance_m"].as<double>();
        anchor.target_angle_deg = encoded["target_angle_deg"].as<double>();
        anchor.target_angle_tolerance_deg =
            encoded["target_angle_tolerance_deg"].as<double>();
        anchor.requested_speed_mps = encoded["requested_speed_mps"].as<double>();
        anchor.requested_speed_tolerance_mps =
            encoded["requested_speed_tolerance_mps"].as<double>();
        anchor.ball_local_x_m = encoded["ball_local_x_m"].as<double>();
        anchor.ball_local_y_m = encoded["ball_local_y_m"].as<double>();
        anchor.ball_x_tolerance_m = encoded["ball_x_tolerance_m"].as<double>();
        anchor.ball_y_tolerance_m = encoded["ball_y_tolerance_m"].as<double>();
        anchor.kp = encoded["kp"].as<double>();
        anchor.kd = encoded["kd"].as<double>();

        const YAML::Node key_times = encoded["key_times_s"];
        const YAML::Node parameters = encoded["parameters"];
        if (!key_times || !key_times.IsSequence() ||
            key_times.size() != kKeyCount ||
            !parameters || !parameters.IsSequence() ||
            parameters.size() != kParameterCount) {
            throw std::runtime_error(
                "procedural kick anchor has an invalid trajectory shape");
        }
        for (std::size_t i = 0; i < kKeyCount; ++i) {
            anchor.key_times_s[i] = key_times[i].as<double>();
            if (!std::isfinite(anchor.key_times_s[i]) ||
                (i > 0U &&
                 anchor.key_times_s[i] <= anchor.key_times_s[i - 1U])) {
                throw std::runtime_error(
                    "procedural kick key times must be finite and increasing");
            }
        }
        for (std::size_t i = 0; i < kParameterCount; ++i) {
            anchor.parameters[i] = parameters[i].as<double>();
            if (!std::isfinite(anchor.parameters[i]) ||
                anchor.parameters[i] < kParameterLower[i] ||
                anchor.parameters[i] > kParameterUpper[i]) {
                throw std::runtime_error(
                    "procedural kick parameter exceeds the training contract");
            }
        }
        const std::array<double, 12> scalars{
            anchor.target_distance_m,
            anchor.target_distance_tolerance_m,
            anchor.target_angle_deg,
            anchor.target_angle_tolerance_deg,
            anchor.requested_speed_mps,
            anchor.requested_speed_tolerance_mps,
            anchor.ball_local_x_m,
            anchor.ball_local_y_m,
            anchor.ball_x_tolerance_m,
            anchor.ball_y_tolerance_m,
            anchor.kp,
            anchor.kd,
        };
        if (!std::all_of(scalars.begin(), scalars.end(), [](double value) {
                return std::isfinite(value);
            }) ||
            anchor.target_distance_tolerance_m <= 0.0 ||
            anchor.target_angle_tolerance_deg <= 0.0 ||
            anchor.requested_speed_tolerance_mps <= 0.0 ||
            anchor.ball_x_tolerance_m <= 0.0 ||
            anchor.ball_y_tolerance_m <= 0.0 ||
            anchor.kp <= 0.0 || anchor.kd < 0.0) {
            throw std::runtime_error("procedural kick anchor has invalid limits");
        }
        anchors_.push_back(std::move(anchor));
    }
}

bool ProceduralKickRunner::begin(
    const world::WorldSnapshot& snapshot,
    const KickExecutionProfile& profile) {
    selected_ = nullptr;
    const bool supported_profile_kind =
        profile.kind == KickProfileKind::ProceduralContact ||
        (profile.kind == KickProfileKind::ParameterizedContact &&
         profile.mode == decision::KickMode::TargetedPass);
    if (!supported_profile_kind ||
        !snapshot.ball.position_valid || !ball_track_usable(snapshot) ||
        snapshot.self.position_m[2] <= world::kFallenHeightThresholdM ||
        !std::isfinite(profile.target_distance_m) ||
        !std::isfinite(profile.relative_target_angle_deg) ||
        !std::isfinite(profile.requested_speed_mps) ||
        math::norm2({snapshot.self.lin_vel_b[0], snapshot.self.lin_vel_b[1]}) >
            decision::kick_contract::kProceduralMaximumStartPlanarSpeedMps ||
        std::abs(snapshot.self.gyro_deg_s[0]) >
            decision::kick_contract::kProceduralMaximumStartTiltRateDegS ||
        std::abs(snapshot.self.gyro_deg_s[1]) >
            decision::kick_contract::kProceduralMaximumStartTiltRateDegS) {
        return false;
    }

    for (std::size_t i = 0; i < kJointCount; ++i) {
        const auto& name = robot_model_.readable_joint_names()[i];
        const auto position = snapshot.self.joint_positions_deg.find(name);
        if (position == snapshot.self.joint_positions_deg.end() ||
            !std::isfinite(position->second)) {
            return false;
        }
        captured_pose_rad_[i] = math::deg_to_rad(position->second);
        if (i >= 11U && std::abs(position->second) >
                decision::kick_contract::kProceduralMaximumStartLegPositionDeg) {
            return false;
        }
        const auto velocity = snapshot.self.joint_velocities_deg_s.find(name);
        if (i >= 11U &&
            (velocity == snapshot.self.joint_velocities_deg_s.end() ||
             !std::isfinite(velocity->second) ||
             std::abs(velocity->second) >
                 decision::kick_contract::kProceduralMaximumStartLegVelocityDegS)) {
            return false;
        }
    }

    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const std::array<double, 2> ball_delta_world{
        snapshot.ball.position_m[0] - snapshot.self.position_m[0],
        snapshot.ball.position_m[1] - snapshot.self.position_m[1],
    };
    const auto ball_local = math::rotate_2d(ball_delta_world, -yaw_deg);

    double best_score = std::numeric_limits<double>::infinity();
    const std::string requested_mode = mode_name(profile.mode);
    for (const Anchor& anchor : anchors_) {
        const double distance_error =
            profile.target_distance_m - anchor.target_distance_m;
        const double angle_error =
            profile.relative_target_angle_deg - anchor.target_angle_deg;
        const double speed_error =
            profile.requested_speed_mps - anchor.requested_speed_mps;
        const double ball_x_error = ball_local[0] - anchor.ball_local_x_m;
        const double ball_y_error = ball_local[1] - anchor.ball_local_y_m;
        if (anchor.mode != requested_mode ||
            std::abs(distance_error) > anchor.target_distance_tolerance_m ||
            std::abs(angle_error) > anchor.target_angle_tolerance_deg ||
            std::abs(speed_error) > anchor.requested_speed_tolerance_mps ||
            std::abs(ball_x_error) > anchor.ball_x_tolerance_m ||
            std::abs(ball_y_error) > anchor.ball_y_tolerance_m) {
            continue;
        }
        const double score =
            std::pow(distance_error / anchor.target_distance_tolerance_m, 2) +
            std::pow(angle_error / anchor.target_angle_tolerance_deg, 2) +
            std::pow(speed_error / anchor.requested_speed_tolerance_mps, 2) +
            std::pow(ball_x_error / anchor.ball_x_tolerance_m, 2) +
            std::pow(ball_y_error / anchor.ball_y_tolerance_m, 2);
        if (score < best_score) {
            best_score = score;
            selected_ = &anchor;
        }
    }
    return selected_ != nullptr;
}

std::array<double, ProceduralKickRunner::kJointCount>
ProceduralKickRunner::joint_delta_at(double elapsed_s) const {
    std::array<double, kJointCount> zero{};
    if (selected_ == nullptr || !std::isfinite(elapsed_s)) return zero;

    std::array<std::array<double, kJointCount>, kKeyCount> keys{};
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
    if (elapsed_s >= selected_->key_times_s.back()) {
        left = kKeyCount - 2U;
    } else {
        while (left + 1U < kKeyCount &&
               elapsed_s > selected_->key_times_s[left + 1U]) {
            ++left;
        }
    }
    const std::size_t right = std::min(left + 1U, kKeyCount - 1U);
    const double denominator =
        selected_->key_times_s[right] - selected_->key_times_s[left];
    const double fraction = smoothstep(
        (elapsed_s - selected_->key_times_s[left]) / denominator);
    std::array<double, kJointCount> result{};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        result[joint] =
            keys[left][joint] * (1.0 - fraction) +
            keys[right][joint] * fraction;
    }
    return result;
}

ProceduralKickStepResult ProceduralKickRunner::step(double elapsed_s) const {
    ProceduralKickStepResult result;
    if (selected_ == nullptr) return result;
    const auto delta = joint_delta_at(elapsed_s);
    const double capture_fraction = smoothstep(elapsed_s / kCaptureDurationS);
    result.joint_targets.reserve(kJointCount);
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        const double captured_base =
            captured_pose_rad_[joint] * (1.0 - capture_fraction);
        const double target_rad = std::clamp(
            captured_base + delta[joint],
            robot::t1_joint_limits::kLowerRad[joint],
            robot::t1_joint_limits::kUpperRad[joint]);
        result.joint_targets.push_back({
            robot_model_.readable_joint_names()[joint],
            math::rad_to_deg(target_rad),
            0.0,
            selected_->kp,
            selected_->kd,
            0.0,
        });
    }
    result.finished = elapsed_s >= selected_->key_times_s.back();
    return result;
}

std::string ProceduralKickRunner::active_anchor_name() const {
    return selected_ != nullptr ? selected_->name : std::string{};
}

double ProceduralKickRunner::duration_s() const {
    return selected_ != nullptr ? selected_->key_times_s.back() : 0.0;
}

}  // namespace behavior
