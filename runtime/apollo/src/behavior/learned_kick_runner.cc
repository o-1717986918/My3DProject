// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/learned_kick_runner.h"

#include "src/behavior/policy_common.h"
#include "src/math/math_utils.h"
#include "src/robot/t1_joint_limits.h"
#include "src/world/frame_normalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace behavior {

namespace {

constexpr std::size_t kJointCount = 23U;
constexpr std::size_t kObservationSize = 98U;
constexpr double kDesiredArrivalSpeedMps = 0.8;
constexpr double kNominalGaitFrequencyHz = 1.6;
constexpr double kNeutralPhaseMagnitudeRad = 0.02;
constexpr double kSupportSwitchSine = 0.15;
// The currently mountable v3 actors were trained on the fixed 2 m transition
// corpus. Keep their active/shadow support inside that measured input slice;
// a future companion manifest will replace these candidate-specific bounds.
constexpr double kMinimumTargetDistanceM = 1.90;
constexpr double kMaximumTargetDistanceM = 2.10;
constexpr double kMaximumTargetAngleDeg = 12.0;
constexpr double kMinimumBallLocalXM = 0.30;
constexpr double kMaximumBallLocalXM = 0.39;
constexpr double kMinimumBallLocalYM = -0.03;
constexpr double kMaximumBallLocalYM = 0.05;

constexpr std::array<double, kJointCount> kKickActionScaleRad{
    0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.20, 0.20, 0.15, 0.35, 0.25, 0.25, 0.45, 0.25,
    0.20, 0.35, 0.25, 0.25, 0.45, 0.25, 0.20,
};

bool ball_track_usable(const world::WorldSnapshot& snapshot) {
    return snapshot.ball.position_valid &&
        (snapshot.ball.visible ||
         snapshot.ball.position_age_s <= world::kBallPositionFreshLifetimeS ||
         (snapshot.ball.near_contact_track &&
          snapshot.ball.position_age_s <= world::kNearContactBallTrackLifetimeS));
}

bool snapshot_has_joint_state(
    const world::WorldSnapshot& snapshot,
    const robot::T1RobotModel& robot_model) {
    for (const auto& name : robot_model.readable_joint_names()) {
        const auto position = snapshot.self.joint_positions_deg.find(name);
        const auto velocity = snapshot.self.joint_velocities_deg_s.find(name);
        if (position == snapshot.self.joint_positions_deg.end() ||
            velocity == snapshot.self.joint_velocities_deg_s.end() ||
            !std::isfinite(position->second) ||
            !std::isfinite(velocity->second)) {
            return false;
        }
    }
    return true;
}

std::array<float, 2> locomotion_phase(
    const std::vector<float>& joint_positions,
    const std::vector<float>& joint_velocities) {
    constexpr std::size_t left_hip_pitch = 11U;
    constexpr std::size_t right_hip_pitch = 17U;
    const double position_signal =
        joint_positions[right_hip_pitch] - joint_positions[left_hip_pitch];
    const double velocity_signal =
        (joint_velocities[right_hip_pitch] -
         joint_velocities[left_hip_pitch]) /
        (2.0 * math::kPi * kNominalGaitFrequencyHz);
    const double magnitude = std::hypot(position_signal, velocity_signal);
    if (magnitude < kNeutralPhaseMagnitudeRad) {
        return {0.0F, 1.0F};
    }
    return {
        static_cast<float>(position_signal / magnitude),
        static_cast<float>(velocity_signal / magnitude),
    };
}

std::array<float, 3> support_hint(const std::array<float, 2>& phase) {
    if (phase[0] > kSupportSwitchSine) return {1.0F, 0.0F, 0.0F};
    if (phase[0] < -kSupportSwitchSine) return {0.0F, 0.0F, 1.0F};
    return {0.0F, 1.0F, 0.0F};
}

}  // namespace

LearnedKickRunner::LearnedKickRunner(const std::filesystem::path& model_path)
    : session_(model_path, OnnxModelContract{{1, 98}, {1, 23}}),
      previous_action_(kJointCount, 0.0F),
      last_observation_(kObservationSize, 0.0F) {}

bool LearnedKickRunner::begin(
    const world::WorldSnapshot& snapshot,
    const KickExecutionProfile& profile) {
    active_ = false;
    failed_ = false;
    std::fill(previous_action_.begin(), previous_action_.end(), 0.0F);
    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const auto ball_local = math::rotate_2d(
        {
            snapshot.ball.position_m[0] - snapshot.self.position_m[0],
            snapshot.ball.position_m[1] - snapshot.self.position_m[1],
        },
        -yaw_deg);
    if (profile.kind != KickProfileKind::ParameterizedContact ||
        profile.mode != decision::KickMode::TargetedPass ||
        !ball_track_usable(snapshot) ||
        snapshot.self.position_m[2] <= world::kFallenHeightThresholdM ||
        !snapshot_has_joint_state(snapshot, robot_model_) ||
        !std::isfinite(profile.target_distance_m) ||
        !std::isfinite(profile.relative_target_angle_deg) ||
        !std::isfinite(profile.requested_speed_mps) ||
        profile.target_distance_m < kMinimumTargetDistanceM ||
        profile.target_distance_m > kMaximumTargetDistanceM ||
        std::abs(profile.relative_target_angle_deg) >
            kMaximumTargetAngleDeg ||
        ball_local[0] < kMinimumBallLocalXM ||
        ball_local[0] > kMaximumBallLocalXM ||
        ball_local[1] < kMinimumBallLocalYM ||
        ball_local[1] > kMaximumBallLocalYM) {
        return false;
    }
    active_ = true;
    return true;
}

std::vector<float> LearnedKickRunner::build_observation(
    const world::WorldSnapshot& snapshot,
    const KickExecutionProfile& profile,
    double elapsed_s,
    const std::vector<float>& previous_action,
    const robot::T1RobotModel& robot_model) {
    if (previous_action.size() != kJointCount ||
        !snapshot_has_joint_state(snapshot, robot_model)) {
        throw std::runtime_error("learned kick observation lacks joint state");
    }

    std::vector<float> observation;
    observation.reserve(kObservationSize);
    const auto imu = build_imu_obs(snapshot);
    observation.insert(observation.end(), imu.begin(), imu.end());

    std::vector<float> joint_positions(kJointCount, 0.0F);
    std::vector<float> joint_velocities(kJointCount, 0.0F);
    std::vector<float> copied_previous_action(kJointCount, 0.0F);
    fill_joint_obs(
        joint_positions.data(),
        joint_velocities.data(),
        copied_previous_action.data(),
        snapshot,
        robot_model,
        previous_action);
    observation.insert(
        observation.end(), joint_positions.begin(), joint_positions.end());
    observation.insert(
        observation.end(), joint_velocities.begin(), joint_velocities.end());
    observation.insert(
        observation.end(),
        copied_previous_action.begin(),
        copied_previous_action.end());

    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    const auto ball_local_xy = math::rotate_2d(
        {
            snapshot.ball.position_m[0] - snapshot.self.position_m[0],
            snapshot.ball.position_m[1] - snapshot.self.position_m[1],
        },
        -yaw_deg);
    const std::array<double, 3> ball_velocity_world =
        snapshot.ball.velocity_valid
        ? snapshot.ball.velocity_mps
        : std::array<double, 3>{0.0, 0.0, 0.0};
    const auto ball_velocity_local_xy = math::rotate_2d(
        {ball_velocity_world[0], ball_velocity_world[1]}, -yaw_deg);
    observation.push_back(static_cast<float>(ball_local_xy[0]));
    observation.push_back(static_cast<float>(ball_local_xy[1]));
    observation.push_back(static_cast<float>(
        snapshot.ball.position_m[2] - snapshot.self.position_m[2]));
    observation.push_back(static_cast<float>(
        ball_velocity_local_xy[0] - snapshot.self.lin_vel_b[0]));
    observation.push_back(static_cast<float>(
        ball_velocity_local_xy[1] - snapshot.self.lin_vel_b[1]));
    observation.push_back(static_cast<float>(
        ball_velocity_world[2] - snapshot.self.lin_vel_b[2]));

    const double target_angle_rad =
        math::deg_to_rad(profile.relative_target_angle_deg);
    observation.push_back(static_cast<float>(std::cos(target_angle_rad)));
    observation.push_back(static_cast<float>(std::sin(target_angle_rad)));
    observation.push_back(static_cast<float>(profile.target_distance_m));
    observation.push_back(static_cast<float>(profile.requested_speed_mps));
    observation.push_back(static_cast<float>(kDesiredArrivalSpeedMps));
    observation.insert(observation.end(), {1.0F, 0.0F, 0.0F});
    observation.push_back(static_cast<float>(std::max(
        0.0,
        std::isfinite(snapshot.ball.position_age_s)
            ? snapshot.ball.position_age_s
            : world::kNearContactBallTrackLifetimeS)));
    observation.push_back(ball_track_usable(snapshot) ? 1.0F : 0.0F);

    const double progress = std::clamp(elapsed_s / kDurationS, 0.0, 1.0);
    const double action_phase = math::kPi * progress;
    observation.push_back(static_cast<float>(std::sin(action_phase)));
    observation.push_back(static_cast<float>(std::cos(action_phase)));
    const auto gait_phase = locomotion_phase(joint_positions, joint_velocities);
    const auto support = support_hint(gait_phase);
    observation.insert(observation.end(), gait_phase.begin(), gait_phase.end());
    observation.insert(observation.end(), support.begin(), support.end());

    if (observation.size() != kObservationSize ||
        !std::all_of(observation.begin(), observation.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error("learned kick observation contract failed");
    }
    return observation;
}

robot::JointTargets LearnedKickRunner::compose_targets(
    const robot::JointTargets& stable_walk_targets,
    const std::vector<float>& action,
    const robot::T1RobotModel& robot_model) {
    if (stable_walk_targets.size() != kJointCount ||
        action.size() != kJointCount) {
        throw std::runtime_error("learned kick action contract failed");
    }
    robot::JointTargets result = stable_walk_targets;
    const auto& names = robot_model.readable_joint_names();
    for (std::size_t i = 0; i < kJointCount; ++i) {
        if (result[i].joint_name != names[i] || !std::isfinite(action[i])) {
            throw std::runtime_error("learned kick joint order is incompatible");
        }
        const double target_rad = std::clamp(
            math::deg_to_rad(result[i].q_deg) +
                std::clamp(static_cast<double>(action[i]), -1.0, 1.0) *
                    kKickActionScaleRad[i],
            robot::t1_joint_limits::kLowerRad[i],
            robot::t1_joint_limits::kUpperRad[i]);
        result[i].q_deg = math::rad_to_deg(target_rad);
    }
    return result;
}

LearnedKickStepResult LearnedKickRunner::step(
    const world::WorldSnapshot& snapshot,
    const KickExecutionProfile& profile,
    double elapsed_s,
    const robot::JointTargets& stable_walk_targets) {
    LearnedKickStepResult result;
    if (!active_ || failed_ || !ball_track_usable(snapshot) ||
        snapshot.self.position_m[2] <= world::kFallenHeightThresholdM) {
        return result;
    }
    try {
        last_observation_ = build_observation(
            snapshot, profile, elapsed_s, previous_action_, robot_model_);
        auto action = session_.run(last_observation_);
        for (const float value : action) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "learned kick actor returned non-finite action");
            }
            result.maximum_absolute_action = std::max(
                result.maximum_absolute_action, std::abs(value));
        }
        for (float& value : action) {
            value = std::clamp(value, -1.0F, 1.0F);
        }
        result.joint_targets = compose_targets(
            stable_walk_targets, action, robot_model_);
        previous_action_ = std::move(action);
        result.valid = true;
        result.finished = elapsed_s >= kDurationS;
        return result;
    } catch (const std::exception&) {
        failed_ = true;
        active_ = false;
        return result;
    }
}

}  // namespace behavior
