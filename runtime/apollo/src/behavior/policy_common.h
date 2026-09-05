// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/math/math_utils.h"
#include "src/robot/joint_targets.h"
#include "src/robot/t1_robot_model.h"
#include "src/world/world_snapshot.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace behavior {

// Per-joint home pose (rad) for action decoding and observation centering.
// The real-robot deployment (5.5 baseline) uses these values; the mjlab
// HOME_KEYFRAME from t1_constants.py is for the simulator only and does not
// apply here.
inline constexpr std::array<float, 23> kDefaultPosRad{
    0.0F,  0.0F,  0.0F,  -1.4F, 0.0F,  -0.4F, 0.0F, 1.4F,
    0.0F,  0.4F,  0.0F,  -0.2F, 0.0F,   0.0F, 0.4F, -0.2F,
    0.0F, -0.2F,  0.0F,   0.0F, 0.4F,  -0.2F, 0.0F,
};

// Phase-v2 run-policy frame. These values deliberately live next to the
// reflection operators so training evaluation and runtime deployment cannot
// silently use different joint conventions.
inline constexpr std::array<float, 23> kRunNominalTrainingPoseRad{
    0.0F, 0.0F, 0.0F, 1.4F, 0.0F, -0.4F, 0.0F, -1.4F,
    0.0F, 0.4F, 0.0F, -0.4F, 0.0F, 0.0F, 0.8F, -0.4F,
    0.0F, 0.4F, 0.0F, 0.0F, -0.8F, 0.4F, 0.0F,
};
inline constexpr std::array<float, 23> kRunTrainingToServerSign{
    1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, -1.0F, -1.0F,
    1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F,
    -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
};
// Exact RCSSServerMJ T1 joint ranges, in physical/server radians. The run
// training environment and CPU evaluator apply these after action decoding;
// runtime must do the same before emitting motor targets.
inline constexpr std::array<float, 23> kRunJointLowerRad{
    -1.57F, -0.35F, -3.31F, -1.74F, -2.27F, -2.44F, -3.31F,
    -1.57F, -2.27F, 0.0F, -1.57F, -1.80F, -0.20F, -1.0F,
    0.0F, -0.87F, -0.44F, -1.80F, -1.57F, -1.0F, 0.0F,
    -0.87F, -0.44F,
};
inline constexpr std::array<float, 23> kRunJointUpperRad{
    1.57F, 1.22F, 1.22F, 1.57F, 2.27F, 0.0F, 1.22F,
    1.74F, 2.27F, 2.44F, 1.57F, 1.57F, 1.57F, 1.0F,
    2.34F, 0.35F, 0.44F, 1.57F, 0.20F, 1.0F, 2.34F,
    0.35F, 0.44F,
};

inline float decode_run_policy_target_rad(
    std::size_t joint_index,
    float action,
    float action_scale = 0.5F) {
    if (joint_index >= kRunNominalTrainingPoseRad.size()) {
        throw std::out_of_range("run joint index is out of range");
    }
    const float target_server =
        (kRunNominalTrainingPoseRad[joint_index] + action_scale * action) *
        kRunTrainingToServerSign[joint_index];
    return std::clamp(
        target_server,
        kRunJointLowerRad[joint_index],
        kRunJointUpperRad[joint_index]);
}
inline constexpr std::array<std::size_t, 23> kRunMirrorSource{
    0U, 1U, 6U, 7U, 8U, 9U, 2U, 3U, 4U, 5U, 10U, 17U,
    18U, 19U, 20U, 21U, 22U, 11U, 12U, 13U, 14U, 15U, 16U,
};
inline constexpr std::array<float, 23> kRunMirrorTrainingFactor{
    -1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
    -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
    -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
};

/// Reflects one phase-v2 run-policy action in the policy's training frame.
inline std::vector<float> mirror_run_policy_action(
    const std::vector<float>& action) {
    if (action.size() != kRunMirrorSource.size()) {
        throw std::invalid_argument("run action must contain 23 values");
    }
    std::vector<float> mirrored(action.size());
    for (std::size_t i = 0; i < mirrored.size(); ++i) {
        mirrored[i] = action[kRunMirrorSource[i]] * kRunMirrorTrainingFactor[i];
    }
    return mirrored;
}

/// Reflects one 78/80-value run observation and exchanges gait phase sides.
inline std::vector<float> mirror_run_policy_observation(
    const std::vector<float>& observation) {
    if (observation.size() != 78U && observation.size() != 80U) {
        throw std::invalid_argument("run observation must contain 78 or 80 values");
    }
    std::vector<float> mirrored(observation);
    for (std::size_t joint = 0; joint < kRunMirrorSource.size(); ++joint) {
        for (std::size_t feature = 0; feature < 3U; ++feature) {
            mirrored[3U * joint + feature] =
                observation[3U * kRunMirrorSource[joint] + feature] *
                kRunMirrorTrainingFactor[joint];
        }
    }
    constexpr std::array<float, 3> gyro_factor{-1.0F, 1.0F, -1.0F};
    constexpr std::array<float, 3> command_factor{1.0F, -1.0F, -1.0F};
    constexpr std::array<float, 3> gravity_factor{1.0F, -1.0F, 1.0F};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        mirrored[69U + axis] = observation[69U + axis] * gyro_factor[axis];
        mirrored[72U + axis] = observation[72U + axis] * command_factor[axis];
        mirrored[75U + axis] = observation[75U + axis] * gravity_factor[axis];
    }
    if (observation.size() == 80U) {
        mirrored[78] = -observation[78];
        mirrored[79] = -observation[79];
    }
    return mirrored;
}

/// Builds angular-velocity and projected-gravity observations in body frame.
inline std::array<float, 6> build_imu_obs(const world::WorldSnapshot& snapshot) {
    const auto gravity_body =
        math::rotate_vec_by_quaternion({0.0, 0.0, -1.0}, math::quaternion_conjugate(snapshot.self.orientation_wxyz));
    return {
        static_cast<float>(math::deg_to_rad(snapshot.self.gyro_deg_s[0])),
        static_cast<float>(math::deg_to_rad(snapshot.self.gyro_deg_s[1])),
        static_cast<float>(math::deg_to_rad(snapshot.self.gyro_deg_s[2])),
        static_cast<float>(gravity_body[0]),
        static_cast<float>(gravity_body[1]),
        static_cast<float>(gravity_body[2]),
    };
}

/// Writes joint position, velocity, and previous-action policy observations.
inline void fill_joint_obs(
    float* out_pos, float* out_vel, float* out_prev_act,
    const world::WorldSnapshot& snapshot,
    const robot::T1RobotModel& robot_model,
    const std::vector<float>& previous_action) {
    const auto& joint_names = robot_model.readable_joint_names();
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
        const auto pos_it = snapshot.self.joint_positions_deg.find(joint_names[i]);
        const double pos_deg = pos_it == snapshot.self.joint_positions_deg.end() ? 0.0 : pos_it->second;
        out_pos[i] = static_cast<float>(math::deg_to_rad(pos_deg) - kDefaultPosRad[i]);
    }
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
        const auto vel_it = snapshot.self.joint_velocities_deg_s.find(joint_names[i]);
        const double vel_deg = vel_it == snapshot.self.joint_velocities_deg_s.end() ? 0.0 : vel_it->second;
        out_vel[i] = static_cast<float>(math::deg_to_rad(vel_deg));
    }
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
        out_prev_act[i] = previous_action[i];
    }
}

/// Converts normalized policy actions into position-controlled joint targets.
inline robot::JointTargets decode_action_base(
    const std::vector<float>& action,
    const robot::T1RobotModel& robot_model,
    float action_scale = 0.25F) {
    robot::JointTargets joint_targets;
    const auto& joint_names = robot_model.readable_joint_names();
    joint_targets.reserve(joint_names.size());
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
        const double target_rad = static_cast<double>(kDefaultPosRad[i] + action[i] * action_scale);
        joint_targets.push_back({
            joint_names[i],
            math::rad_to_deg(target_rad),
            0.0,
            robot_model.joint_kp(joint_names[i]),
            robot_model.joint_kd(joint_names[i]),
            0.0,
        });
    }
    return joint_targets;
}

/// Appends one observation to a fixed-length, oldest-first history buffer.
inline void advance_history(
    std::vector<float>& history,
    const std::vector<float>& step_obs,
    int step_obs_dim,
    int history_length,
    bool reset) {
    if (reset) {
        for (int h = 0; h < history_length; ++h) {
            std::copy(step_obs.begin(), step_obs.end(), history.begin() + h * step_obs_dim);
        }
    } else {
        if (history_length > 1) {
            std::move(
                history.begin() + step_obs_dim,
                history.end(),
                history.begin());
        }
        std::copy(
            step_obs.begin(),
            step_obs.end(),
            history.end() - step_obs_dim);
    }
}

}  // namespace behavior
