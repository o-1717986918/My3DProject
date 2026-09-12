// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/dynamic_pass_runner.h"

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
constexpr float kSelectorThreshold = 0.96F;
constexpr int kConfirmationFrames = 2;
constexpr double kReleaseBallFreshnessS = 0.10;
constexpr double kNominalGaitFrequencyHz = 1.6;
constexpr double kNeutralMagnitudeRad = 0.02;
constexpr double kSupportSwitchSine = 0.15;
constexpr std::array<double, 6> kKeyTimes{0.0, 0.18, 0.34, 0.54, 0.76, 1.20};
constexpr std::array<int, 10> kPrototypeRolloutIds{302, 117, 4, 84, 99, 107, 43, 79, 65, 17};
constexpr std::array<double, kJointCount> kKickActionScaleRad{
    0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.20, 0.20, 0.15, 0.35, 0.25, 0.25, 0.45, 0.25,
    0.20, 0.35, 0.25, 0.25, 0.45, 0.25, 0.20,
};

// Frozen order matches the selector outputs. Values are the accepted
// fourteen-parameter teacher rows recorded in the asset provenance file.
constexpr std::array<std::array<double, 14>, 10> kPrototypeParameters{{
    {{-0.01525139, -0.08081315, -0.00740486, -0.00918799, -0.32853943, -0.66942644, 0.74557120, 0.13354625, -0.14192195, 0.22820915, -0.37221354, 0.18193147, 0.39421314, -0.28855804}},
    {{0.00918042, -0.05674657, 0.08044184, 0.12201948, -0.37835607, -0.64818686, 0.90702653, -0.03172085, -0.14275746, 0.20548317, -0.34681413, -0.07010191, 0.42492211, -0.32737646}},
    {{-0.07810739, 0.18151353, 0.19064227, 0.02859146, -0.47873834, -0.81816888, 1.09884787, 0.20272595, 0.03538802, 0.09161026, -0.36008984, -0.41411686, 0.44235754, -0.19193180}},
    {{0.02796975, -0.07858982, 0.13037452, -0.31784868, -0.43007362, -0.81474537, 0.05821725, 0.39867592, 0.34436229, 0.27465042, -0.46540225, -0.17346750, 0.29477319, -0.26154158}},
    {{-0.04218716, -0.07589049, -0.02720311, -0.07062093, -0.41534364, -0.89058924, 0.79595214, 0.14430849, -0.11186865, 0.25821224, -0.31681228, -0.25289449, 0.40470237, -0.24811527}},
    {{0.14074804, 0.00934197, 0.38215378, -0.46544045, -0.39745080, -0.84070283, 0.41343638, 0.11929701, 0.28863597, 0.17330882, -0.28781301, -0.25102204, 0.40292665, -0.21173345}},
    {{-0.03227513, 0.17428787, 0.07669497, 0.26635519, -0.48785776, -0.86625141, 1.17584693, 0.20223701, 0.26017201, -0.00159158, -0.37834969, -0.24205349, 0.41576573, 0.05669852}},
    {{0.01364377, -0.06955606, 0.08036270, 0.11145449, -0.50203502, -0.66156733, 0.89178240, -0.00496340, -0.16648939, 0.19813035, -0.34195593, -0.10031623, 0.21906446, -0.14632015}},
    {{-0.06687591, 0.17324013, 0.03355756, 0.22084114, -0.53499049, -0.79430616, 1.02704823, 0.34562504, 0.13406256, 0.00860855, -0.38314834, -0.37528443, 0.30675778, -0.18181908}},
    {{-0.03100448, -0.10645301, -0.03303166, 0.13965136, -0.40579990, -0.79496008, 0.24881774, 0.31227469, -0.03208317, 0.27840254, -0.35401016, -0.09580568, 0.21010955, -0.24599387}},
}};

bool finite_joint_state(
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

double smoothstep(double value) {
    const double x = std::clamp(value, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

std::array<double, kJointCount> raw_keyframe(
    const std::array<double, 14>& parameters,
    std::size_t key_index) {
    std::array<double, kJointCount> key{};
    if (key_index >= 1U && key_index <= 3U) {
        key[12] = parameters[0];
        key[18] = parameters[0];
        key[16] = parameters[1];
        key[22] = parameters[1];
    }
    if (key_index == 2U) {
        key[17] = parameters[2];
        key[20] = parameters[3];
        key[21] = parameters[4];
        key[10] = parameters[9];
        key[2] = parameters[10];
        key[6] = parameters[11];
    }
    if (key_index == 3U) {
        key[17] = parameters[5];
        key[20] = parameters[6];
        key[21] = parameters[7];
        key[19] = parameters[8];
        key[18] += parameters[12];
        key[22] += parameters[13];
        key[10] = parameters[9];
        key[2] = parameters[10];
        key[6] = parameters[11];
    }
    if (key_index == 4U) {
        key = raw_keyframe(parameters, 3U);
        for (double& value : key) value *= 0.35;
    }
    return key;
}

}  // namespace

DynamicPassRunner::DynamicPassRunner(const std::filesystem::path& selector_path)
    : selector_(selector_path, OnnxModelContract{{1, 98}, {1, 10}}) {}

bool DynamicPassRunner::release_geometry(const world::WorldSnapshot& snapshot) {
    const bool ball_fresh = snapshot.ball.position_valid &&
        (snapshot.ball.visible ||
         snapshot.ball.position_age_s <= kReleaseBallFreshnessS);
    if (!ball_fresh ||
        snapshot.self.position_m[2] <= world::kFallenHeightThresholdM) {
        return false;
    }
    const double yaw_deg = world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
        snapshot.self.orientation_wxyz);
    const auto ball_local = math::rotate_2d(
        {snapshot.ball.position_m[0] - snapshot.self.position_m[0],
         snapshot.ball.position_m[1] - snapshot.self.position_m[1]},
        -yaw_deg);
    return ball_local[0] >= 0.25 && ball_local[0] <= 0.50 &&
        ball_local[1] >= -0.15 && ball_local[1] <= 0.15;
}

std::vector<float> DynamicPassRunner::build_selector_observation(
    const world::WorldSnapshot& snapshot,
    const robot::T1RobotModel& robot_model) {
    if (!finite_joint_state(snapshot, robot_model)) {
        throw std::runtime_error("dynamic-pass selector lacks joint state");
    }
    std::vector<float> observation;
    observation.reserve(kObservationSize);
    const auto imu = build_imu_obs(snapshot);
    observation.insert(observation.end(), imu.begin(), imu.end());

    std::vector<float> positions(kJointCount, 0.0F);
    std::vector<float> velocities(kJointCount, 0.0F);
    std::vector<float> previous(kJointCount, 0.0F);
    const std::vector<float> zero_action(kJointCount, 0.0F);
    fill_joint_obs(
        positions.data(), velocities.data(), previous.data(), snapshot,
        robot_model, zero_action);
    observation.insert(observation.end(), positions.begin(), positions.end());
    observation.insert(observation.end(), velocities.begin(), velocities.end());
    observation.insert(observation.end(), previous.begin(), previous.end());

    const double yaw_deg = world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
        snapshot.self.orientation_wxyz);
    const auto ball_local = math::rotate_2d(
        {snapshot.ball.position_m[0] - snapshot.self.position_m[0],
         snapshot.ball.position_m[1] - snapshot.self.position_m[1]},
        -yaw_deg);
    const std::array<double, 3> ball_velocity_world = snapshot.ball.velocity_valid
        ? snapshot.ball.velocity_mps
        : std::array<double, 3>{0.0, 0.0, 0.0};
    const auto ball_velocity_local = math::rotate_2d(
        {ball_velocity_world[0], ball_velocity_world[1]}, -yaw_deg);
    observation.insert(observation.end(), {
        static_cast<float>(ball_local[0]),
        static_cast<float>(ball_local[1]),
        static_cast<float>(snapshot.ball.position_m[2] - snapshot.self.position_m[2]),
        static_cast<float>(ball_velocity_local[0] - snapshot.self.lin_vel_b[0]),
        static_cast<float>(ball_velocity_local[1] - snapshot.self.lin_vel_b[1]),
        static_cast<float>(ball_velocity_world[2] - snapshot.self.lin_vel_b[2]),
    });

    const auto target_local = math::rotate_2d({1.0, 0.0}, -yaw_deg);
    observation.insert(observation.end(), {
        static_cast<float>(target_local[0]),
        static_cast<float>(target_local[1]),
        2.0F, 1.43F, 0.8F,
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F,
        0.0F, 1.0F,
    });

    constexpr std::size_t left_hip_pitch = 11U;
    constexpr std::size_t right_hip_pitch = 17U;
    const double position_signal = positions[right_hip_pitch] - positions[left_hip_pitch];
    const double velocity_signal =
        (velocities[right_hip_pitch] - velocities[left_hip_pitch]) /
        (2.0 * math::kPi * kNominalGaitFrequencyHz);
    const double magnitude = std::hypot(position_signal, velocity_signal);
    std::array<float, 2> phase{0.0F, 1.0F};
    if (magnitude >= kNeutralMagnitudeRad) {
        phase = {
            static_cast<float>(position_signal / magnitude),
            static_cast<float>(velocity_signal / magnitude),
        };
    }
    observation.insert(observation.end(), phase.begin(), phase.end());
    if (phase[0] > kSupportSwitchSine) {
        observation.insert(observation.end(), {1.0F, 0.0F, 0.0F});
    } else if (phase[0] < -kSupportSwitchSine) {
        observation.insert(observation.end(), {0.0F, 0.0F, 1.0F});
    } else {
        observation.insert(observation.end(), {0.0F, 1.0F, 0.0F});
    }

    if (observation.size() != kObservationSize ||
        !std::all_of(observation.begin(), observation.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error("dynamic-pass selector observation is invalid");
    }
    return observation;
}

DynamicPassActivation DynamicPassRunner::consider(
    const world::WorldSnapshot& snapshot,
    bool eligible) {
    if (!eligible) {
        reset();
        return {};
    }
    if (active_) {
        return {};
    }
    release_candidate_ = release_geometry(snapshot);
    if (!release_candidate_) {
        streak_.fill(0);
        armed_ = true;
        max_probability_ = -1.0F;
        best_prototype_rollout_id_ = -1;
        return {};
    }
    if (!armed_) {
        return {};
    }

    try {
        const auto probabilities = selector_.run(
            build_selector_observation(snapshot, robot_model_));
        if (probabilities.size() != kPrototypeCount) {
            throw std::runtime_error("dynamic-pass selector output size mismatch");
        }
        const auto best = std::max_element(probabilities.begin(), probabilities.end());
        const std::size_t best_index = static_cast<std::size_t>(
            std::distance(probabilities.begin(), best));
        max_probability_ = *best;
        best_prototype_rollout_id_ = kPrototypeRolloutIds[best_index];
        for (std::size_t i = 0; i < kPrototypeCount; ++i) {
            streak_[i] = std::isfinite(probabilities[i]) &&
                    probabilities[i] >= kSelectorThreshold
                ? streak_[i] + 1
                : 0;
        }
        std::size_t choice = kPrototypeCount;
        for (std::size_t i = 0; i < kPrototypeCount; ++i) {
            if (streak_[i] >= kConfirmationFrames &&
                (choice == kPrototypeCount || probabilities[i] > probabilities[choice])) {
                choice = i;
            }
        }
        if (choice == kPrototypeCount) {
            return {};
        }
        active_ = true;
        selected_prototype_ = choice;
        selected_confidence_ = probabilities[choice];
        start_time_s_ = snapshot.server_time;
        streak_.fill(0);
        return {true, kPrototypeRolloutIds[choice], selected_confidence_};
    } catch (const std::exception&) {
        streak_.fill(0);
        max_probability_ = -1.0F;
        best_prototype_rollout_id_ = -1;
        return {};
    }
}

int DynamicPassRunner::max_confirmation_streak() const {
    return *std::max_element(streak_.begin(), streak_.end());
}

double DynamicPassRunner::elapsed_s(double server_time) const {
    return active_ ? std::max(0.0, server_time - start_time_s_) : 0.0;
}

std::array<double, kJointCount> DynamicPassRunner::prototype_delta_rad(
    std::size_t prototype_index,
    double elapsed) {
    if (prototype_index >= kPrototypeCount || !std::isfinite(elapsed)) {
        throw std::invalid_argument("dynamic-pass prototype request is invalid");
    }
    const double time = std::clamp(elapsed, kKeyTimes.front(), kKeyTimes.back());
    std::size_t right = 1U;
    while (right < kKeyTimes.size() && time > kKeyTimes[right]) ++right;
    if (right >= kKeyTimes.size()) right = kKeyTimes.size() - 1U;
    const std::size_t left = right - 1U;
    const double span = kKeyTimes[right] - kKeyTimes[left];
    const double weight = span > 0.0
        ? smoothstep((time - kKeyTimes[left]) / span)
        : 1.0;
    const auto left_key = raw_keyframe(kPrototypeParameters[prototype_index], left);
    const auto right_key = raw_keyframe(kPrototypeParameters[prototype_index], right);
    std::array<double, kJointCount> delta{};
    for (std::size_t i = 0; i < kJointCount; ++i) {
        const double raw = left_key[i] * (1.0 - weight) + right_key[i] * weight;
        const double bounded_from_default =
            std::clamp(
                static_cast<double>(kDefaultPosRad[i]) + raw,
                robot::t1_joint_limits::kLowerRad[i],
                robot::t1_joint_limits::kUpperRad[i]) -
            static_cast<double>(kDefaultPosRad[i]);
        delta[i] = std::clamp(
            bounded_from_default,
            -kKickActionScaleRad[i],
            kKickActionScaleRad[i]);
    }
    return delta;
}

robot::JointTargets DynamicPassRunner::compose_targets(
    const robot::JointTargets& stable_walk_targets,
    const std::array<double, kJointCount>& delta_rad,
    const robot::T1RobotModel& robot_model) {
    if (stable_walk_targets.size() != kJointCount) {
        throw std::runtime_error("dynamic-pass base target size mismatch");
    }
    robot::JointTargets targets = stable_walk_targets;
    const auto& names = robot_model.readable_joint_names();
    for (std::size_t i = 0; i < kJointCount; ++i) {
        if (targets[i].joint_name != names[i] || !std::isfinite(delta_rad[i])) {
            throw std::runtime_error("dynamic-pass joint order mismatch");
        }
        targets[i].q_deg = math::rad_to_deg(std::clamp(
            math::deg_to_rad(targets[i].q_deg) + delta_rad[i],
            robot::t1_joint_limits::kLowerRad[i],
            robot::t1_joint_limits::kUpperRad[i]));
    }
    return targets;
}

DynamicPassStepResult DynamicPassRunner::step(
    const world::WorldSnapshot& snapshot,
    const robot::JointTargets& stable_walk_targets) {
    if (!active_) {
        return {};
    }
    DynamicPassStepResult result;
    result.prototype_rollout_id = kPrototypeRolloutIds[selected_prototype_];
    result.confidence = selected_confidence_;
    try {
        const double elapsed = elapsed_s(snapshot.server_time);
        result.joint_targets = compose_targets(
            stable_walk_targets,
            prototype_delta_rad(selected_prototype_, elapsed),
            robot_model_);
        result.valid = true;
        result.finished = elapsed >= kDurationS;
        if (result.finished) {
            active_ = false;
            armed_ = false;
            streak_.fill(0);
        }
    } catch (const std::exception&) {
        active_ = false;
        armed_ = false;
        streak_.fill(0);
    }
    return result;
}

void DynamicPassRunner::reset() {
    streak_.fill(0);
    active_ = false;
    armed_ = true;
    selected_prototype_ = 0U;
    selected_confidence_ = 0.0F;
    start_time_s_ = 0.0;
    release_candidate_ = false;
    max_probability_ = -1.0F;
    best_prototype_rollout_id_ = -1;
}

}  // namespace behavior
