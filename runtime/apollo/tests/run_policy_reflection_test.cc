// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/behavior/policy_common.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool same(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (std::abs(left[i] - right[i]) > 1.0e-6F) return false;
    }
    return true;
}

}  // namespace

int main() {
    std::vector<float> action(23U);
    for (std::size_t i = 0; i < action.size(); ++i) {
        action[i] = static_cast<float>(i + 1U) / 10.0F;
    }
    const auto mirrored_action = behavior::mirror_run_policy_action(action);
    if (!same(
            behavior::mirror_run_policy_action(mirrored_action), action) ||
        std::abs(mirrored_action[2] + action[6]) > 1.0e-6F ||
        std::abs(mirrored_action[0] + action[0]) > 1.0e-6F ||
        std::abs(mirrored_action[1] - action[1]) > 1.0e-6F) {
        std::cerr << "run action reflection contract drifted\n";
        return 1;
    }

    std::vector<float> observation(80U);
    for (std::size_t i = 0; i < observation.size(); ++i) {
        observation[i] = static_cast<float>(i + 1U) / 100.0F;
    }
    const auto mirrored_observation =
        behavior::mirror_run_policy_observation(observation);
    if (!same(
            behavior::mirror_run_policy_observation(mirrored_observation),
            observation) ||
        std::abs(mirrored_observation[73] + observation[73]) > 1.0e-6F ||
        std::abs(mirrored_observation[74] + observation[74]) > 1.0e-6F ||
        std::abs(mirrored_observation[78] + observation[78]) > 1.0e-6F) {
        std::cerr << "run observation reflection contract drifted\n";
        return 1;
    }

    bool invalid_rejected = false;
    try {
        static_cast<void>(behavior::mirror_run_policy_observation(
            std::vector<float>(79U, 0.0F)));
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    if (!invalid_rejected) {
        std::cerr << "invalid run observation was accepted\n";
        return 1;
    }
    if (std::abs(
            behavior::decode_run_policy_target_rad(12U, -10.0F) -
            behavior::kRunJointUpperRad[12]) > 1.0e-6F ||
        std::abs(
            behavior::decode_run_policy_target_rad(18U, -10.0F) -
            behavior::kRunJointUpperRad[18]) > 1.0e-6F ||
        std::abs(
            behavior::decode_run_policy_target_rad(14U, -10.0F) -
            behavior::kRunJointLowerRad[14]) > 1.0e-6F) {
        std::cerr << "run target decoder did not enforce T1 limits\n";
        return 1;
    }
    return 0;
}
