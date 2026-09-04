// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/kick_contract.h"
#include "src/strategy/action_capability.h"

#include <iostream>

namespace {

strategy::CooperativeAction make_pass(double distance_m, double speed_mps) {
    strategy::CooperativeAction action;
    action.category = strategy::ActionCategory::Pass;
    action.start_ball_point_m = {1.0, -2.0};
    action.target_point_m = {1.0 + distance_m, -2.0};
    action.requested_ball_speed_mps = speed_mps;
    return action;
}

}  // namespace

int main() {
    using namespace decision::kick_contract;
    const strategy::ActionCapabilityRegistry disabled(false);
    const strategy::ActionCapabilityRegistry enabled(true);
    const auto nominal = make_pass(2.0, 1.43);

    if (disabled.executable(nominal, 0.0) ||
        !enabled.executable(nominal, 0.0)) {
        std::cerr << "targeted pass feature gate is incorrect\n";
        return 1;
    }

    if (!enabled.executable(
            make_pass(kParameterizedPassMinimumTargetDistanceM,
                      kParameterizedPassRequestedSpeedMps),
            -kParameterizedPassMaximumTargetAngleDeg) ||
        !enabled.executable(
            make_pass(kParameterizedPassMaximumTargetDistanceM,
                      kParameterizedPassRequestedSpeedMps),
            kParameterizedPassMaximumTargetAngleDeg)) {
        std::cerr << "valid targeted pass envelope boundary was rejected\n";
        return 1;
    }

    if (enabled.executable(make_pass(0.24, 1.43), 0.0) ||
        enabled.executable(make_pass(
            kParameterizedPassMaximumTargetDistanceM + 0.01, 1.43), 0.0) ||
        enabled.executable(make_pass(2.0, 0.79), 0.0) ||
        enabled.executable(make_pass(2.0, 3.01), 0.0) ||
        enabled.executable(nominal, kParameterizedPassMaximumTargetAngleDeg + 0.01)) {
        std::cerr << "out-of-envelope targeted pass was accepted\n";
        return 1;
    }
    return 0;
}
