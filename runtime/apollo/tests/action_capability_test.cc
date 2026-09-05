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

strategy::CooperativeAction make_dribble(double distance_m, double speed_mps) {
    auto action = make_pass(distance_m, speed_mps);
    action.category = strategy::ActionCategory::Dribble;
    return action;
}

strategy::CooperativeAction make_shot(double distance_m, double speed_mps) {
    auto action = make_pass(distance_m, speed_mps);
    action.category = strategy::ActionCategory::Shoot;
    return action;
}

strategy::CooperativeAction make_clear(double distance_m, double speed_mps) {
    auto action = make_pass(distance_m, speed_mps);
    action.category = strategy::ActionCategory::Clear;
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
    if (disabled.executable(make_dribble(0.55, 0.90), 0.0) ||
        !enabled.executable(make_dribble(0.55, 0.90), 0.0) ||
        enabled.state(strategy::SkillCapability::DribbleTouch) !=
            strategy::CapabilityState::Experimental ||
        !enabled.executable(make_dribble(0.55, 0.90), 5.99) ||
        enabled.executable(make_dribble(0.55, 0.90), 6.01) ||
        enabled.executable(make_dribble(0.70, 0.90), 0.0) ||
        enabled.executable(make_dribble(0.55, 0.95), 0.0)) {
        std::cerr << "procedural dribble capability envelope is incorrect\n";
        return 1;
    }
    if (disabled.executable(make_shot(4.0, 2.50), 0.0) ||
        disabled.supported(make_shot(4.0, 2.50)) ||
        !enabled.executable(make_shot(4.0, 2.50), 0.0) ||
        !enabled.supported(make_shot(4.0, 2.50)) ||
        enabled.state(strategy::SkillCapability::Shot) !=
            strategy::CapabilityState::Experimental ||
        enabled.executable(make_shot(4.51, 2.50), 0.0) ||
        enabled.executable(make_shot(4.0, 2.49), 0.0) ||
        enabled.executable(make_shot(4.0, 2.50), 1.01)) {
        std::cerr << "procedural shot capability envelope is incorrect\n";
        return 1;
    }
    if (!enabled.supported(make_shot(4.0, 2.50)) ||
        enabled.executable(make_shot(4.0, 2.50), 30.0)) {
        std::cerr << "supported shot could not be distinguished from release-ready\n";
        return 1;
    }
    if (disabled.executable(make_clear(6.0, 3.50), 0.0) ||
        !enabled.executable(make_clear(6.0, 3.50), 0.0) ||
        enabled.state(strategy::SkillCapability::Clear) !=
            strategy::CapabilityState::Experimental ||
        enabled.executable(make_clear(6.51, 3.50), 0.0) ||
        enabled.executable(make_clear(6.0, 3.49), 0.0) ||
        enabled.executable(make_clear(6.0, 3.50), 1.01)) {
        std::cerr << "procedural clear capability envelope is incorrect\n";
        return 1;
    }

    if (!enabled.executable(
            make_pass(kParameterizedPassMinimumTargetDistanceM,
                      parameterized_pass_requested_speed_mps(
                          kParameterizedPassMinimumTargetDistanceM)),
            -kParameterizedPassMaximumTargetAngleDeg) ||
        !enabled.executable(
            make_pass(kParameterizedPassMaximumTargetDistanceM,
                      parameterized_pass_requested_speed_mps(
                          kParameterizedPassMaximumTargetDistanceM)),
            kParameterizedPassMaximumTargetAngleDeg) ||
        !enabled.executable(make_pass(3.5, 2.20), 0.0) ||
        !enabled.executable(make_pass(5.0, 3.00), 0.0)) {
        std::cerr << "valid targeted pass envelope boundary was rejected\n";
        return 1;
    }

    if (enabled.executable(make_pass(0.24, 1.43), 0.0) ||
        enabled.executable(make_pass(
            kParameterizedPassMaximumTargetDistanceM + 0.01, 1.43), 0.0) ||
        enabled.executable(make_pass(2.0, 0.79), 0.0) ||
        enabled.executable(make_pass(2.0, 3.01), 0.0) ||
        enabled.executable(make_pass(3.5, 1.43), 0.0) ||
        enabled.executable(make_pass(5.0, 2.20), 0.0) ||
        enabled.executable(nominal, kParameterizedPassMaximumTargetAngleDeg + 0.01)) {
        std::cerr << "out-of-envelope targeted pass was accepted\n";
        return 1;
    }
    return 0;
}
