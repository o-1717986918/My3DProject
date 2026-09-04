// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/strategy/cooperative_action.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace strategy {

enum class CapabilityState : std::uint8_t {
    Unavailable,
    Stable,
    Experimental,
};

enum class SkillCapability : std::uint8_t {
    Walk,
    Turn,
    GetUp,
    ApproachRecover,
    ForwardContact,
    DribbleTouch,
    TargetedPass,
    Shot,
    Clear,
};

struct ActionEnvelope {
    SkillCapability capability{SkillCapability::Walk};
    CapabilityState state{CapabilityState::Unavailable};
    double minimum_distance_m{0.0};
    double maximum_distance_m{0.0};
    double maximum_abs_angle_deg{180.0};
    double minimum_requested_speed_mps{0.0};
    double maximum_requested_speed_mps{0.0};
    bool requires_target{false};
    SkillCapability fallback{SkillCapability::Walk};
};

/// Describes the action surface that the strategy layer is allowed to use.
/// This is an execution contract, not a declaration of future training goals.
class ActionCapabilityRegistry {
public:
    explicit ActionCapabilityRegistry(bool enable_parameterized_kick = false);

    const ActionEnvelope& envelope(SkillCapability capability) const;
    CapabilityState state(SkillCapability capability) const;
    bool executable(
        const CooperativeAction& action,
        double relative_target_angle_deg) const;

private:
    std::array<ActionEnvelope, 9> envelopes_{};
};

std::string_view to_string(CapabilityState state);
std::string_view to_string(SkillCapability capability);

}  // namespace strategy
