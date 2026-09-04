// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/action_capability.h"

#include "src/decision/kick_contract.h"

#include <cmath>

namespace strategy {

namespace {
constexpr std::size_t index_of(SkillCapability capability) {
    return static_cast<std::size_t>(capability);
}
}  // namespace

ActionCapabilityRegistry::ActionCapabilityRegistry(bool enable_parameterized_kick) {
    envelopes_[index_of(SkillCapability::Walk)] = {
        SkillCapability::Walk, CapabilityState::Stable, 0.0, 100.0, 180.0, 0.0, 1.0,
        false, SkillCapability::Walk};
    envelopes_[index_of(SkillCapability::Turn)] = {
        SkillCapability::Turn, CapabilityState::Stable, 0.0, 100.0, 180.0, 0.0, 0.5,
        false, SkillCapability::Walk};
    envelopes_[index_of(SkillCapability::GetUp)] = {
        SkillCapability::GetUp, CapabilityState::Stable, 0.0, 0.0, 0.0, 0.0, 0.0,
        false, SkillCapability::GetUp};
    envelopes_[index_of(SkillCapability::ApproachRecover)] = {
        SkillCapability::ApproachRecover, CapabilityState::Stable, 0.0, 8.0, 180.0,
        0.0, 1.0, false, SkillCapability::Walk};
    envelopes_[index_of(SkillCapability::ForwardContact)] = {
        SkillCapability::ForwardContact, CapabilityState::Stable, 0.25, 0.85, 15.0,
        0.0, 1.0, false, SkillCapability::Walk};
    envelopes_[index_of(SkillCapability::TargetedPass)] = {
        SkillCapability::TargetedPass,
        enable_parameterized_kick ? CapabilityState::Experimental
                                   : CapabilityState::Unavailable,
        decision::kick_contract::kParameterizedPassMinimumTargetDistanceM,
        decision::kick_contract::kParameterizedPassMaximumTargetDistanceM,
        decision::kick_contract::kParameterizedPassMaximumTargetAngleDeg,
        decision::kick_contract::kParameterizedPassRequestedSpeedMps,
        decision::kick_contract::kParameterizedPassRequestedSpeedMps,
        true, SkillCapability::ForwardContact};
    envelopes_[index_of(SkillCapability::Shot)] = {
        SkillCapability::Shot, CapabilityState::Unavailable,
        decision::kick_contract::kMinimumTargetDistanceM,
        decision::kick_contract::kMaximumTargetDistanceM,
        decision::kick_contract::kMaximumTargetAngleDeg,
        decision::kick_contract::kMinimumRequestedSpeedMps,
        decision::kick_contract::kMaximumRequestedSpeedMps,
        true, SkillCapability::ForwardContact};
    envelopes_[index_of(SkillCapability::Clear)] = {
        SkillCapability::Clear, CapabilityState::Unavailable,
        decision::kick_contract::kMinimumTargetDistanceM,
        decision::kick_contract::kMaximumTargetDistanceM, 180.0,
        decision::kick_contract::kMinimumRequestedSpeedMps,
        decision::kick_contract::kMaximumRequestedSpeedMps,
        true, SkillCapability::ForwardContact};
}

const ActionEnvelope& ActionCapabilityRegistry::envelope(
    SkillCapability capability) const {
    return envelopes_[index_of(capability)];
}

CapabilityState ActionCapabilityRegistry::state(SkillCapability capability) const {
    return envelope(capability).state;
}

bool ActionCapabilityRegistry::executable(
    const CooperativeAction& action,
    double relative_target_angle_deg) const {
    SkillCapability capability = SkillCapability::ForwardContact;
    if (action.category == ActionCategory::Pass) {
        capability = SkillCapability::TargetedPass;
    } else if (action.category == ActionCategory::Shoot) {
        capability = SkillCapability::Shot;
    } else if (action.category == ActionCategory::Clear) {
        capability = SkillCapability::Clear;
    } else {
        return false;
    }
    const auto& limits = envelope(capability);
    if (limits.state == CapabilityState::Unavailable ||
        !std::isfinite(relative_target_angle_deg) ||
        std::abs(relative_target_angle_deg) > limits.maximum_abs_angle_deg) {
        return false;
    }
    const double distance = std::hypot(
        action.target_point_m[0] - action.start_ball_point_m[0],
        action.target_point_m[1] - action.start_ball_point_m[1]);
    return std::isfinite(distance) &&
        distance >= limits.minimum_distance_m &&
        distance <= limits.maximum_distance_m &&
        std::isfinite(action.requested_ball_speed_mps) &&
        action.requested_ball_speed_mps >= limits.minimum_requested_speed_mps &&
        action.requested_ball_speed_mps <= limits.maximum_requested_speed_mps;
}

std::string_view to_string(CapabilityState state) {
    switch (state) {
        case CapabilityState::Unavailable: return "Unavailable";
        case CapabilityState::Stable: return "Stable";
        case CapabilityState::Experimental: return "Experimental";
    }
    return "Unavailable";
}

std::string_view to_string(SkillCapability capability) {
    switch (capability) {
        case SkillCapability::Walk: return "Walk";
        case SkillCapability::Turn: return "Turn";
        case SkillCapability::GetUp: return "GetUp";
        case SkillCapability::ApproachRecover: return "ApproachRecover";
        case SkillCapability::ForwardContact: return "ForwardContact";
        case SkillCapability::TargetedPass: return "TargetedPass";
        case SkillCapability::Shot: return "Shot";
        case SkillCapability::Clear: return "Clear";
    }
    return "Walk";
}

}  // namespace strategy
