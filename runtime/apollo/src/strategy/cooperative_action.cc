// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/cooperative_action.h"

namespace strategy {

std::string_view to_string(ActionCategory category) {
    switch (category) {
        case ActionCategory::Hold: return "Hold";
        case ActionCategory::Dribble: return "Dribble";
        case ActionCategory::Pass: return "Pass";
        case ActionCategory::Shoot: return "Shoot";
        case ActionCategory::Clear: return "Clear";
        case ActionCategory::Move: return "Move";
        case ActionCategory::NoAction: return "NoAction";
    }
    return "NoAction";
}

std::string_view to_string(PassType pass_type) {
    switch (pass_type) {
        case PassType::None: return "None";
        case PassType::Direct: return "Direct";
        case PassType::Leading: return "Leading";
        case PassType::Through: return "Through";
    }
    return "None";
}

std::string_view to_string(RejectionReason reason) {
    switch (reason) {
        case RejectionReason::None: return "None";
        case RejectionReason::StrategyDisabled: return "StrategyDisabled";
        case RejectionReason::NotOpenPlay: return "NotOpenPlay";
        case RejectionReason::BallNotVisible: return "BallNotVisible";
        case RejectionReason::TeammateStale: return "TeammateStale";
        case RejectionReason::TeammateFallen: return "TeammateFallen";
        case RejectionReason::ReceiverInvalid: return "ReceiverInvalid";
        case RejectionReason::TooNear: return "TooNear";
        case RejectionReason::TooFar: return "TooFar";
        case RejectionReason::OutOfField: return "OutOfField";
        case RejectionReason::BallCannotReach: return "BallCannotReach";
        case RejectionReason::ReceiverLate: return "ReceiverLate";
        case RejectionReason::OpponentFirst: return "OpponentFirst";
        case RejectionReason::UnsafeBackPass: return "UnsafeBackPass";
        case RejectionReason::BelowUtilityFloor: return "BelowUtilityFloor";
    }
    return "None";
}

}  // namespace strategy
