// SPDX-License-Identifier: GPL-3.0-or-later
// Strategy design informed by Cyrus2D/HELIOS CooperativeAction; this is an
// independent 3D implementation for My3DProject.

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace strategy {

using Position2 = std::array<double, 2>;

enum class ActionCategory : std::uint8_t {
    Hold,
    Dribble,
    Pass,
    Shoot,
    Clear,
    Move,
    NoAction,
};

enum class PassType : std::uint8_t {
    None,
    Direct,
    Leading,
    Through,
};

enum class RejectionReason : std::uint8_t {
    None,
    StrategyDisabled,
    NotOpenPlay,
    BallNotVisible,
    TeammateStale,
    TeammateFallen,
    ReceiverInvalid,
    TooNear,
    TooFar,
    OutOfField,
    BallCannotReach,
    ReceiverLate,
    OpponentFirst,
    UnsafeBackPass,
    CapabilityUnavailable,
    BelowUtilityFloor,
};

/// One 3D-aware cooperative action candidate. Times are seconds, not 2D
/// server cycles, and are derived from the measured Apollo execution contract.
struct CooperativeAction {
    std::uint32_t action_id{0U};
    std::uint8_t sequence_id{0U};
    ActionCategory category{ActionCategory::NoAction};
    PassType pass_type{PassType::None};
    int actor_player_number{0};
    int target_player_number{0};
    Position2 start_ball_point_m{0.0, 0.0};
    Position2 target_point_m{0.0, 0.0};
    double requested_ball_speed_mps{0.0};
    double predicted_ball_time_s{0.0};
    double predicted_receiver_time_s{0.0};
    double predicted_opponent_time_s{0.0};
    double interception_margin_s{0.0};
    double utility{0.0};
    double confidence{0.0};
};

struct RejectedCandidate {
    PassType pass_type{PassType::None};
    int target_player_number{0};
    Position2 target_point_m{0.0, 0.0};
    RejectionReason reason{RejectionReason::None};
};

std::string_view to_string(ActionCategory category);
std::string_view to_string(PassType pass_type);
std::string_view to_string(RejectionReason reason);

}  // namespace strategy
