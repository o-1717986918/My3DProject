// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace decision {

/// Absolute pose requested from the server beam operation.
struct BeamCommand {
    double x_m{0.0};
    double y_m{0.0};
    double yaw_deg{0.0};
};

/// Planar walking target and optional final orientation.
struct WalkCommand {
    std::array<double, 2> target_2d_m{0.0, 0.0};
    bool target_absolute{true};
    std::optional<double> orientation_deg;
    bool orientation_absolute{true};
    // Multiplier for the walk runner's orientation controller. Keep the
    // default neutral; precision tasks may raise it to overcome the learned
    // locomotion policy's small-command dead zone.
    double orientation_gain{1.0};
    std::optional<int> role_id;
};

/// Requests execution of the learned get-up policy.
struct GetUpCommand {};
enum class KickMode : std::uint8_t {
    ForwardContact,
    DribbleTouch,
    TargetedPass,
    Shot,
    Clear,
};

/// Requests the validated My3D contact macro. Target metadata is consumed by
/// the decision/coordination layers; a default value preserves the original
/// forward-contact behavior and remains the same-cycle safety fallback.
struct KickCommand {
    std::optional<std::array<double, 2>> target_point_m;
    double requested_ball_speed_mps{0.0};
    std::optional<int> receiver_player_number;
    std::uint32_t action_id{0U};
    std::uint8_t sequence_id{0U};
    std::optional<std::uint64_t> restart_epoch;
    std::optional<std::uint32_t> restart_revision;
    KickMode mode{KickMode::ForwardContact};
    // A target-aware request may opt in to the original Apollo-style
    // walk-through contact only after the decision layer's bounded setup
    // timeout. MotionManager reports that path with an explicit Fallback name;
    // it must never silently reinterpret an ordinary targeted request.
    bool allow_forward_contact_fallback{false};
};
/// Requests the neutral standing keyframe.
struct NeutralCommand {};

/// Command variants emitted by the decision layer and consumed by motion control.
using HighLevelCommand = std::variant<BeamCommand, WalkCommand, GetUpCommand, KickCommand, NeutralCommand>;

}  // namespace decision
