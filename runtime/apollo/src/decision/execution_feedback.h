// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <optional>

namespace decision {

enum class ExecutionStatus : std::uint8_t {
    Running,
    Completed,
    Rejected,
    TimedOut,
};

enum class MotionRequestKind : std::uint8_t {
    Unknown,
    Walk,
    GetUp,
    Kick,
    Neutral,
};

/// Result of one concrete motion-layer request, delivered on the next
/// decision cycle. Cooperative identifiers are present only when the request
/// carried a coordinated action such as a targeted pass.
struct ExecutionFeedback {
    std::uint64_t request_id{0U};
    double server_time{0.0};
    ExecutionStatus status{ExecutionStatus::Running};
    MotionRequestKind request_kind{MotionRequestKind::Unknown};
    std::optional<std::uint32_t> cooperative_action_id;
    std::optional<std::uint8_t> sequence_id;
    std::optional<std::uint64_t> restart_epoch;
    std::optional<std::uint32_t> restart_revision;
};

constexpr bool is_terminal(ExecutionStatus status) {
    return status != ExecutionStatus::Running;
}

constexpr bool is_failure(ExecutionStatus status) {
    return status == ExecutionStatus::Rejected ||
           status == ExecutionStatus::TimedOut;
}

constexpr const char* to_string(ExecutionStatus status) {
    switch (status) {
        case ExecutionStatus::Running: return "Running";
        case ExecutionStatus::Completed: return "Completed";
        case ExecutionStatus::Rejected: return "Rejected";
        case ExecutionStatus::TimedOut: return "TimedOut";
    }
    return "Rejected";
}

constexpr const char* to_string(MotionRequestKind kind) {
    switch (kind) {
        case MotionRequestKind::Unknown: return "Unknown";
        case MotionRequestKind::Walk: return "Walk";
        case MotionRequestKind::GetUp: return "GetUp";
        case MotionRequestKind::Kick: return "Kick";
        case MotionRequestKind::Neutral: return "Neutral";
    }
    return "Unknown";
}

}  // namespace decision
