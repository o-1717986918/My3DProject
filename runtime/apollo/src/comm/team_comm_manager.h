// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/comm/team_comm_codec.h"
#include "src/world/world_snapshot.h"

#include <optional>
#include <unordered_map>

namespace comm {

/// Schedules outgoing packets and maintains recent records from teammates.
class TeamCommManager {
public:
    static constexpr std::uint8_t kProtocolVersion = 3U;
    static constexpr int kMaxRecordAgeCycles = 30;

    explicit TeamCommManager(const std::string& team_name);

    /// Returns whether this player owns the current time-division send slot.
    bool is_send_slot(int player_number, int server_cycle) const;
    TeamCommPacket make_packet(
        const world::WorldSnapshot& snapshot,
        int current_role = -1,
        const std::optional<OutgoingPassIntent>& outgoing_pass = std::nullopt) const;
    /// Ingests a decoded packet when its protocol and team version match.
    void ingest(const TeamCommPacket& packet, int current_server_cycle);
    TeamCommSnapshot make_snapshot(int current_server_cycle) const;

private:
    struct ReadyGateState {
        int passer_player_number{0};
        int receiver_player_number{0};
        std::uint8_t sequence_id{0U};
        double target_x_m{0.0};
        double target_y_m{0.0};
        double stable_since_s{0.0};
    };

    std::uint8_t version_byte_;  // (team_hash << 4) | protocol_version
    std::unordered_map<int, TeamCommRecord> records_;
    std::unordered_map<int, PassIntentRecord> pass_intents_;
    mutable std::optional<ReadyGateState> ready_gate_state_;
};

}  // namespace comm
