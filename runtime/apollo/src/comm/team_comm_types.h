// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include <cstdint>
#include <vector>

namespace comm {

enum class TeamCommPacketKind : std::uint8_t {
    State,
    PassIntent,
};

enum class PassIntentState : std::uint8_t {
    Proposed,
    Ready,
    Committed,
    Commanded,
    Executed,
    ReceiverZone,
    Received,
    Intercepted,
    Out,
    Timeout,
    Cancelled,
    Expired,
};

/// Which participant authored a pass-lifecycle update. The other participant
/// is encoded explicitly as the peer, so authorship never has to be inferred
/// from the lifecycle state.
enum class PassIntentAuthor : std::uint8_t {
    Passer,
    Receiver,
};

/// Quantizable state sent by one player over the server speech channel.
struct TeamCommPacket {
    std::uint8_t version{1};
    TeamCommPacketKind kind{TeamCommPacketKind::State};
    std::uint8_t sender_player_number{0};
    double self_x_m{0.0};
    double self_y_m{0.0};
    bool fallen{false};
    bool ball_seen{false};
    double ball_x_m{0.0};
    double ball_y_m{0.0};
    bool opponent_seen{false};
    double opponent_x_m{0.0};
    double opponent_y_m{0.0};
    std::int8_t current_role{-1};

    PassIntentState pass_intent_state{PassIntentState::Proposed};
    PassIntentAuthor pass_intent_author{PassIntentAuthor::Passer};
    std::uint8_t pass_peer_player_number{0U};
    std::uint8_t pass_sequence_id{0U};
    std::uint8_t passer_player_number{0U};
    std::uint8_t receiver_player_number{0U};
    double pass_target_x_m{0.0};
    double pass_target_y_m{0.0};
    double requested_ball_speed_mps{0.0};
    double predicted_ball_time_s{0.0};
};

/// A decoded teammate packet annotated with its receive cycle.
struct TeamCommRecord {
    int sender_player_number{0};
    int server_cycle{0};
    double self_x_m{0.0};
    double self_y_m{0.0};
    bool fallen{false};
    bool ball_seen{false};
    double ball_x_m{0.0};
    double ball_y_m{0.0};
    bool opponent_seen{false};
    double opponent_x_m{0.0};
    double opponent_y_m{0.0};
    int current_role{-1};
};

struct PassIntentRecord {
    int sender_player_number{0};
    int server_cycle{0};
    PassIntentState state{PassIntentState::Proposed};
    int passer_player_number{0};
    int receiver_player_number{0};
    std::uint8_t sequence_id{0U};
    double target_x_m{0.0};
    double target_y_m{0.0};
    double requested_ball_speed_mps{0.0};
    double predicted_ball_time_s{0.0};
    PassIntentAuthor author{PassIntentAuthor::Passer};
    int peer_player_number{0};
};

struct OutgoingPassIntent {
    int receiver_player_number{0};
    std::uint8_t sequence_id{0U};
    double target_x_m{0.0};
    double target_y_m{0.0};
    double requested_ball_speed_mps{0.0};
    double predicted_ball_time_s{0.0};
};

/// Recent team records exposed to the world model as one cycle-level view.
struct TeamCommSnapshot {
    std::vector<TeamCommRecord> records;
    std::vector<PassIntentRecord> pass_intents;
};

}  // namespace comm
