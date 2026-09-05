// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/comm/team_comm_codec.h"

#include "src/server/server_constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace comm {

namespace {

constexpr std::uint8_t kInvalidQuantized = 0xFFU;
constexpr std::uint8_t kUnknownRoleBits = 0x07U;
constexpr std::uint8_t kPassIntentBit = 0x80U;
constexpr std::uint8_t kPassAuthorReceiverBit = 0x40U;
constexpr std::uint8_t kIntentChecksumSalt = 0xA5U;
constexpr std::uint8_t kSixBitMaximum = 0x3FU;
constexpr double kMaximumPassSpeedMps = 3.0;
constexpr double kMaximumPredictedBallTimeS = 25.2;

static_assert(
    static_cast<std::uint8_t>(PassIntentState::Expired) <= 0x0FU,
    "Pass lifecycle state must fit in four bits");

using server_constants::kFieldHalfLengthM;
using server_constants::kFieldHalfWidthM;

std::uint8_t quantize_axis(double value, double min_value, double max_value) {
    const double clamped = std::clamp(value, min_value, max_value);
    const double normalized = (clamped - min_value) / (max_value - min_value);
    return static_cast<std::uint8_t>(std::lround(normalized * 254.0));
}

double dequantize_axis(std::uint8_t value, double min_value, double max_value) {
    if (value == kInvalidQuantized) {
        return 0.0;
    }
    const double normalized = static_cast<double>(value) / 254.0;
    return min_value + normalized * (max_value - min_value);
}

std::uint8_t pack_header(const TeamCommPacket& packet) {
    const std::uint8_t sender_bits = static_cast<std::uint8_t>(
        std::clamp<int>(static_cast<int>(packet.sender_player_number), 1, 7) - 1);
    const std::uint8_t role_bits = packet.current_role >= 0 && packet.current_role <= 6
        ? static_cast<std::uint8_t>(packet.current_role)
        : kUnknownRoleBits;
    return static_cast<std::uint8_t>(
        sender_bits |
        static_cast<std::uint8_t>(role_bits << 3U) |
        static_cast<std::uint8_t>(packet.fallen ? 0x40U : 0x00U));
}

std::uint8_t quantize_range_6bit(double value, double max_value) {
    const double clamped = std::clamp(value, 0.0, max_value);
    return static_cast<std::uint8_t>(
        std::lround(clamped / max_value * kSixBitMaximum));
}

double dequantize_range_6bit(std::uint8_t value, double max_value) {
    return static_cast<double>(value & kSixBitMaximum) /
        static_cast<double>(kSixBitMaximum) * max_value;
}

std::uint8_t intent_checksum(const std::vector<std::uint8_t>& encoded) {
    std::uint8_t checksum = kIntentChecksumSalt;
    for (std::size_t i = 0; i < 7U && i < encoded.size(); ++i) {
        checksum ^= encoded[i];
    }
    return checksum;
}

}  // namespace

std::uint8_t TeamCommCodec::team_hash(const std::string& team_name) {
    // DJB2 hash truncated to 4 bits. Different team names almost always
    // produce different nibbles, filtering out opponent messages.
    std::uint32_t h = 5381U;
    for (char c : team_name) {
        h = ((h << 5) + h) + static_cast<std::uint8_t>(c);
    }
    return static_cast<std::uint8_t>(h & 0x0FU);
}

std::vector<std::uint8_t> TeamCommCodec::encode(const TeamCommPacket& packet) {
    std::vector<std::uint8_t> encoded;
    encoded.reserve(kPacketSizeBytes);
    if (packet.kind == TeamCommPacketKind::PassIntent) {
        const int sender = static_cast<int>(packet.sender_player_number);
        const int passer = static_cast<int>(packet.passer_player_number);
        const int receiver = static_cast<int>(packet.receiver_player_number);
        if (sender < 1 || sender > 7 || passer < 1 || passer > 7 ||
            receiver < 1 || receiver > 7 || passer == receiver) {
            throw std::runtime_error("Invalid pass-intent participants");
        }
        const bool authored_by_receiver =
            packet.pass_intent_author == PassIntentAuthor::Receiver;
        const int author = authored_by_receiver ? receiver : passer;
        const int expected_peer = authored_by_receiver ? passer : receiver;
        const int peer = packet.pass_peer_player_number == 0U
            ? expected_peer
            : static_cast<int>(packet.pass_peer_player_number);
        if (sender != author || peer != expected_peer) {
            throw std::runtime_error("Pass-intent author/peer mismatch");
        }
        const std::uint8_t state =
            static_cast<std::uint8_t>(packet.pass_intent_state);
        if (state > static_cast<std::uint8_t>(PassIntentState::Expired)) {
            throw std::runtime_error("Invalid pass-intent lifecycle state");
        }
        const std::uint8_t sender_bits =
            static_cast<std::uint8_t>(sender - 1);
        const std::uint8_t peer_bits = static_cast<std::uint8_t>(
            peer - 1);
        const std::uint8_t header = static_cast<std::uint8_t>(
            kPassIntentBit | sender_bits | (peer_bits << 3U) |
            (authored_by_receiver ? kPassAuthorReceiverBit : 0U));
        const std::uint16_t lifecycle_and_metrics =
            static_cast<std::uint16_t>(state) |
            (static_cast<std::uint16_t>(quantize_range_6bit(
                packet.requested_ball_speed_mps,
                kMaximumPassSpeedMps)) << 4U) |
            (static_cast<std::uint16_t>(quantize_range_6bit(
                packet.predicted_ball_time_s,
                kMaximumPredictedBallTimeS)) << 10U);
        encoded.push_back(packet.version);
        encoded.push_back(header);
        encoded.push_back(quantize_axis(
            packet.pass_target_x_m, -kFieldHalfLengthM, kFieldHalfLengthM));
        encoded.push_back(quantize_axis(
            packet.pass_target_y_m, -kFieldHalfWidthM, kFieldHalfWidthM));
        encoded.push_back(packet.pass_sequence_id);
        encoded.push_back(static_cast<std::uint8_t>(
            lifecycle_and_metrics & 0x00FFU));
        encoded.push_back(static_cast<std::uint8_t>(
            lifecycle_and_metrics >> 8U));
        encoded.push_back(intent_checksum(encoded));
        return encoded;
    }
    encoded.push_back(packet.version);
    encoded.push_back(pack_header(packet));
    encoded.push_back(quantize_axis(packet.self_x_m, -kFieldHalfLengthM, kFieldHalfLengthM));
    encoded.push_back(quantize_axis(packet.self_y_m, -kFieldHalfWidthM, kFieldHalfWidthM));
    encoded.push_back(packet.ball_seen
        ? quantize_axis(packet.ball_x_m, -kFieldHalfLengthM, kFieldHalfLengthM)
        : kInvalidQuantized);
    encoded.push_back(packet.ball_seen
        ? quantize_axis(packet.ball_y_m, -kFieldHalfWidthM, kFieldHalfWidthM)
        : kInvalidQuantized);
    encoded.push_back(packet.opponent_seen
        ? quantize_axis(packet.opponent_x_m, -kFieldHalfLengthM, kFieldHalfLengthM)
        : kInvalidQuantized);
    encoded.push_back(packet.opponent_seen
        ? quantize_axis(packet.opponent_y_m, -kFieldHalfWidthM, kFieldHalfWidthM)
        : kInvalidQuantized);
    return encoded;
}

TeamCommPacket TeamCommCodec::decode(const std::vector<std::uint8_t>& encoded) {
    if (encoded.size() != kPacketSizeBytes) {
        throw std::runtime_error("Team communication packet size mismatch");
    }
    TeamCommPacket packet;
    packet.version = encoded[0];
    const std::uint8_t header = encoded[1];
    if ((header & kPassIntentBit) != 0U) {
        if (encoded[7] != intent_checksum(encoded)) {
            throw std::runtime_error("Team pass-intent checksum mismatch");
        }
        packet.kind = TeamCommPacketKind::PassIntent;
        const std::uint8_t sender_bits = header & 0x07U;
        const std::uint8_t peer_bits = (header >> 3U) & 0x07U;
        if (sender_bits >= 7U || peer_bits >= 7U ||
            sender_bits == peer_bits) {
            throw std::runtime_error("Invalid pass-intent author/peer encoding");
        }
        packet.sender_player_number =
            static_cast<std::uint8_t>(sender_bits + 1U);
        packet.pass_peer_player_number =
            static_cast<std::uint8_t>(peer_bits + 1U);
        packet.pass_intent_author =
            (header & kPassAuthorReceiverBit) != 0U
                ? PassIntentAuthor::Receiver
                : PassIntentAuthor::Passer;
        if (packet.pass_intent_author == PassIntentAuthor::Passer) {
            packet.passer_player_number = packet.sender_player_number;
            packet.receiver_player_number = packet.pass_peer_player_number;
        } else {
            packet.passer_player_number = packet.pass_peer_player_number;
            packet.receiver_player_number = packet.sender_player_number;
        }
        const std::uint16_t lifecycle_and_metrics =
            static_cast<std::uint16_t>(encoded[5]) |
            (static_cast<std::uint16_t>(encoded[6]) << 8U);
        const std::uint8_t state = static_cast<std::uint8_t>(
            lifecycle_and_metrics & 0x000FU);
        if (state > static_cast<std::uint8_t>(PassIntentState::Expired)) {
            throw std::runtime_error("Invalid pass-intent lifecycle state");
        }
        packet.pass_intent_state = static_cast<PassIntentState>(state);
        packet.pass_target_x_m = dequantize_axis(
            encoded[2], -kFieldHalfLengthM, kFieldHalfLengthM);
        packet.pass_target_y_m = dequantize_axis(
            encoded[3], -kFieldHalfWidthM, kFieldHalfWidthM);
        packet.pass_sequence_id = encoded[4];
        packet.requested_ball_speed_mps = dequantize_range_6bit(
            static_cast<std::uint8_t>(
                (lifecycle_and_metrics >> 4U) & kSixBitMaximum),
            kMaximumPassSpeedMps);
        packet.predicted_ball_time_s = dequantize_range_6bit(
            static_cast<std::uint8_t>(
                (lifecycle_and_metrics >> 10U) & kSixBitMaximum),
            kMaximumPredictedBallTimeS);
        return packet;
    }
    packet.kind = TeamCommPacketKind::State;
    const std::uint8_t sender_bits = header & 0x07U;
    if (sender_bits >= 7U) {
        throw std::runtime_error("Invalid state-packet sender encoding");
    }
    packet.sender_player_number = static_cast<std::uint8_t>(sender_bits + 1U);
    const std::uint8_t role_bits = static_cast<std::uint8_t>((header >> 3U) & 0x07U);
    packet.current_role = role_bits == kUnknownRoleBits
        ? static_cast<std::int8_t>(-1)
        : static_cast<std::int8_t>(role_bits);
    packet.fallen = (header & 0x40U) != 0U;
    packet.self_x_m = dequantize_axis(encoded[2], -kFieldHalfLengthM, kFieldHalfLengthM);
    packet.self_y_m = dequantize_axis(encoded[3], -kFieldHalfWidthM, kFieldHalfWidthM);
    packet.ball_seen = encoded[4] != kInvalidQuantized && encoded[5] != kInvalidQuantized;
    if (packet.ball_seen) {
        packet.ball_x_m = dequantize_axis(encoded[4], -kFieldHalfLengthM, kFieldHalfLengthM);
        packet.ball_y_m = dequantize_axis(encoded[5], -kFieldHalfWidthM, kFieldHalfWidthM);
    }
    packet.opponent_seen = encoded[6] != kInvalidQuantized && encoded[7] != kInvalidQuantized;
    if (packet.opponent_seen) {
        packet.opponent_x_m = dequantize_axis(encoded[6], -kFieldHalfLengthM, kFieldHalfLengthM);
        packet.opponent_y_m = dequantize_axis(encoded[7], -kFieldHalfWidthM, kFieldHalfWidthM);
    }
    return packet;
}

TeamCommRecord TeamCommCodec::to_record(const TeamCommPacket& packet) {
    if (packet.kind != TeamCommPacketKind::State) {
        throw std::runtime_error("Pass-intent packet cannot be converted to state record");
    }
    return {
        static_cast<int>(packet.sender_player_number),
        0,
        packet.self_x_m,
        packet.self_y_m,
        packet.fallen,
        packet.ball_seen,
        packet.ball_x_m,
        packet.ball_y_m,
        packet.opponent_seen,
        packet.opponent_x_m,
        packet.opponent_y_m,
        static_cast<int>(packet.current_role),
    };
}

PassIntentRecord TeamCommCodec::to_pass_intent_record(const TeamCommPacket& packet) {
    if (packet.kind != TeamCommPacketKind::PassIntent) {
        throw std::runtime_error("State packet cannot be converted to pass-intent record");
    }
    return {
        static_cast<int>(packet.sender_player_number),
        0,
        packet.pass_intent_state,
        static_cast<int>(packet.passer_player_number),
        static_cast<int>(packet.receiver_player_number),
        packet.pass_sequence_id,
        packet.pass_target_x_m,
        packet.pass_target_y_m,
        packet.requested_ball_speed_mps,
        packet.predicted_ball_time_s,
        packet.pass_intent_author,
        static_cast<int>(packet.pass_peer_player_number),
    };
}

std::string_view to_string(PassIntentState state) {
    switch (state) {
        case PassIntentState::Proposed: return "Proposed";
        case PassIntentState::Ready: return "Ready";
        case PassIntentState::Committed: return "Committed";
        case PassIntentState::Commanded: return "Commanded";
        case PassIntentState::Executed: return "Executed";
        case PassIntentState::ReceiverZone: return "ReceiverZone";
        case PassIntentState::Received: return "Received";
        case PassIntentState::Intercepted: return "Intercepted";
        case PassIntentState::Out: return "Out";
        case PassIntentState::Timeout: return "Timeout";
        case PassIntentState::Cancelled: return "Cancelled";
        case PassIntentState::Expired: return "Expired";
    }
    return "Expired";
}

}  // namespace comm
