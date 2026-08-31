// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/comm/team_comm_manager.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

world::WorldSnapshot make_snapshot(int player_number) {
    world::WorldSnapshot snapshot;
    snapshot.team_name = "My3D";
    snapshot.player_number = player_number;
    snapshot.server_time = 1.0;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {0.0, 0.0, 0.8};
    snapshot.ball.visible = true;
    snapshot.ball.position_m = {0.5, -0.25, 0.11};
    return snapshot;
}

bool nearly_equal(double lhs, double rhs, double tolerance) {
    return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    comm::TeamCommManager passer("My3D");
    comm::TeamCommManager receiver("My3D");
    world::WorldSnapshot passer_snapshot = make_snapshot(7);
    world::WorldSnapshot receiver_snapshot = make_snapshot(6);

    const comm::TeamCommPacket state = passer.make_packet(passer_snapshot, 3);
    const auto decoded_state = comm::TeamCommCodec::decode(
        comm::TeamCommCodec::encode(state));
    if (decoded_state.kind != comm::TeamCommPacketKind::State ||
        decoded_state.sender_player_number != 7U || !decoded_state.ball_seen ||
        decoded_state.current_role != 3) {
        std::cerr << "legacy state packet round trip failed\n";
        return 1;
    }

    comm::OutgoingPassIntent outgoing;
    outgoing.receiver_player_number = 6;
    outgoing.sequence_id = 42U;
    outgoing.target_x_m = 4.0;
    outgoing.target_y_m = 1.0;
    outgoing.requested_ball_speed_mps = 1.43;
    outgoing.predicted_ball_time_s = 3.1;
    const auto proposed = comm::TeamCommCodec::decode(
        comm::TeamCommCodec::encode(
            passer.make_packet(passer_snapshot, 3, outgoing)));
    if (proposed.kind != comm::TeamCommPacketKind::PassIntent ||
        proposed.pass_intent_state != comm::PassIntentState::Proposed ||
        proposed.passer_player_number != 7U ||
        proposed.receiver_player_number != 6U ||
        proposed.pass_sequence_id != 42U) {
        std::cerr << "pass proposal packet round trip failed\n";
        return 1;
    }

    receiver.ingest(proposed, 10);
    receiver_snapshot.team_comm_snapshot = receiver.make_snapshot(10);
    const auto ready = comm::TeamCommCodec::decode(
        comm::TeamCommCodec::encode(
            receiver.make_packet(receiver_snapshot, 4)));
    if (ready.pass_intent_state != comm::PassIntentState::Ready ||
        ready.passer_player_number != 7U || ready.receiver_player_number != 6U ||
        ready.pass_sequence_id != 42U ||
        !nearly_equal(ready.pass_target_x_m, 4.0, 0.2) ||
        !nearly_equal(ready.pass_target_y_m, 1.0, 0.15)) {
        std::cerr << "pass ready acknowledgement failed\n";
        return 1;
    }

    passer.ingest(ready, 11);
    const auto handshake = passer.make_snapshot(11);
    if (handshake.pass_intents.size() != 1U ||
        handshake.pass_intents.front().state != comm::PassIntentState::Ready) {
        std::cerr << "ready acknowledgement was not retained\n";
        return 1;
    }

    auto corrupted = comm::TeamCommCodec::encode(proposed);
    corrupted[3] ^= 0x01U;
    bool rejected_corruption = false;
    try {
        static_cast<void>(comm::TeamCommCodec::decode(corrupted));
    } catch (const std::runtime_error&) {
        rejected_corruption = true;
    }
    if (!rejected_corruption) {
        std::cerr << "corrupted pass intent was accepted\n";
        return 1;
    }
    return 0;
}
