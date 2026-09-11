// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/role_behaviors.h"
#include "src/decision/field_geometry.h"

#include <cassert>
#include <cmath>
#include <variant>

int main() {
    world::WorldSnapshot snapshot;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {19.515, 4.259, 0.65};
    snapshot.ball.position_m = {20.0, 4.0, 0.11};

    decision::Blackboard blackboard;
    decision::RoleManager role_manager;
    decision::APBehavior attacker;
    const auto command = attacker.make_command(snapshot, blackboard, role_manager);
    const auto* walk = std::get_if<decision::WalkCommand>(&command);

    assert(walk != nullptr);
    assert(walk->orientation_deg.has_value());
    const double expected_heading_deg =
        std::atan2(-4.0, decision::field_geometry::kActualHalfLengthM - 20.0) *
        180.0 / 3.14159265358979323846;
    assert(std::abs(*walk->orientation_deg - expected_heading_deg) < 1.0e-6);
    return 0;
}
