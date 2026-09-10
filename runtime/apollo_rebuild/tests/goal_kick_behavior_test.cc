// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/role_behaviors.h"
#include "src/decision/field_geometry.h"

#include <cassert>
#include <variant>

int main() {
    for (const int player_number : {6, 7}) {
        const auto pose =
            decision::field_geometry::player_defensive_kickoff_beam_pose(
                player_number);
        const decision::field_geometry::Position2 position{pose[0], pose[1]};
        assert(!decision::field_geometry::is_in_our_goalie_area(position));
        assert(position[0] < 0.0);
        assert(decision::field_geometry::squared_norm(position) >
               decision::field_geometry::kCenterCircleRadiusM *
                   decision::field_geometry::kCenterCircleRadiusM);
    }

    world::WorldSnapshot snapshot;
    snapshot.play_mode = world::PlayMode::OurGoalKick;
    snapshot.play_mode_group = world::PlayModeGroup::OurKick;
    snapshot.self.position_m = {-26.1, 0.0, 0.8};
    snapshot.ball.position_m = {-25.5, 0.0, 0.11};

    decision::Blackboard blackboard;
    decision::GKBehavior goalkeeper;
    const auto command = goalkeeper.make_command(snapshot, blackboard);
    const auto* walk = std::get_if<decision::WalkCommand>(&command);

    assert(walk != nullptr);
    assert(!walk->target_absolute);
    assert(walk->target_2d_m[0] > 1.4);
    assert(walk->target_2d_m[1] > -0.1);
    assert(walk->target_2d_m[1] < 0.1);
    return 0;
}
