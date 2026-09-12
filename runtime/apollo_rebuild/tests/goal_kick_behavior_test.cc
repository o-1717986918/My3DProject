// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/role_behaviors.h"
#include "src/decision/field_geometry.h"

#include <cassert>
#include <cmath>
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
    assert(walk->target_2d_m[0] > 0.5);
    assert(walk->target_2d_m[0] < 1.2);
    assert(walk->target_2d_m[1] > -0.1);
    assert(walk->target_2d_m[1] < 0.1);

    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.server_time = 10.0;
    snapshot.self.position_m = {-27.0, 0.0, 0.8};
    snapshot.ball.position_m = {-18.0, 0.7, 0.11};
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.velocity_mps = {-5.0, 0.0, 0.0};
    snapshot.ball.velocity_valid = true;

    goalkeeper.reset_state();
    const auto disabled = goalkeeper.make_command(snapshot, blackboard);
    const auto* disabled_walk = std::get_if<decision::WalkCommand>(&disabled);
    assert(disabled_walk != nullptr);
    assert(disabled_walk->target_absolute);
    assert(std::abs(disabled_walk->target_2d_m[1]) < 1.0e-9);

    const auto intercept = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* intercept_walk = std::get_if<decision::WalkCommand>(&intercept);
    assert(intercept_walk != nullptr);
    assert(!intercept_walk->target_absolute);
    assert(intercept_walk->target_2d_m[1] > 0.1);

    // Keep the selected side across a short velocity-confidence dropout.
    snapshot.server_time = 10.1;
    snapshot.ball.position_m[0] = -18.5;
    snapshot.ball.velocity_valid = false;
    const auto latched = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* latched_walk = std::get_if<decision::WalkCommand>(&latched);
    assert(latched_walk != nullptr);
    assert(!latched_walk->target_absolute);
    assert(latched_walk->target_2d_m[1] > 0.1);

    snapshot.server_time = 13.5;
    const auto expired = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* expired_walk = std::get_if<decision::WalkCommand>(&expired);
    assert(expired_walk != nullptr);
    assert(expired_walk->target_absolute);
    assert(std::abs(expired_walk->target_2d_m[1]) < 1.0e-9);
    return 0;
}
