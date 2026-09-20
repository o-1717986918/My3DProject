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
    // Once the shot latch expires, fresh ball position still causes early
    // lateral positioning instead of reverting blindly to the center.
    assert(expired_walk->target_2d_m[1] > 0.0);

    snapshot.ball.position_age_s = 1.0;
    const auto stale_ball = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* stale_walk = std::get_if<decision::WalkCommand>(&stale_ball);
    assert(stale_walk != nullptr);
    assert(stale_walk->target_absolute);
    assert(std::abs(stale_walk->target_2d_m[1]) < 1.0e-9);

    // The opponent->ball ray triggers before a ball-velocity estimate exists.
    // Of two credible lines through the goal, select the player closer to the
    // ball, not a more distant player whose line happens to be centered.
    goalkeeper.reset_state();
    snapshot.server_time = 20.0;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {-12.0, 0.2, 0.11};
    snapshot.ball.velocity_valid = false;
    snapshot.self.position_m = {-27.0, -1.3, 0.8};
    snapshot.opponents.resize(2);
    snapshot.opponents[0].last_seen_time = 20.0;
    snapshot.opponents[0].position_m = {-11.0, 0.3, 0.5};
    snapshot.opponents[1].last_seen_time = 20.0;
    snapshot.opponents[1].position_m = {-10.2, 0.2, 0.5};
    const auto pre_shot = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* pre_shot_walk = std::get_if<decision::WalkCommand>(&pre_shot);
    assert(pre_shot_walk != nullptr);
    assert(pre_shot_walk->target_absolute);
    assert(std::abs(pre_shot_walk->target_2d_m[1] + 1.35) < 1.0e-6);

    snapshot.server_time = 21.0;
    snapshot.opponents[0].last_seen_time = 20.0;  // stale
    snapshot.opponents[1].last_seen_time = 21.0;
    snapshot.self.position_m[1] = 0.2;
    const auto fresher_shooter = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* fresher_walk = std::get_if<decision::WalkCommand>(&fresher_shooter);
    assert(fresher_walk != nullptr);
    assert(fresher_walk->target_absolute);
    assert(std::abs(fresher_walk->target_2d_m[1] - 0.2) < 1.0e-6);

    snapshot.opponents.clear();
    snapshot.self.position_m[1] = 0.0;
    const auto ball_only = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* ball_only_walk = std::get_if<decision::WalkCommand>(&ball_only);
    assert(ball_only_walk != nullptr);
    assert(ball_only_walk->target_absolute);
    assert(ball_only_walk->target_2d_m[1] > 0.0);

    snapshot.ball.position_m[0] = 5.0;
    const auto far_ball = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* far_walk = std::get_if<decision::WalkCommand>(&far_ball);
    assert(far_walk != nullptr);
    assert(far_walk->target_absolute);
    assert(std::abs(far_walk->target_2d_m[1]) < 1.0e-9);

    // A fresh, stationary loose ball in our small area can be claimed before
    // an opponent arrives; the field AP waits outside the area for the push.
    goalkeeper.reset_state();
    snapshot.server_time = 30.0;
    snapshot.player_number = 1;
    snapshot.self.position_m = {-27.0, 0.0, 0.8};
    snapshot.ball.position_m = {-25.2, 1.0, 0.11};
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.velocity_valid = false;
    snapshot.ball.velocity_mps = {0.0, 0.0, 0.0};
    const auto unconfirmed = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* unconfirmed_walk = std::get_if<decision::WalkCommand>(&unconfirmed);
    assert(unconfirmed_walk != nullptr);
    assert(unconfirmed_walk->target_absolute ||
           std::hypot(unconfirmed_walk->target_2d_m[0],
                      unconfirmed_walk->target_2d_m[1]) > 0.51);

    snapshot.server_time += 0.16;
    (void)goalkeeper.make_command(snapshot, blackboard, true);
    snapshot.server_time += 0.16;
    const auto claim = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* claim_walk = std::get_if<decision::WalkCommand>(&claim);
    assert(claim_walk != nullptr);
    assert(!claim_walk->target_absolute);
    assert(claim_walk->target_2d_m[0] > 0.0);

    // Continue the outward approach when the ball starts moving away; do not
    // retreat immediately at the first contact frame.
    snapshot.ball.velocity_valid = true;
    snapshot.ball.velocity_mps = {0.8, 0.0, 0.0};
    const auto follow_through = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* follow_through_walk = std::get_if<decision::WalkCommand>(&follow_through);
    assert(follow_through_walk != nullptr);
    assert(!follow_through_walk->target_absolute);

    snapshot.ball.position_m[0] = -23.55;
    const auto edge_follow_through = goalkeeper.make_command(
        snapshot, blackboard, true);
    const auto* edge_walk = std::get_if<decision::WalkCommand>(&edge_follow_through);
    assert(edge_walk != nullptr);
    assert(!edge_walk->target_absolute);
    assert(edge_walk->target_2d_m[0] > 0.0);
    snapshot.ball.position_m[0] = -25.2;
    snapshot.self.position_m = {-24.65, 1.0, 0.8};
    snapshot.ball.position_age_s = 0.8;
    const auto occluded_contact = goalkeeper.make_command(
        snapshot, blackboard, true);
    const auto* occluded_walk = std::get_if<decision::WalkCommand>(&occluded_contact);
    assert(occluded_walk != nullptr);
    assert(!occluded_walk->target_absolute);
    assert(occluded_walk->target_2d_m[0] > 0.0);

    decision::configure_candidate_action_features(false, false, true);
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.velocity_mps = {0.0, 0.0, 0.0};
    snapshot.player_number = 7;
    snapshot.self.position_m = {-24.0, 1.0, 0.8};
    decision::RoleManager role_manager;
    decision::APBehavior attacker;
    const auto receive = attacker.make_command(snapshot, blackboard, role_manager);
    const auto* receive_walk = std::get_if<decision::WalkCommand>(&receive);
    assert(receive_walk != nullptr);
    assert(receive_walk->target_2d_m[0] > 0.0);

    // A close opponent or an approaching ball cancels the claim. The keeper
    // returns to positional defense and the AP no longer yields possession.
    snapshot.player_number = 1;
    snapshot.self.position_m = {-27.0, 0.0, 0.8};
    snapshot.opponents.resize(1);
    snapshot.opponents[0].last_seen_time = snapshot.server_time;
    snapshot.opponents[0].position_m = {-24.8, 1.0, 0.5};
    const auto pressured = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* pressured_walk = std::get_if<decision::WalkCommand>(&pressured);
    assert(pressured_walk != nullptr);
    assert(pressured_walk->target_absolute ||
           std::hypot(pressured_walk->target_2d_m[0],
                      pressured_walk->target_2d_m[1]) > 0.51);

    snapshot.opponents.clear();
    snapshot.ball.velocity_mps = {-0.4, 0.0, 0.0};
    const auto moving = goalkeeper.make_command(snapshot, blackboard, true);
    const auto* moving_walk = std::get_if<decision::WalkCommand>(&moving);
    assert(moving_walk != nullptr);
    assert(moving_walk->target_absolute ||
           std::hypot(moving_walk->target_2d_m[0],
                      moving_walk->target_2d_m[1]) > 0.51);
    return 0;
}
