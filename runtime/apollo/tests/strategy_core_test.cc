// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/action_planner.h"
#include "src/strategy/action_capability.h"
#include "src/strategy/ball_trajectory_model.h"
#include "src/strategy/pass_candidate_generator.h"

#include <cmath>
#include <iostream>

namespace {

world::WorldSnapshot make_open_pass_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.team_name = "My3D";
    snapshot.player_number = 7;
    snapshot.server_time = 1.0;
    snapshot.match_time_s = 1.0;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {-0.6, 0.0, 0.8};
    snapshot.ball.visible = true;
    snapshot.ball.position_valid = true;
    snapshot.ball.position_age_s = 0.0;
    snapshot.ball.position_m = {0.0, 0.0, 0.11};
    snapshot.teammates.resize(7);
    for (int number = 1; number <= 7; ++number) {
        snapshot.teammates[static_cast<std::size_t>(number - 1)].player_number = number;
        snapshot.teammates[static_cast<std::size_t>(number - 1)].is_teammate = true;
    }
    auto& receiver = snapshot.teammates[5];
    receiver.seen = true;
    receiver.last_seen_time = snapshot.server_time;
    receiver.position_m = {2.0, 0.0, 0.8};
    snapshot.opponents.resize(7);
    for (int number = 1; number <= 7; ++number) {
        snapshot.opponents[static_cast<std::size_t>(number - 1)].player_number = number;
    }
    return snapshot;
}

bool has_rejection(
    const strategy::CandidateGenerationResult& result,
    strategy::RejectionReason reason) {
    for (const auto& rejection : result.rejections) {
        if (rejection.reason == reason) return true;
    }
    return false;
}

}  // namespace

int main() {
    const strategy::BallTrajectoryModel ball_model;
    const double travel_time = ball_model.travel_time_s(4.0, 1.43);
    if (!std::isfinite(travel_time) || travel_time <= 0.0 ||
        std::isfinite(ball_model.travel_time_s(20.0, 1.43))) {
        std::cerr << "ball trajectory feasibility model failed\n";
        return 1;
    }

    const world::WorldSnapshot open = make_open_pass_snapshot();
    const strategy::ActionPlanner planner;
    const auto first = planner.plan(open);
    const auto second = planner.plan(open);
    if (!first.selected.has_value() || !second.selected.has_value() ||
        first.selected->category != strategy::ActionCategory::Pass ||
        first.selected->target_player_number != 6 ||
        first.selected->action_id != second.selected->action_id ||
        first.tactical_state.possession != strategy::PossessionOwner::Ours ||
        first.tactical_state.ball_owner_player_number != 7 ||
        !first.tactical_state.ball_owner_is_teammate) {
        std::cerr << "open passing lane was not selected deterministically\n";
        return 1;
    }

    const strategy::ActionCapabilityRegistry enabled_capabilities(true);
    world::WorldSnapshot no_pass = open;
    no_pass.teammates.clear();
    const auto dribble = planner.plan(
        no_pass, enabled_capabilities, false);
    if (!dribble.selected.has_value() ||
        dribble.selected->category != strategy::ActionCategory::Dribble) {
        std::cerr << "unified planner did not retain executable dribble\n";
        return 1;
    }

    world::WorldSnapshot shot = no_pass;
    shot.ball.position_m = {24.0, 0.0, 0.11};
    const auto shot_plan = planner.plan(
        shot, enabled_capabilities, false);
    if (!shot_plan.selected.has_value() ||
        shot_plan.selected->category != strategy::ActionCategory::Shoot) {
        std::cerr << "unified planner did not prefer an in-envelope shot\n";
        return 1;
    }

    world::WorldSnapshot clear = no_pass;
    clear.ball.position_m = {-20.0, 0.0, 0.11};
    const auto clear_plan = planner.plan(
        clear, enabled_capabilities, false);
    if (!clear_plan.selected.has_value() ||
        clear_plan.selected->category != strategy::ActionCategory::Clear) {
        std::cerr << "unified planner did not prefer a defensive clearance\n";
        return 1;
    }

    const strategy::ActionCapabilityRegistry disabled_capabilities(false);
    const auto unavailable = planner.plan(
        no_pass, disabled_capabilities, false);
    if (!unavailable.selected.has_value() ||
        unavailable.selected->category != strategy::ActionCategory::Hold) {
        std::cerr << "disabled ball actions did not retain stable hold\n";
        return 1;
    }

    world::WorldSnapshot far_from_ball = no_pass;
    far_from_ball.self.position_m = {-10.0, 0.0, 0.8};
    const auto move = planner.plan(
        far_from_ball, enabled_capabilities, false);
    if (!move.selected.has_value() ||
        move.selected->category != strategy::ActionCategory::Move) {
        std::cerr << "unified planner did not select movement to a distant ball\n";
        return 1;
    }

    strategy::TacticalStateTracker phase_tracker;
    world::WorldSnapshot possession = make_open_pass_snapshot();
    possession.server_time = 10.0;
    if (phase_tracker.update(possession).phase !=
        strategy::TacticalPhase::Attack) {
        std::cerr << "phase tracker did not initialize from clear possession\n";
        return 1;
    }
    world::WorldSnapshot weak_turnover = possession;
    weak_turnover.server_time = 10.1;
    weak_turnover.teammates.clear();
    weak_turnover.self.position_m = {1.2, 0.0, 0.8};
    weak_turnover.self.orientation_wxyz = {0.0, 0.0, 0.0, 1.0};
    weak_turnover.opponents.clear();
    world::PlayerObservation opponent;
    opponent.player_number = 1;
    opponent.seen = true;
    opponent.last_seen_time = weak_turnover.server_time;
    opponent.position_m = {0.8, 0.0, 0.8};
    weak_turnover.opponents.push_back(opponent);
    const auto pending_turnover = phase_tracker.update(weak_turnover);
    if (pending_turnover.possession != strategy::PossessionOwner::Ours ||
        pending_turnover.phase != strategy::TacticalPhase::Attack) {
        std::cerr << "single weak turnover sample bypassed possession hysteresis\n";
        return 1;
    }
    weak_turnover.server_time = 10.6;
    weak_turnover.opponents.front().last_seen_time = weak_turnover.server_time;
    const auto counter_press = phase_tracker.update(weak_turnover);
    if (counter_press.possession != strategy::PossessionOwner::Theirs ||
        counter_press.phase != strategy::TacticalPhase::Transition) {
        std::cerr << "confirmed turnover did not enter counter-press transition\n";
        return 1;
    }
    weak_turnover.server_time = 12.0;
    weak_turnover.opponents.front().last_seen_time = weak_turnover.server_time;
    if (phase_tracker.update(weak_turnover).phase !=
        strategy::TacticalPhase::Defend) {
        std::cerr << "counter-press transition did not expire into defense\n";
        return 1;
    }
    weak_turnover.server_time = 12.1;
    weak_turnover.ball.visible = false;
    weak_turnover.ball.position_age_s = 0.8;
    const auto stale_phase = phase_tracker.update(weak_turnover);
    if (stale_phase.possession != strategy::PossessionOwner::Unknown ||
        stale_phase.phase != strategy::TacticalPhase::Unknown) {
        std::cerr << "stale ball did not invalidate tactical possession\n";
        return 1;
    }

    world::WorldSnapshot tracked = open;
    tracked.ball.visible = false;
    tracked.ball.position_age_s = 0.12;
    const auto tracked_result = planner.plan(tracked);
    if (!tracked_result.selected.has_value() ||
        tracked_result.selected->category != strategy::ActionCategory::Pass) {
        std::cerr << "validated ball track was rejected between camera frames\n";
        return 1;
    }

    world::WorldSnapshot invalid_ball = tracked;
    invalid_ball.ball.position_valid = false;
    const strategy::PassCandidateGenerator default_generator;
    const auto invalid_ball_result = default_generator.generate(invalid_ball);
    if (!invalid_ball_result.candidates.empty() ||
        !has_rejection(
            invalid_ball_result, strategy::RejectionReason::BallNotVisible)) {
        std::cerr << "invalid ball track was not rejected\n";
        return 1;
    }

    world::WorldSnapshot blocked = open;
    auto& blocker = blocked.opponents[0];
    blocker.seen = true;
    blocker.last_seen_time = blocked.server_time;
    blocker.position_m = {2.0, 0.0, 0.8};
    strategy::PassCandidateGenerator::Parameters direct_only_parameters;
    direct_only_parameters.enable_leading_pass = false;
    const strategy::PassCandidateGenerator direct_only(direct_only_parameters);
    const auto blocked_result = direct_only.generate(blocked);
    if (!blocked_result.candidates.empty() ||
        !has_rejection(blocked_result, strategy::RejectionReason::OpponentFirst)) {
        std::cerr << "opponent-first direct lane was not rejected\n";
        return 1;
    }

    world::WorldSnapshot stopped = open;
    stopped.play_mode = world::PlayMode::BeforeKickOff;
    const auto stopped_result = direct_only.generate(stopped);
    if (!stopped_result.candidates.empty() ||
        !has_rejection(stopped_result, strategy::RejectionReason::NotOpenPlay)) {
        std::cerr << "non-PlayOn pass generation was not gated\n";
        return 1;
    }

    world::WorldSnapshot stale = open;
    stale.server_time = 3.0;
    stale.teammates[5].seen = false;
    stale.teammates[5].last_seen_time = 1.0;
    const auto stale_result = direct_only.generate(stale);
    if (!stale_result.candidates.empty() ||
        !has_rejection(stale_result, strategy::RejectionReason::TeammateStale)) {
        std::cerr << "stale receiver position was not rejected\n";
        return 1;
    }

    world::WorldSnapshot reach_race = open;
    reach_race.self.position_m = {-1.0, 0.0, 0.8};
    reach_race.self.orientation_wxyz = {0.0, 0.0, 0.0, 1.0};
    reach_race.teammates[6].seen = true;
    reach_race.teammates[6].position_m = reach_race.self.position_m;
    reach_race.opponents[0].seen = true;
    reach_race.opponents[0].last_seen_time = reach_race.server_time;
    reach_race.opponents[0].position_m = {1.2, 0.0, 0.8};
    const auto race_state = strategy::build_tactical_state(reach_race);
    if (race_state.nearest_teammate_ball_distance_m >=
            race_state.nearest_opponent_ball_distance_m ||
        race_state.nearest_teammate_ball_time_s <=
            race_state.nearest_opponent_ball_time_s ||
        race_state.possession != strategy::PossessionOwner::Theirs) {
        std::cerr << "possession ignored the reach-time race\n";
        return 1;
    }

    world::WorldSnapshot risk_snapshot = open;
    risk_snapshot.own_score = 1;
    risk_snapshot.opponent_score = 0;
    risk_snapshot.match_time_s = 239.0;
    if (strategy::build_tactical_state(risk_snapshot).risk_mode !=
        strategy::TacticalRiskMode::Balanced) {
        std::cerr << "late-match threshold triggered too early\n";
        return 1;
    }
    risk_snapshot.match_time_s = 240.0;
    if (strategy::build_tactical_state(risk_snapshot).risk_mode !=
        strategy::TacticalRiskMode::ProtectLead) {
        std::cerr << "protect-lead mode was not selected\n";
        return 1;
    }
    risk_snapshot.own_score = 0;
    risk_snapshot.opponent_score = 1;
    if (strategy::build_tactical_state(risk_snapshot).risk_mode !=
        strategy::TacticalRiskMode::ChaseGoal) {
        std::cerr << "chase-goal mode was not selected\n";
        return 1;
    }
    strategy::TacticalRiskParameters custom_threshold;
    custom_threshold.late_match_threshold_s = 120.0;
    risk_snapshot.match_time_s = 119.9;
    if (strategy::build_tactical_state(risk_snapshot, custom_threshold).risk_mode !=
        strategy::TacticalRiskMode::Balanced) {
        std::cerr << "custom late-match threshold was ignored\n";
        return 1;
    }
    risk_snapshot.match_time_s = 120.0;
    if (strategy::build_tactical_state(risk_snapshot, custom_threshold).risk_mode !=
        strategy::TacticalRiskMode::ChaseGoal) {
        std::cerr << "custom chase-goal threshold was ignored\n";
        return 1;
    }
    return 0;
}
