// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/formation.h"

#include <cmath>
#include <iostream>

namespace {

decision::FormationContext make_context(
    strategy::TacticalPhase phase,
    strategy::TacticalRiskMode risk_mode) {
    decision::FormationContext context;
    context.ball_position_m = {0.0, 0.0};
    context.play_mode = world::PlayMode::PlayOn;
    context.field_length_m = 55.0;
    context.field_width_m = 36.0;
    context.phase = phase;
    context.risk_mode = risk_mode;
    return context;
}

}  // namespace

int main() {
    const decision::Formation formation;
    const auto attack = formation.compute(make_context(
        strategy::TacticalPhase::Attack,
        strategy::TacticalRiskMode::Balanced));
    const auto defend = formation.compute(make_context(
        strategy::TacticalPhase::Defend,
        strategy::TacticalRiskMode::Balanced));
    const auto protect = formation.compute(make_context(
        strategy::TacticalPhase::Attack,
        strategy::TacticalRiskMode::ProtectLead));
    const auto chase = formation.compute(make_context(
        strategy::TacticalPhase::Attack,
        strategy::TacticalRiskMode::ChaseGoal));

    // Role index 5 is ST. Defending and protecting a lead must not leave the
    // striker farther forward than balanced attack; chasing a goal must.
    if (!(defend[5][0] < attack[5][0] &&
          protect[5][0] < attack[5][0] &&
          chase[5][0] > attack[5][0])) {
        std::cerr << "phase/risk formation did not change striker depth\n";
        return 1;
    }
    // Defending shape widens the centre-back pair and keeps all targets finite.
    if (!(std::abs(defend[1][1]) > std::abs(attack[1][1]) &&
          std::isfinite(defend[3][0]) && std::isfinite(defend[3][1]))) {
        std::cerr << "defensive formation was not widened safely\n";
        return 1;
    }
    return 0;
}
