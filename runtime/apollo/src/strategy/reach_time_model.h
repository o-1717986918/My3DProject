// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "src/strategy/cooperative_action.h"

#include <optional>

namespace strategy {

class ReachTimeModel {
public:
    struct Parameters {
        double travel_speed_mps{1.10};
        double turn_rate_deg_s{120.0};
        double startup_time_s{0.20};
        double control_radius_m{0.55};
        double observation_margin_s{0.20};
    };

    ReachTimeModel();
    explicit ReachTimeModel(Parameters parameters);

    double estimate_s(
        const Position2& player_position_m,
        const Position2& target_position_m,
        std::optional<double> player_yaw_deg = std::nullopt,
        bool fallen = false) const;

    const Parameters& parameters() const { return parameters_; }

private:
    Parameters parameters_;
};

}  // namespace strategy
