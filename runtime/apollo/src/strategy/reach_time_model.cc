// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/reach_time_model.h"

#include "src/math/math_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strategy {

ReachTimeModel::ReachTimeModel() = default;

ReachTimeModel::ReachTimeModel(Parameters parameters)
    : parameters_(parameters) {}

double ReachTimeModel::estimate_s(
    const Position2& player_position_m,
    const Position2& target_position_m,
    std::optional<double> player_yaw_deg,
    bool fallen) const {
    if (fallen || parameters_.travel_speed_mps <= 0.0 ||
        parameters_.turn_rate_deg_s <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double distance = math::planar_dist(player_position_m, target_position_m);
    const double travel_distance = std::max(0.0, distance - parameters_.control_radius_m);
    double turn_time = 0.0;
    if (player_yaw_deg.has_value() && distance > 1.0e-6) {
        const double target_heading = math::vector_angle_deg(
            math::vec2_sub(target_position_m, player_position_m));
        turn_time = std::abs(math::normalize_deg(target_heading - *player_yaw_deg)) /
                    parameters_.turn_rate_deg_s;
    }
    return parameters_.startup_time_s + turn_time +
           travel_distance / parameters_.travel_speed_mps +
           parameters_.observation_margin_s;
}

}  // namespace strategy
