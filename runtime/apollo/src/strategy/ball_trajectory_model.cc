// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/strategy/ball_trajectory_model.h"

#include <algorithm>
#include <cmath>

namespace strategy {

BallTrajectoryModel::BallTrajectoryModel() = default;

BallTrajectoryModel::BallTrajectoryModel(Parameters parameters)
    : parameters_(parameters) {}

double BallTrajectoryModel::travel_time_s(
    double distance_m,
    double initial_speed_mps) const {
    if (distance_m < 0.0 || initial_speed_mps <= 0.0 ||
        parameters_.rolling_deceleration_mps2 <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (distance_m == 0.0) return 0.0;
    const double v2 = initial_speed_mps * initial_speed_mps -
                      2.0 * parameters_.rolling_deceleration_mps2 * distance_m;
    if (v2 < parameters_.minimum_control_speed_mps *
                 parameters_.minimum_control_speed_mps) {
        return std::numeric_limits<double>::infinity();
    }
    return (initial_speed_mps - std::sqrt(v2)) /
           parameters_.rolling_deceleration_mps2;
}

double BallTrajectoryModel::arrival_speed_mps(
    double distance_m,
    double initial_speed_mps) const {
    const double v2 = initial_speed_mps * initial_speed_mps -
                      2.0 * parameters_.rolling_deceleration_mps2 *
                          std::max(0.0, distance_m);
    return v2 > 0.0 ? std::sqrt(v2) : 0.0;
}

double BallTrajectoryModel::max_controlled_distance_m(double initial_speed_mps) const {
    const double vmin = parameters_.minimum_control_speed_mps;
    if (initial_speed_mps <= vmin || parameters_.rolling_deceleration_mps2 <= 0.0) {
        return 0.0;
    }
    return (initial_speed_mps * initial_speed_mps - vmin * vmin) /
           (2.0 * parameters_.rolling_deceleration_mps2);
}

}  // namespace strategy
