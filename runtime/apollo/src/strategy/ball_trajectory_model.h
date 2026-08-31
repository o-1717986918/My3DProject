// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <limits>

namespace strategy {

/// Conservative rolling-ball model for the current safe contact kick. The
/// defaults are seeded by the measured 1.43 m/s match contact and MuJoCo rolling
/// friction; calibration artifacts can replace them without changing planning.
class BallTrajectoryModel {
public:
    struct Parameters {
        double default_initial_speed_mps{1.43};
        double rolling_deceleration_mps2{0.08};
        double minimum_control_speed_mps{0.20};
    };

    BallTrajectoryModel();
    explicit BallTrajectoryModel(Parameters parameters);

    double travel_time_s(double distance_m, double initial_speed_mps) const;
    double arrival_speed_mps(double distance_m, double initial_speed_mps) const;
    double max_controlled_distance_m(double initial_speed_mps) const;
    const Parameters& parameters() const { return parameters_; }

private:
    Parameters parameters_;
};

}  // namespace strategy
