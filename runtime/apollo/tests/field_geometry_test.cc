// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/decision/field_geometry.h"

#include <cmath>
#include <iostream>

int main() {
    namespace geometry = decision::field_geometry;

    for (int player = 2; player <= 7; ++player) {
        const geometry::Pose3 pose = geometry::player_defensive_kickoff_beam_pose(player);
        const geometry::Position2 position{pose[0], pose[1]};

        if (position[0] >= 0.0) {
            std::cerr << "player " << player << " is not in its own half\n";
            return 1;
        }
        if (std::hypot(position[0], position[1]) <= geometry::kCenterCircleRadiusM) {
            std::cerr << "player " << player << " is inside the center circle\n";
            return 1;
        }
        if (geometry::is_in_our_goalie_area(position)) {
            std::cerr << "field player " << player << " is inside our goalkeeper area\n";
            return 1;
        }
    }

    const geometry::Pose3 deep_guard = geometry::player_defensive_kickoff_beam_pose(6);
    const double goalkeeper_area_boundary_x =
        -geometry::kActualHalfLengthM + geometry::kGoalieAreaDepthM;
    const double clearance = deep_guard[0] - goalkeeper_area_boundary_x;
    if (clearance < 1.29 || clearance > 1.31) {
        std::cerr << "unexpected defensive-kickoff goalkeeper-area clearance: "
                  << clearance << '\n';
        return 1;
    }

    return 0;
}
