// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

namespace robot::t1_joint_limits {

// Canonical Apollo policy joint order. Keep the hard safety limits in one
// place so deterministic and ONNX-backed motion runners clamp identically.
inline constexpr std::array<double, 23> kLowerRad{
    -1.57, -0.35, -3.31, -1.74, -2.27, -2.44, -3.31, -1.57,
    -2.27, 0.0, -1.57, -1.8, -0.2, -1.0, 0.0, -0.87,
    -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44,
};

inline constexpr std::array<double, 23> kUpperRad{
    1.57, 1.22, 1.22, 1.57, 2.27, 0.0, 1.22, 1.74,
    2.27, 2.44, 1.57, 1.57, 1.57, 1.0, 2.34, 0.35,
    0.44, 1.57, 0.2, 1.0, 2.34, 0.35, 0.44,
};

}  // namespace robot::t1_joint_limits
