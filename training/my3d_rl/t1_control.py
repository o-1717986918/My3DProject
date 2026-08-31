"""Shared Booster T1 control constants for kick training and deployment."""

from __future__ import annotations

import numpy as np


APOLLO_DEFAULT_POSE = np.array(
    [
        0.0,
        0.0,
        0.0,
        -1.4,
        0.0,
        -0.4,
        0.0,
        1.4,
        0.0,
        0.4,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
    ],
    dtype=np.float64,
)

KICK_ACTION_SCALE = np.array(
    [
        0.10,
        0.10,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.15,
        0.35,
        0.25,
        0.25,
        0.45,
        0.25,
        0.20,
        0.35,
        0.25,
        0.25,
        0.45,
        0.25,
        0.20,
    ],
    dtype=np.float64,
)


def apollo_joint_gains(name: str) -> tuple[float, float]:
    """Return the exact per-joint gains used by Apollo's runtime adapter."""
    if name == "AAHead_yaw":
        return 10.0, 1.0
    if name == "Head_pitch":
        return 20.0, 1.0
    if name == "Waist":
        return 85.0, 5.0
    if "Shoulder" in name:
        return 45.0, 2.5
    if "Elbow" in name:
        return 30.0, 1.2
    if "Hip_Pitch" in name:
        return 130.0, 10.0
    if "Hip_Roll" in name:
        return 90.0, 8.0
    if "Hip_Yaw" in name:
        return 70.0, 3.0
    if "Knee" in name:
        return 140.0, 6.0
    if "Ankle_Pitch" in name:
        return 45.0, 2.0
    if "Ankle_Roll" in name:
        return 40.0, 1.8
    return 10.0, 0.1
