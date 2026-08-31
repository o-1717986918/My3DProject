"""Left/right reflection operators for T1 run-policy observations and actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


MIRROR_JOINT = {
    "AAHead_yaw": ("AAHead_yaw", -1.0),
    "Head_pitch": ("Head_pitch", 1.0),
    "Left_Shoulder_Pitch": ("Right_Shoulder_Pitch", 1.0),
    "Left_Shoulder_Roll": ("Right_Shoulder_Roll", -1.0),
    "Left_Elbow_Pitch": ("Right_Elbow_Pitch", 1.0),
    "Left_Elbow_Yaw": ("Right_Elbow_Yaw", -1.0),
    "Right_Shoulder_Pitch": ("Left_Shoulder_Pitch", 1.0),
    "Right_Shoulder_Roll": ("Left_Shoulder_Roll", -1.0),
    "Right_Elbow_Pitch": ("Left_Elbow_Pitch", 1.0),
    "Right_Elbow_Yaw": ("Left_Elbow_Yaw", -1.0),
    "Waist": ("Waist", -1.0),
    "Left_Hip_Pitch": ("Right_Hip_Pitch", 1.0),
    "Left_Hip_Roll": ("Right_Hip_Roll", -1.0),
    "Left_Hip_Yaw": ("Right_Hip_Yaw", -1.0),
    "Left_Knee_Pitch": ("Right_Knee_Pitch", 1.0),
    "Left_Ankle_Pitch": ("Right_Ankle_Pitch", 1.0),
    "Left_Ankle_Roll": ("Right_Ankle_Roll", -1.0),
    "Right_Hip_Pitch": ("Left_Hip_Pitch", 1.0),
    "Right_Hip_Roll": ("Left_Hip_Roll", -1.0),
    "Right_Hip_Yaw": ("Left_Hip_Yaw", -1.0),
    "Right_Knee_Pitch": ("Left_Knee_Pitch", 1.0),
    "Right_Ankle_Pitch": ("Left_Ankle_Pitch", 1.0),
    "Right_Ankle_Roll": ("Left_Ankle_Roll", -1.0),
}


def training_mirror_map(
    joint_order: Sequence[str], training_to_server_sign: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices and signs for reflection in the training frame."""
    names = tuple(joint_order)
    signs = np.asarray(training_to_server_sign, dtype=np.float32)
    if signs.shape != (len(names),):
        raise ValueError("training_to_server_sign has incompatible shape")
    if set(names) != set(MIRROR_JOINT):
        raise ValueError("joint order does not match the T1 reflection registry")
    lookup = {name: index for index, name in enumerate(names)}
    source = np.array([lookup[MIRROR_JOINT[name][0]] for name in names])
    physical_factor = np.array(
        [MIRROR_JOINT[name][1] for name in names], dtype=np.float32
    )
    training_factor = physical_factor * signs / signs[source]
    return source, training_factor


def mirror_run_action(
    action: np.ndarray, source: np.ndarray, factor: np.ndarray
) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape[-1] != len(source):
        raise ValueError("run action must end in 23 values")
    return values[..., source] * factor


def mirror_run_observation(
    observation: np.ndarray, source: np.ndarray, factor: np.ndarray
) -> np.ndarray:
    """Reflect a 78/80-value run observation and exchange left/right gait phase."""
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[-1] not in (78, 80):
        raise ValueError("run observation must end in 78 or 80 values")
    result = values.copy()
    joint_features = values[..., :69].reshape(values.shape[:-1] + (23, 3))
    result[..., :69] = (joint_features[..., source, :] * factor[:, None]).reshape(
        values.shape[:-1] + (69,)
    )
    result[..., 69:72] = values[..., 69:72] * np.array([-1.0, 1.0, -1.0])
    result[..., 72:75] = values[..., 72:75] * np.array([1.0, -1.0, -1.0])
    result[..., 75:78] = values[..., 75:78] * np.array([1.0, -1.0, 1.0])
    if values.shape[-1] == 80:
        result[..., 78:80] = -values[..., 78:80]
    return result
