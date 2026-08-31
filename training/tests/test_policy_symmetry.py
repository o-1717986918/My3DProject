import numpy as np

from my3d_rl.policy_symmetry import (
    mirror_run_action,
    mirror_run_observation,
    training_mirror_map,
)
from my3d_rl.run_env import TRAIN_TO_SERVER_SIGN


JOINT_ORDER = (
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)


def test_t1_reflection_is_an_involution_for_actions_and_observations():
    source, factor = training_mirror_map(JOINT_ORDER, TRAIN_TO_SERVER_SIGN)
    rng = np.random.default_rng(303)
    action = rng.normal(size=(5, 23)).astype(np.float32)
    observation = rng.normal(size=(5, 80)).astype(np.float32)

    np.testing.assert_allclose(
        mirror_run_action(mirror_run_action(action, source, factor), source, factor),
        action,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        mirror_run_observation(
            mirror_run_observation(observation, source, factor), source, factor
        ),
        observation,
        atol=1.0e-6,
    )
