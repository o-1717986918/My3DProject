from pathlib import Path

import numpy as np
import pytest

from my3d_rl import load_policy_contract
from my3d_rl.rcss_scene import RcssKickScene, build_single_t1_soccer_model


CONTRACT = Path(__file__).parents[1] / "contracts" / "kick_policy_v1.yaml"


def test_rcss_scene_compiles_with_exact_ball_and_timestep():
    model = build_single_t1_soccer_model()

    assert model.opt.timestep == 0.005
    assert model.geom("ball").size[0] == 0.11
    assert model.body("ball").mass[0] == 0.41
    assert model.joint("train_Left_Knee_Pitch").range.tolist() == [0.0, 2.34]


def test_pd_surface_runs_one_50hz_control_step():
    contract = load_policy_contract(CONTRACT)
    scene = RcssKickScene(contract)
    before = scene.joint_state()
    after = scene.step_joint_targets(np.zeros(23), kp=20.0, kd=0.5)

    assert scene.n_substeps == 4
    assert before.position.shape == (23,)
    assert after.velocity.shape == (23,)
    assert np.all(np.isfinite(after.position))


def test_pd_surface_preserves_server_velocity_and_feedforward_controls():
    scene = RcssKickScene(load_policy_contract(CONTRACT))
    velocities = np.full(23, 0.2)
    torques = np.full(23, 0.1)

    scene.step_joint_targets(
        np.zeros(23), kp=20.0, kd=0.5,
        target_velocity_rad_s=velocities, feedforward_tau=torques,
    )

    np.testing.assert_allclose(scene.data.ctrl[scene._vel_actuator], velocities)
    np.testing.assert_allclose(scene.data.ctrl[scene._tau_actuator], torques)
    with pytest.raises(ValueError, match="velocity/torque"):
        scene.step_joint_targets(
            np.zeros(23), kp=20.0, kd=0.5,
            target_velocity_rad_s=np.zeros(22),
        )
