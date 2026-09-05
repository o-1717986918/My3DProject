import jax
import jax.numpy as jp
import numpy as np

from my3d_rl.contract import load_policy_contract
from my3d_rl.striker_cpu import (
    StrikerCpuEvaluator,
    closed_loop_approach_control_numpy,
)
from my3d_rl.striker_env import (
    DEFAULT_CONTRACT,
    LongHorizonStriker,
    closed_loop_approach_control,
    default_config,
)


def _control(ball_local_xy, target_local):
    return closed_loop_approach_control(
        jp.asarray(ball_local_xy),
        jp.asarray(target_local),
        standoff=0.31,
        ball_lateral=-0.04,
        position_gain=2.0,
        lateral_gain=2.5,
        yaw_gain=2.0,
        max_forward_speed=0.70,
        max_backward_speed=0.15,
        max_lateral_speed=0.25,
        max_yaw_speed=0.55,
        activation_radius=0.28,
        full_radius=0.07,
        heading_radius=0.45,
        full_heading=0.10,
    )


def test_approach_controller_hands_over_at_the_contact_pose():
    command, activation, error, heading = _control([0.31, -0.04], [1.0, 0.0])

    np.testing.assert_allclose(error, np.zeros(2), atol=1.0e-6)
    np.testing.assert_allclose(command, np.zeros(3), atol=1.0e-6)
    assert float(activation) == 1.0
    assert float(heading) == 0.0


def test_approach_controller_keeps_kick_disabled_while_far_away():
    command, activation, error, _ = _control([0.81, -0.04], [1.0, 0.0])

    np.testing.assert_allclose(error, [0.50, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(command, [0.70, 0.0, 0.0], atol=1.0e-6)
    assert float(activation) == 0.0


def test_striker_default_release_gate_is_bounded_but_not_perfect_pose_only():
    config = default_config()

    assert config.kick_trigger_threshold < 1.0
    assert config.kick_settled_confirmation_steps <= 5
    assert config.kick_full_radius < config.kick_settled_distance
    assert config.kick_full_heading < config.kick_settled_heading
    assert config.learned_approach_residual_floor == 0.0


def test_numpy_and_jax_approach_controllers_match():
    kwargs = {
        "standoff": 0.31,
        "ball_lateral": -0.04,
        "position_gain": 2.0,
        "lateral_gain": 2.5,
        "yaw_gain": 2.0,
        "max_forward_speed": 0.70,
        "max_backward_speed": 0.15,
        "max_lateral_speed": 0.25,
        "max_yaw_speed": 0.55,
        "activation_radius": 0.28,
        "full_radius": 0.07,
        "heading_radius": 0.45,
        "full_heading": 0.10,
    }
    ball = np.array([0.43, -0.09])
    target = np.array([0.96, 0.28])

    jax_result = closed_loop_approach_control(
        jp.asarray(ball), jp.asarray(target), **kwargs
    )
    numpy_result = closed_loop_approach_control_numpy(ball, target, **kwargs)

    for actual, expected in zip(jax_result, numpy_result, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2.0e-7)


def test_striker_environment_preserves_student_and_teacher_boundaries():
    env = LongHorizonStriker(
        config_overrides={
            "episode_length": 4,
            "robot_distance_range": [0.80, 0.80],
            "robot_lateral_range": [0.0, 0.0],
            "robot_yaw_noise_range": [0.0, 0.0],
            "target_angle_range": [0.0, 0.0],
            "target_distance_range": [2.0, 2.0],
            "fixed_action_mode": 0,
            "fixed_desired_arrival_speed": 0.8,
        }
    )
    state = env.reset(jax.random.PRNGKey(19))

    assert state.obs["state"].shape == (102,)
    assert state.obs["teacher_state"].shape == (138,)
    assert state.obs["privileged_state"].shape == (138,)
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert 0.45 < float(state.metrics["diagnostic/contact_distance"]) < 0.55

    stepped = env.step(state, jp.zeros(env.action_size))
    assert np.isfinite(float(stepped.reward))
    assert np.isfinite(np.asarray(stepped.obs["teacher_state"])).all()
    assert int(stepped.info["step"]) == 1


def test_exact_cpu_striker_runs_the_same_versioned_controller_surface():
    contract = load_policy_contract(DEFAULT_CONTRACT)
    config = default_config()
    config.episode_length = 2
    config.robot_distance_range = [0.80, 0.80]
    config.robot_lateral_range = [0.0, 0.0]
    config.robot_yaw_noise_range = [0.0, 0.0]
    config.target_angle_range = [0.0, 0.0]
    config.target_distance_range = [2.0, 2.0]
    config.reset_joint_noise = 0.0
    config.reset_root_velocity_noise = 0.0
    evaluator = StrikerCpuEvaluator(
        contract,
        np.zeros((2, contract.action_size), dtype=np.float32),
        config.to_dict(),
        prefix="test_striker_cpu_",
    )

    result = evaluator.rollout(23)

    assert result.episode_steps == 2
    assert not result.triggered
    assert not result.contacted
    assert not result.fallen
    assert np.isfinite(result.final_contact_distance_m)
