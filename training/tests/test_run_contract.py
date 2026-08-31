from pathlib import Path

import jax
import numpy as np
import yaml

from my3d_rl import load_policy_contract
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.run_env import (
    NOMINAL_TRAINING_POSE,
    TRAIN_TO_SERVER_SIGN,
    DirectionalRun,
)


CONTRACT = Path(__file__).parents[1] / "contracts" / "run_policy_v1.yaml"
PHASE_CONTRACT = Path(__file__).parents[1] / "contracts" / "run_policy_v2.yaml"
REFERENCE_CONTRACT = Path(__file__).parents[1] / "contracts" / "run_policy_v3.yaml"


def _write_reference_residual_fixture(path: Path) -> None:
    frames = 4
    joint_position = np.zeros((frames, 23), dtype=np.float32)
    joint_position[:, 0] = np.arange(frames, dtype=np.float32) * 0.01
    np.savez(
        path,
        root_position=np.tile(
            np.array([0.0, 0.0, 0.61], dtype=np.float32), (frames, 1)
        ),
        root_quaternion_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)
        ),
        root_linear_velocity=np.tile(
            np.array([3.2, 0.0, 0.0], dtype=np.float32), (frames, 1)
        ),
        root_angular_velocity=np.zeros((frames, 3), dtype=np.float32),
        joint_position=joint_position,
        joint_velocity=np.full((frames, 23), 1.6, dtype=np.float32),
        foot_contact=np.ones((frames, 2), dtype=bool),
    )


def test_run_policy_contract_preserves_runtime_boundary():
    contract = load_policy_contract(CONTRACT)

    assert contract.policy_name == "run_policy_v1"
    assert contract.frequency_hz == 50
    assert contract.observation_size == 78
    assert contract.action_size == 23
    assert contract.input_shape == (1, 78)
    assert contract.output_shape == (1, 23)
    assert contract.action_clip == (-10.0, 10.0)
    assert contract.action_scale == 0.5
    assert contract.kp == 25.0
    assert contract.kd == 0.6
    assert sum(size for _, size in contract.observation_fields) == 78


def test_runtime_pose_and_sign_tables_cover_all_effectors():
    expected_runtime_sign = np.array(
        [
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ],
        dtype=np.float32,
    )

    assert NOMINAL_TRAINING_POSE.shape == (23,)
    assert TRAIN_TO_SERVER_SIGN.shape == (23,)
    np.testing.assert_array_equal(TRAIN_TO_SERVER_SIGN, expected_runtime_sign)

    raw = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    np.testing.assert_array_equal(
        raw["decoder"]["training_to_server_sign"], expected_runtime_sign
    )
    assert raw["control"]["action_clip"] == [-10.0, 10.0]


def test_run_environment_uses_exact_control_period_and_safe_targets():
    env = DirectionalRun()

    assert env.n_substeps == 4
    assert env.action_size == 23
    assert env.observation_size == {"state": (78,), "privileged_state": (84,)}
    assert np.all(np.asarray(env._nominal_physical) >= np.asarray(env._lowers))
    assert np.all(np.asarray(env._nominal_physical) <= np.asarray(env._uppers))
    assert np.isclose(np.asarray(env._nominal_physical)[15], -0.4)
    assert np.isclose(np.asarray(env._nominal_physical)[21], -0.4)
    np.testing.assert_allclose(env._left_foot_half_size, [0.1115, 0.05, 0.015])
    np.testing.assert_allclose(env._right_foot_half_size, [0.1115, 0.05, 0.015])
    assert np.isclose(env._config.foot_contact_tolerance, 0.01)

    decoded = np.asarray(env.decode_action_targets(jax.numpy.zeros(23)))
    np.testing.assert_allclose(decoded, np.asarray(env._nominal_physical))


def test_phase_policy_contract_extends_actor_without_changing_actions():
    contract = load_policy_contract(PHASE_CONTRACT)
    env = DirectionalRun(contract=contract)

    assert contract.policy_name == "run_policy_v2"
    assert contract.observation_size == 80
    assert contract.input_shape == (1, 80)
    assert contract.output_shape == (1, 23)
    assert contract.observation_fields[-1] == ("gait_phase_cos_sin", 2)
    assert env.observation_size == {"state": (80,), "privileged_state": (86,)}


def test_reference_residual_contract_requires_external_motion(tmp_path):
    contract = load_policy_contract(REFERENCE_CONTRACT)

    assert contract.control_mode == "motion_reference_residual_joint_position"
    assert contract.action_clip == (-1.0, 1.0)
    assert contract.action_scale == 0.15
    assert contract.reference_sha256 == (
        "ab81912570d746965162f1d84cfd6d215a1265bd28dfc2d371c72f095aa40f9a"
    )
    with np.testing.assert_raises_regex(ValueError, "requires a motion reference"):
        DirectionalRun(contract=contract)


def test_reference_residual_zero_action_reconstructs_phase_target(tmp_path):
    path = tmp_path / "reference-residual.npz"
    _write_reference_residual_fixture(path)
    env = DirectionalRun(
        contract=load_policy_contract(REFERENCE_CONTRACT), motion_reference=path
    )

    phase = jax.numpy.array(0.125)
    decoded = np.asarray(env.decode_action_targets(jax.numpy.zeros(23), phase))

    expected = np.zeros(23)
    expected[0] = 0.005
    np.testing.assert_allclose(decoded, expected, atol=1.0e-7)
    assert np.isclose(env._config.action_scale, 0.15)
    with np.testing.assert_raises_regex(ValueError, "requires gait_phase"):
        env.decode_action_targets(jax.numpy.zeros(23))


def test_reference_residual_reset_scales_cadence_velocity_and_control(tmp_path):
    path = tmp_path / "reference-residual-reset.npz"
    _write_reference_residual_fixture(path)
    env = DirectionalRun(
        config_overrides={
            "use_fixed_command": True,
            "fixed_command": [1.8, 0.0, 0.0],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "reference_init_probability": 1.0,
        },
        contract=load_policy_contract(REFERENCE_CONTRACT),
        motion_reference=path,
    )

    state = jax.jit(env.reset)(jax.random.PRNGKey(71))
    phase = state.info["gait_phase"]
    expected_ctrl = env.decode_action_targets(jax.numpy.zeros(23), phase)

    np.testing.assert_allclose(
        np.asarray(state.data.ctrl)[env._pos_actuator],
        np.asarray(expected_ctrl),
        atol=1.0e-6,
    )
    assert np.isclose(np.asarray(state.info["gait_frequency"]), 7.03125)
    assert np.isclose(np.asarray(state.data.qvel)[env._root_dof], 1.8, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(state.obs["state"])[:69], 0.0, atol=1.0e-6)


def test_formal_profile_uses_bounded_action_and_official_t1_widths():
    profile = get_ppo_profile("t1_tanh_v1")

    assert profile.distribution_type == "tanh_normal"
    assert profile.policy_hidden_layer_sizes == (512, 256, 128)
    assert profile.unroll_length == 20
    assert profile.num_updates_per_batch == 4


def test_motion_transfer_profile_has_explicit_conservative_kl_bounds():
    profile = get_ppo_profile("legacy_motion_track_v3")

    assert profile.policy_contract == "run_policy_v2"
    assert profile.num_updates_per_batch == 1
    assert profile.learning_rate_min <= profile.learning_rate
    assert profile.learning_rate <= profile.learning_rate_max
    assert profile.learning_rate_max < 1.0e-5
    assert profile.desired_kl == 0.002


def test_motion_reset_initializes_complete_reference_state(tmp_path):
    frames = 4
    root_position = np.zeros((frames, 3), dtype=np.float32)
    root_position[:, 2] = 0.61
    root_linear_velocity = np.tile(
        np.array([1.7, 0.0, 0.1], dtype=np.float32), (frames, 1)
    )
    root_angular_velocity = np.tile(
        np.array([0.0, 0.2, 0.0], dtype=np.float32), (frames, 1)
    )
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        root_position=root_position,
        root_quaternion_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)
        ),
        root_linear_velocity=root_linear_velocity,
        root_angular_velocity=root_angular_velocity,
        joint_position=np.zeros((frames, 23), dtype=np.float32),
        joint_velocity=np.zeros((frames, 23), dtype=np.float32),
        foot_contact=np.ones((frames, 2), dtype=bool),
    )
    env = DirectionalRun(
        config_overrides={
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "reference_init_probability": 1.0,
        },
        contract=load_policy_contract(PHASE_CONTRACT),
        motion_reference=path,
    )

    # Eager MJX executes hundreds of tiny XLA operations and is unsuitable for
    # CI; compile the reset as production training does.
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))

    assert np.isclose(np.asarray(state.data.qpos)[env._root_qpos + 2], 0.61)
    np.testing.assert_allclose(
        np.asarray(state.data.qpos)[env._root_qpos + 3 : env._root_qpos + 7],
        [1.0, 0.0, 0.0, 0.0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(state.data.qvel)[env._root_dof : env._root_dof + 6],
        [1.7, 0.0, 0.1, 0.0, 0.2, 0.0],
        atol=1.0e-6,
    )
