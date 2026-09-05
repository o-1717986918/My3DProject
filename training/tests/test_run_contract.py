from pathlib import Path

import jax
import numpy as np
import yaml
from brax.training import types as brax_types
from brax.training.acme import running_statistics

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
GMR_REFERENCE_CONTRACT = Path(__file__).parents[1] / "contracts" / "run_policy_v4.yaml"


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


def test_axis_aligned_command_sampler_zeroes_two_command_axes():
    env = DirectionalRun(
        config_overrides={
            "axis_aligned_command_probability": 1.0,
            "stand_probability": 0.0,
            "lin_vel_x": [-0.25, 1.65],
            "lin_vel_y": [-0.45, 0.45],
            "ang_vel_yaw": [-0.75, 0.75],
        }
    )
    commands = np.asarray(
        jax.vmap(env._sample_command)(jax.random.split(jax.random.PRNGKey(91), 256))
    )

    assert np.all(np.sum(np.abs(commands) > 1.0e-7, axis=1) <= 1)
    assert np.any(np.abs(commands[:, 0]) > 1.0e-7)
    assert np.any(np.abs(commands[:, 1]) > 1.0e-7)
    assert np.any(np.abs(commands[:, 2]) > 1.0e-7)


def test_axis_sampler_skips_disabled_axis_and_can_bias_weak_turn_direction():
    env = DirectionalRun(
        config_overrides={
            "axis_aligned_command_probability": 1.0,
            "axis_command_weights": [0.0, 0.0, 1.0],
            "stand_probability": 0.0,
            "lin_vel_x": [0.15, 1.55],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_yaw": [-0.9, 0.9],
            "minimum_abs_yaw": 0.25,
            "yaw_negative_probability": 1.0,
        }
    )
    commands = np.asarray(
        jax.vmap(env._sample_command)(jax.random.split(jax.random.PRNGKey(93), 128))
    )

    np.testing.assert_allclose(commands[:, :2], 0.0)
    assert np.all(commands[:, 2] <= -0.25)


def test_command_sampler_rejects_invalid_axis_weights():
    with np.testing.assert_raises_regex(ValueError, "axis_command_weights"):
        DirectionalRun(config_overrides={"axis_command_weights": [1.0, 0.0]})


def test_pure_yaw_advances_phase_and_reports_planted_foot_slip():
    env = DirectionalRun(
        config_overrides={
            "use_fixed_command": True,
            "fixed_command": [0.0, 0.0, 0.75],
            "gait_frequency": [1.5, 1.5],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
            "reward.foot_slip": -0.03,
        },
        contract=load_policy_contract(PHASE_CONTRACT),
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(92))

    assert np.isclose(np.asarray(state.info["gait_frequency"]), 1.5)
    assert state.info["last_foot_positions"].shape == (2, 3)
    initial_phase = float(np.asarray(state.info["gait_phase"]))
    next_state = jax.jit(env.step)(state, jax.numpy.zeros(23))
    assert not np.isclose(
        float(np.asarray(next_state.info["gait_phase"])), initial_phase
    )
    assert np.isfinite(np.asarray(next_state.metrics["cost/foot_slip"]))


def test_transition_reset_initializes_coherent_policy_state():
    env = DirectionalRun(
        config_overrides={
            "use_fixed_command": True,
            "fixed_command": [1.5, 0.0, 0.0],
            "reset_joint_noise": 0.0,
            "reset_joint_velocity_noise": 0.8,
            "reset_policy_action_noise": 1.25,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
        },
        contract=load_policy_contract(PHASE_CONTRACT),
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(94))
    action = np.asarray(state.info["last_action"])
    positions = np.asarray(state.data.qpos)[env._joint_qpos]
    velocities = np.asarray(state.data.qvel)[env._joint_dof]
    expected = np.asarray(
        env.decode_action_targets(state.info["last_action"], state.info["gait_phase"])
    )

    assert np.any(np.abs(action) > 1.0e-5)
    np.testing.assert_allclose(positions, expected, atol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(state.data.ctrl)[env._pos_actuator], expected, atol=1.0e-6
    )
    np.testing.assert_allclose(
        state.info["last_last_action"], state.info["last_action"], atol=1.0e-6
    )
    assert np.any(np.abs(velocities) > 1.0e-5)
    assert np.all(np.abs(velocities) <= 0.8 + 1.0e-6)


def test_transition_reset_rejects_invalid_noise():
    with np.testing.assert_raises_regex(ValueError, "reset_policy_action_noise"):
        DirectionalRun(config_overrides={"reset_policy_action_noise": -0.1})


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


def test_gmr_reference_contract_changes_only_the_pinned_reference() -> None:
    previous = load_policy_contract(REFERENCE_CONTRACT)
    gmr = load_policy_contract(GMR_REFERENCE_CONTRACT)

    assert gmr.policy_name == "run_policy_v4"
    assert gmr.reference_sha256 == (
        "02cd640919d81f0417246559bae491439e7afbfda614039d5ecae1293076c523"
    )
    assert gmr.joint_order == previous.joint_order
    assert gmr.observation_fields == previous.observation_fields
    assert gmr.action_clip == previous.action_clip
    assert gmr.action_scale == previous.action_scale
    assert gmr.kp == 50.0
    assert gmr.kd == 1.2


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


def test_reference_residual_reset_accepts_fixed_phase_weights(tmp_path):
    path = tmp_path / "reference-residual-weighted-reset.npz"
    _write_reference_residual_fixture(path)
    env = DirectionalRun(
        config_overrides={
            "use_fixed_command": True,
            "fixed_command": [1.8, 0.0, 0.0],
            "reference_init_probability": 1.0,
            "reference_phase_sampling_weights": [0.0, 0.0, 1.0, 0.0],
        },
        contract=load_policy_contract(REFERENCE_CONTRACT),
        motion_reference=path,
    )

    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(73), 16))
    phases = np.asarray(states.info["gait_phase"])
    assert np.all(phases >= 0.5)
    assert np.all(phases < 0.75)


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


def test_soccer_phase_profile_cannot_raise_learning_rate_above_bound():
    historical = get_ppo_profile("legacy_phase_warmstart_v2")
    profile = get_ppo_profile("legacy_phase_soccer_v3")

    assert profile.policy_contract == historical.policy_contract
    assert profile.factory_kind == historical.factory_kind
    assert profile.policy_hidden_layer_sizes == historical.policy_hidden_layer_sizes
    assert profile.learning_rate_min <= profile.learning_rate
    assert profile.learning_rate <= profile.learning_rate_max
    assert profile.learning_rate_max <= 5.0e-6
    assert profile.desired_kl < historical.desired_kl


def test_soccer_motion_v4_resume_is_conservative_and_checkpoint_compatible():
    aggressive = get_ppo_profile("soccer_motion_residual_v3")
    profile = get_ppo_profile("soccer_motion_residual_v4")

    assert profile.policy_contract == aggressive.policy_contract
    assert profile.policy_hidden_layer_sizes == aggressive.policy_hidden_layer_sizes
    assert profile.value_hidden_layer_sizes == aggressive.value_hidden_layer_sizes
    assert profile.distribution_type == aggressive.distribution_type
    assert profile.normalize_observations == aggressive.normalize_observations
    assert profile.num_updates_per_batch == 1
    assert profile.learning_rate == 1.0e-5
    assert profile.learning_rate_max == 2.0e-5
    assert profile.desired_kl == 0.002


def test_reference_residual_profile_starts_at_zero_with_low_noise():
    rejected_profile = get_ppo_profile("reference_residual_v1")
    profile = get_ppo_profile("reference_residual_v2")
    sizes = {"state": (80,), "privileged_state": (86,)}
    networks = profile.network_factory()(
        sizes,
        23,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )
    params = networks.policy_network.init(jax.random.PRNGKey(19))
    spec = {
        key: jax.ShapeDtypeStruct(shape, jax.numpy.float32)
        for key, shape in sizes.items()
    }
    normalizer = running_statistics.init_state(spec)
    mean, standard_deviation = networks.policy_network.apply(
        normalizer,
        params,
        {
            "state": jax.numpy.zeros((2, 80)),
            "privileged_state": jax.numpy.zeros((2, 86)),
        },
    )

    assert rejected_profile.distribution_type == "tanh_normal"
    assert rejected_profile.learning_rate == 1.0e-4
    assert profile.distribution_type == "normal"
    assert profile.zero_mean_init
    np.testing.assert_allclose(mean, 0.0)
    np.testing.assert_allclose(standard_deviation, 0.1)


def test_reference_residual_exploration_profile_keeps_zero_mean():
    profile = get_ppo_profile("reference_residual_v3")
    sizes = {"state": (80,), "privileged_state": (86,)}
    networks = profile.network_factory()(
        sizes,
        23,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )
    params = networks.policy_network.init(jax.random.PRNGKey(23))
    spec = {
        key: jax.ShapeDtypeStruct(shape, jax.numpy.float32)
        for key, shape in sizes.items()
    }
    normalizer = running_statistics.init_state(spec)
    mean, standard_deviation = networks.policy_network.apply(
        normalizer,
        params,
        {
            "state": jax.numpy.zeros((2, 80)),
            "privileged_state": jax.numpy.zeros((2, 86)),
        },
    )

    assert profile.zero_mean_init
    assert profile.num_updates_per_batch == 5
    assert profile.normalize_observations
    assert profile.desired_kl == 0.01
    np.testing.assert_allclose(mean, 0.0)
    np.testing.assert_allclose(standard_deviation, 0.5)


def test_gmr_reference_profile_uses_v4_contract() -> None:
    profile = get_ppo_profile("reference_residual_v4")

    assert profile.policy_contract == "run_policy_v4"
    assert profile.zero_mean_init
    assert profile.num_updates_per_batch == 5
    assert profile.init_noise_std == 0.5


def test_reference_curriculum_profile_is_transfer_safe() -> None:
    profile = get_ppo_profile("reference_curriculum_v5")

    assert profile.policy_contract == "run_policy_v4"
    assert profile.zero_mean_init
    assert not profile.normalize_observations
    assert profile.num_updates_per_batch == 3
    assert profile.init_noise_std == 0.3
    assert profile.learning_rate_max == 1.0e-4
    assert profile.desired_kl == 0.005


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
