from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort
from brax.training import types as brax_types
from brax.training.acme import running_statistics

from my3d_rl.apollo_run_env import ApolloWaypointRun, apollo_waypoint_command
from my3d_rl.contract import load_policy_contract
from my3d_rl.legacy_policy import load_apollo_onnx_teacher_params
from my3d_rl.ppo_profile import get_ppo_profile
from my3d_rl.t1_control import APOLLO_DEFAULT_POSE, apollo_joint_gains


CONTRACT = (
    Path(__file__).parents[1] / "contracts" / "apollo_walk_policy_v1.yaml"
)
REPOSITORY_ROOT = Path(__file__).parents[2]
WALK_POLICY = (
    REPOSITORY_ROOT
    / "runtime"
    / "apollo_rebuild"
    / "assets"
    / "networks"
    / "walk"
    / "policy.onnx"
)


def test_apollo_walk_contract_matches_frozen_cpp_boundary():
    contract = load_policy_contract(CONTRACT)

    assert contract.policy_name == "apollo_walk_policy_v1"
    assert contract.observation_size == 78
    assert contract.action_size == 23
    assert contract.action_clip == (-5.0, 5.0)
    assert contract.action_scale == 0.25
    assert contract.kp is None and contract.kd is None
    assert contract.gain_profile == "apollo_runtime_per_joint"
    assert contract.observation_fields == (
        ("body_angular_velocity", 3),
        ("projected_gravity", 3),
        ("velocity_command", 3),
        ("joint_position_offset", 23),
        ("joint_velocity", 23),
        ("previous_action", 23),
    )


def test_waypoint_command_matches_cpp_clamps_and_body_rotation():
    origin = jp.zeros(2)

    behind = np.asarray(
        apollo_waypoint_command(origin, jp.array(0.0), jp.array([-5.0, 0.0]))
    )
    left = np.asarray(
        apollo_waypoint_command(origin, jp.array(0.0), jp.array([0.0, 5.0]))
    )
    world_left_body_forward = np.asarray(
        apollo_waypoint_command(
            origin, jp.array(np.pi / 2.0), jp.array([0.0, 5.0])
        )
    )

    np.testing.assert_allclose(behind, [-0.5, 0.0, 0.5], atol=1.0e-6)
    np.testing.assert_allclose(left, [0.0, 0.5, 0.1 * np.pi], atol=1.0e-6)
    np.testing.assert_allclose(
        world_left_body_forward, [1.0, 0.0, 0.0], atol=1.0e-6
    )
    np.testing.assert_allclose(
        apollo_waypoint_command(
            origin, jp.array(0.0), jp.array([0.1, 0.0]), stop_radius_m=0.25
        ),
        np.zeros(3),
        atol=1.0e-7,
    )


def test_apollo_waypoint_environment_uses_runtime_pose_gains_and_observation():
    env = ApolloWaypointRun(
        config_overrides={
            "waypoint_distance_range": [3.0, 3.0],
            "waypoint_bearing_range": [np.pi / 2.0, np.pi / 2.0],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
        }
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(20_261_401))
    observation = np.asarray(state.obs["state"])

    assert env.observation_size == {"state": (78,), "privileged_state": (84,)}
    np.testing.assert_allclose(
        env.decode_action_targets(jp.zeros(23)), APOLLO_DEFAULT_POSE, atol=1.0e-7
    )
    np.testing.assert_allclose(
        observation[6:9], [0.0, 0.5, 0.1 * np.pi], atol=1.0e-6
    )
    np.testing.assert_allclose(observation[9:11], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(observation[32:34], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(observation[55:57], 0.0, atol=1.0e-7)
    assert np.isclose(float(state.info["waypoint_distance"]), 3.0, atol=1.0e-5)

    hip_name = "Left_Hip_Pitch"
    hip_effector = "lle1"
    position_id = env.mj_model.actuator(env.prefix + hip_effector + "_pos").id
    velocity_id = env.mj_model.actuator(env.prefix + hip_effector + "_vel").id
    expected_kp, expected_kd = apollo_joint_gains(hip_name)
    assert np.isclose(env.mj_model.actuator_gainprm[position_id, 0], expected_kp)
    assert np.isclose(env.mj_model.actuator_gainprm[velocity_id, 0], expected_kd)


def test_apollo_waypoint_step_recomputes_command_and_is_finite():
    env = ApolloWaypointRun(
        config_overrides={
            "waypoint_distance_range": [2.0, 2.0],
            "waypoint_bearing_range": [0.0, 0.0],
            "reset_joint_noise": 0.0,
            "reset_root_velocity_noise": 0.0,
            "reset_yaw_range": 0.0,
        }
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(20_261_402))
    next_state = jax.jit(env.step)(state, jp.zeros(23))

    assert np.isfinite(np.asarray(next_state.obs["state"])).all()
    assert np.isfinite(float(next_state.reward))
    assert np.isfinite(float(next_state.info["waypoint_distance"]))
    np.testing.assert_allclose(
        next_state.obs["state"][6:9], next_state.info["command"], atol=1.0e-7
    )


def test_apollo_warmstart_import_matches_frozen_onnx_runtime():
    profile = get_ppo_profile("apollo_walk_warmstart_v1")
    networks = profile.network_factory()(
        {"state": (78,), "privileged_state": (84,)},
        23,
        preprocess_observations_fn=brax_types.identity_observation_preprocessor,
    )
    params = networks.policy_network.init(jax.random.PRNGKey(20_261_403))
    params = load_apollo_onnx_teacher_params(params, WALK_POLICY)
    normalizer = running_statistics.init_state(
        {
            "state": jax.ShapeDtypeStruct((78,), jp.float32),
            "privileged_state": jax.ShapeDtypeStruct((84,), jp.float32),
        }
    )
    observations = np.random.default_rng(20_261_404).normal(
        0.0, 0.25, size=(32, 78)
    ).astype(np.float32)
    actual = np.asarray(
        networks.policy_network.apply(
            normalizer,
            params,
            {
                "state": jp.asarray(observations),
                "privileged_state": jp.zeros((32, 84), dtype=jp.float32),
            },
        )[0]
    )

    session = ort.InferenceSession(
        str(WALK_POLICY), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    expected = np.concatenate(
        [
            session.run(None, {input_name: observation[None, :]})[0]
            for observation in observations
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        actual, np.clip(expected, -5.0, 5.0), atol=2.0e-5, rtol=1.0e-5
    )
