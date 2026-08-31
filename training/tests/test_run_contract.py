from pathlib import Path

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


def test_run_policy_contract_preserves_runtime_boundary():
    contract = load_policy_contract(CONTRACT)

    assert contract.policy_name == "run_policy_v1"
    assert contract.frequency_hz == 50
    assert contract.observation_size == 78
    assert contract.action_size == 23
    assert contract.input_shape == (1, 78)
    assert contract.output_shape == (1, 23)
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


def test_phase_policy_contract_extends_actor_without_changing_actions():
    contract = load_policy_contract(PHASE_CONTRACT)
    env = DirectionalRun(contract=contract)

    assert contract.policy_name == "run_policy_v2"
    assert contract.observation_size == 80
    assert contract.input_shape == (1, 80)
    assert contract.output_shape == (1, 23)
    assert contract.observation_fields[-1] == ("gait_phase_cos_sin", 2)
    assert env.observation_size == {"state": (80,), "privileged_state": (86,)}


def test_formal_profile_uses_bounded_action_and_official_t1_widths():
    profile = get_ppo_profile("t1_tanh_v1")

    assert profile.distribution_type == "tanh_normal"
    assert profile.policy_hidden_layer_sizes == (512, 256, 128)
    assert profile.unroll_length == 20
    assert profile.num_updates_per_batch == 4
