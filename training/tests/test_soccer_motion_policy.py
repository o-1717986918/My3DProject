from __future__ import annotations

import numpy as np
import pytest

from my3d_rl.soccer_motion_policy import load_soccer_motion_policy


def test_zero_soccer_motion_policy_has_declared_shape():
    policy = load_soccer_motion_policy(
        zero_policy=True,
        checkpoint=None,
        profile_name="soccer_motion_residual_v3",
        policy_contract_name="soccer_motion_policy_v2",
        observation_size=110,
        action_size=23,
    )

    np.testing.assert_array_equal(policy(np.ones(110)), np.zeros(23))


def test_soccer_motion_policy_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_soccer_motion_policy(
            zero_policy=True,
            checkpoint=tmp_path / "checkpoint",
            profile_name="soccer_motion_residual_v3",
            policy_contract_name="soccer_motion_policy_v2",
            observation_size=110,
            action_size=23,
        )
    with pytest.raises(ValueError, match="exactly one"):
        load_soccer_motion_policy(
            zero_policy=False,
            checkpoint=None,
            profile_name="soccer_motion_residual_v3",
            policy_contract_name="soccer_motion_policy_v2",
            observation_size=110,
            action_size=23,
        )


def test_zero_soccer_motion_policy_rejects_bad_observation():
    policy = load_soccer_motion_policy(
        zero_policy=True,
        checkpoint=None,
        profile_name="soccer_motion_residual_v3",
        policy_contract_name="soccer_motion_policy_v2",
        observation_size=110,
        action_size=23,
    )

    with pytest.raises(ValueError, match="observation"):
        policy(np.ones(109))
    with pytest.raises(ValueError, match="non-finite"):
        policy(np.full(110, np.nan))


def test_soccer_motion_policy_rejects_profile_contract_mismatch():
    with pytest.raises(ValueError, match="profile and policy contract"):
        load_soccer_motion_policy(
            zero_policy=True,
            checkpoint=None,
            profile_name="soccer_motion_residual_v3",
            policy_contract_name="soccer_motion_policy_v1",
            observation_size=110,
            action_size=23,
        )
