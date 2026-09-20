"""FastWalkV2 continuation from sparse server-observed Apollo Walk states."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

from .contract import PolicyContract
from .run_env import DirectionalRun


class ServerFastWalkRun(DirectionalRun):
    """Reset the unchanged 80→23 actor at real Walk→FastWalk boundaries.

    The corpus is a projection of noisy server observations, not a full
    multi-robot RCSS snapshot. The runtime starts FastWalk with zero previous
    action and zero phase; keeping that switch exact is the purpose here.
    """

    def __init__(
        self,
        corpus: Path,
        *,
        split: int,
        near_ball_probability: float | None = None,
        forward_entry_probability: float | None = None,
        config_overrides: dict[str, Any] | None = None,
        contract: PolicyContract,
    ) -> None:
        if contract.policy_name != "run_policy_v2":
            raise ValueError("server FastWalk reset requires run_policy_v2")
        if split not in (0, 1):
            raise ValueError("split must be train=0 or validation=1")
        if near_ball_probability is not None and not 0.0 <= near_ball_probability <= 1.0:
            raise ValueError("near-ball probability must be in [0, 1]")
        if forward_entry_probability is not None and not 0.0 <= forward_entry_probability <= 1.0:
            raise ValueError("forward-entry probability must be in [0, 1]")
        super().__init__(config_overrides=config_overrides, contract=contract)
        if (
            not self._config.use_fixed_command
            or list(self._config.fixed_command) != [1.5, 0.0, 0.0]
            or list(self._config.gait_frequency) != [1.75, 1.75]
            or self._config.action_clip != 10.0
        ):
            raise ValueError("server FastWalk reset must mirror runtime command/phase/action clip")
        with np.load(corpus, allow_pickle=False) as archive:
            required = {"qpos", "qvel", "split", "near_ball", "source_row"}
            if required - set(archive.files):
                raise ValueError("server reset corpus is missing required arrays")
            if forward_entry_probability and "forward_entry_proxy" not in archive.files:
                raise ValueError("forward-entry sampling needs a labeled corpus")
            all_split = np.asarray(archive["split"], dtype=np.uint8)
            selected = all_split == split
            qpos = np.asarray(archive["qpos"][selected], dtype=np.float32)
            qvel = np.asarray(archive["qvel"][selected], dtype=np.float32)
            near_ball = np.asarray(archive["near_ball"][selected], dtype=np.uint8)
            forward_entry = (
                np.asarray(archive["forward_entry_proxy"][selected], dtype=np.uint8)
                if "forward_entry_proxy" in archive.files
                else np.zeros(len(qpos), dtype=np.uint8)
            )
            source_row = np.asarray(archive["source_row"][selected], dtype=np.int32)
        if (
            qpos.ndim != 2 or len(qpos) == 0
            or qpos.shape[1] != self._mj_model.nq
            or qvel.shape != (len(qpos), self._mj_model.nv)
            or near_ball.shape != (len(qpos),)
            or forward_entry.shape != (len(qpos),)
            or source_row.shape != (len(qpos),)
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
            or set(np.unique(near_ball)) - {0, 1}
            or set(np.unique(forward_entry)) - {0, 1}
        ):
            raise ValueError("server reset corpus has incompatible state arrays")
        near_indices = np.flatnonzero(near_ball == 1)
        far_indices = np.flatnonzero(near_ball == 0)
        forward_indices = np.flatnonzero(
            (near_ball == 0) & (forward_entry == 1)
        )
        other_indices = np.flatnonzero(
            (near_ball == 0) & (forward_entry == 0)
        )
        if len(far_indices) == 0:
            raise ValueError("server reset corpus needs non-near-ball states")
        self._entry_qpos = jp.asarray(qpos)
        self._entry_qvel = jp.asarray(qvel)
        self._entry_source_row = jp.asarray(source_row)
        self._near_indices = jp.asarray(near_indices, dtype=jp.int32)
        self._far_indices = jp.asarray(far_indices, dtype=jp.int32)
        self._forward_indices = jp.asarray(forward_indices, dtype=jp.int32)
        self._other_indices = jp.asarray(other_indices, dtype=jp.int32)
        self._near_count = len(near_indices)
        self._far_count = len(far_indices)
        self._forward_count = len(forward_indices)
        self._other_count = len(other_indices)
        self._near_probability = (
            float(np.mean(near_ball))
            if near_ball_probability is None else near_ball_probability
        )
        if self._near_count == 0:
            self._near_probability = 0.0
        self._forward_probability = (
            float(len(forward_indices) / len(far_indices))
            if forward_entry_probability is None
            else forward_entry_probability
        )
        if self._forward_count == 0:
            self._forward_probability = 0.0
        if self._other_count == 0:
            self._forward_probability = 1.0

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, near_rng, far_rng, bucket_rng, forward_rng, other_rng, forward_bucket_rng = (
            jax.random.split(rng, 7)
        )
        far_index = self._far_indices[
            jax.random.randint(far_rng, (), 0, self._far_count)
        ]
        if self._forward_count and self._other_count:
            forward_index = self._forward_indices[
                jax.random.randint(forward_rng, (), 0, self._forward_count)
            ]
            other_index = self._other_indices[
                jax.random.randint(other_rng, (), 0, self._other_count)
            ]
            far_index = jp.where(
                jax.random.bernoulli(
                    forward_bucket_rng, self._forward_probability
                ),
                forward_index, other_index,
            )
        if self._near_count:
            near_index = self._near_indices[
                jax.random.randint(near_rng, (), 0, self._near_count)
            ]
            choose_near = jax.random.bernoulli(
                bucket_rng, self._near_probability
            )
            index = jp.where(choose_near, near_index, far_index)
        else:
            index = far_index
        qpos = self._entry_qpos[index]
        qvel = self._entry_qvel[index]
        command = jp.asarray(self._config.fixed_command, dtype=jp.float32)
        previous_action = jp.zeros(self.action_size, dtype=jp.float32)
        ctrl = jp.zeros(self._mj_model.nu)
        ctrl = ctrl.at[self._pos_actuator].set(qpos[self._joint_qpos])
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)
        yaw = jp.arctan2(
            data.site_xmat[self._torso_site][1, 0],
            data.site_xmat[self._torso_site][0, 0],
        )
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "command": command,
            "gait_phase": jp.array(0.0),
            "gait_frequency": jp.array(1.75),
            "last_action": previous_action,
            "last_last_action": previous_action,
            "delay_steps": jp.array(0, dtype=jp.int32),
            "reference_init": jp.array(False),
            "last_foot_positions": jp.stack([
                data.site_xpos[self._left_foot_site],
                data.site_xpos[self._right_foot_site],
            ]),
            "initial_torso_xy": data.site_xpos[self._torso_site, :2],
            "initial_yaw": yaw,
            "entry_source_row": self._entry_source_row[index],
        }
        return mjx_env.State(
            data=data,
            obs=self._get_obs(data, info),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=self._initial_metrics(),
            info=info,
        )
