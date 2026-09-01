"""Versioned PPO network and optimizer profiles for reproducible checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Any, Callable

import jax
from brax.training.agents.ppo import networks as ppo_networks

from .legacy_policy import make_legacy_ppo_networks


@dataclass(frozen=True)
class PpoProfile:
    name: str
    policy_hidden_layer_sizes: tuple[int, ...]
    value_hidden_layer_sizes: tuple[int, ...]
    distribution_type: str
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    discounting: float
    entropy_cost: float
    learning_rate: float = 3.0e-4
    normalize_observations: bool = True
    adaptive_kl: bool = False
    factory_kind: str = "standard"
    policy_contract: str = "run_policy_v1"
    desired_kl: float = 0.01
    learning_rate_min: float = 1.0e-5
    learning_rate_max: float = 1.0e-2
    init_noise_std: float = 1.0
    zero_mean_init: bool = False

    def network_factory(self) -> Callable[..., Any]:
        if self.factory_kind == "legacy_teacher":
            return make_legacy_ppo_networks
        options: dict[str, Any] = {
            "policy_hidden_layer_sizes": self.policy_hidden_layer_sizes,
            "value_hidden_layer_sizes": self.value_hidden_layer_sizes,
            "policy_obs_key": "state",
            "value_obs_key": "privileged_state",
            "distribution_type": self.distribution_type,
            "init_noise_std": self.init_noise_std,
        }
        if self.zero_mean_init:
            options.update(
                {
                    "mean_kernel_init_fn": jax.nn.initializers.constant,
                    "mean_kernel_init_kwargs": {"value": 0.0},
                }
            )
        return functools.partial(
            ppo_networks.make_ppo_networks,
            **options,
        )


PROFILES = {
    "soccer_motion_smoke_v1": PpoProfile(
        name="soccer_motion_smoke_v1",
        policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(128, 128),
        distribution_type="normal",
        unroll_length=8,
        batch_size=32,
        num_minibatches=4,
        num_updates_per_batch=1,
        discounting=0.99,
        entropy_cost=1.0e-3,
        learning_rate=5.0e-5,
        normalize_observations=False,
        policy_contract="soccer_motion_policy_v1",
        init_noise_std=0.2,
        zero_mean_init=True,
    ),
    # Finite PAiD-derived kicks start at the exact zero-residual controller.
    # A low-noise normal head and one PPO pass protect that measured baseline
    # while failure-focused phase resets teach the missing support correction.
    "soccer_motion_residual_v1": PpoProfile(
        name="soccer_motion_residual_v1",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=1,
        discounting=0.99,
        entropy_cost=1.0e-3,
        learning_rate=5.0e-5,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="soccer_motion_policy_v1",
        desired_kl=0.005,
        learning_rate_min=5.0e-6,
        learning_rate_max=1.0e-4,
        init_noise_std=0.2,
        zero_mean_init=True,
    ),
    # v1 remained numerically indistinguishable from zero residual on a fixed
    # 128-state evaluation. v2 increases exploration and performs five passes
    # per batch so dense root/contact terms can escape that local optimum.
    "soccer_motion_residual_v2": PpoProfile(
        name="soccer_motion_residual_v2",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=5,
        discounting=0.99,
        entropy_cost=2.0e-3,
        learning_rate=1.0e-4,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="soccer_motion_policy_v1",
        desired_kl=0.01,
        learning_rate_min=1.0e-5,
        learning_rate_max=3.0e-4,
        init_noise_std=0.5,
        zero_mean_init=True,
    ),
    "soccer_motion_residual_v3": PpoProfile(
        name="soccer_motion_residual_v3",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=5,
        discounting=0.99,
        entropy_cost=2.0e-3,
        learning_rate=1.0e-4,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="soccer_motion_policy_v2",
        desired_kl=0.01,
        learning_rate_min=1.0e-5,
        learning_rate_max=3.0e-4,
        init_noise_std=0.5,
        zero_mean_init=True,
    ),
    # The first PPO resume from the state-feedback clone gained a few phase
    # completions but regressed survival under v3's five-pass, high-entropy
    # update. v4 keeps the checkpoint-compatible actor/critic boundary while
    # using one pass, one tenth the learning rate and a tighter KL region.
    "soccer_motion_residual_v4": PpoProfile(
        name="soccer_motion_residual_v4",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=1,
        discounting=0.99,
        entropy_cost=1.0e-4,
        learning_rate=1.0e-5,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="soccer_motion_policy_v2",
        desired_kl=0.002,
        learning_rate_min=2.5e-6,
        learning_rate_max=2.0e-5,
        init_noise_std=0.2,
        zero_mean_init=True,
    ),
    # K2 appends ball/target commands to the retained K1-D actor.  The transfer
    # tool copies this architecture and zero-initializes only the 16 new first-
    # layer rows, preserving the inherited motion controller at bootstrap.
    "soccer_ball_motion_residual_v1": PpoProfile(
        name="soccer_ball_motion_residual_v1",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=1,
        discounting=0.99,
        entropy_cost=1.0e-4,
        learning_rate=1.0e-5,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="soccer_ball_motion_policy_v1",
        desired_kl=0.002,
        learning_rate_min=2.5e-6,
        learning_rate_max=2.0e-5,
        init_noise_std=0.2,
        zero_mean_init=True,
    ),
    # Kept solely so the first integration checkpoints remain reproducible.
    "smoke_20260830": PpoProfile(
        name="smoke_20260830",
        policy_hidden_layer_sizes=(256, 128, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        distribution_type="tanh_normal",
        unroll_length=16,
        batch_size=64,
        num_minibatches=4,
        num_updates_per_batch=2,
        discounting=0.995,
        entropy_cost=5.0e-3,
    ),
    # Matches the legacy runtime's unbounded-before-clip action convention and
    # the current Playground T1 hidden-layer sizes/training scale.
    "t1_legacy_normal_v1": PpoProfile(
        name="t1_legacy_normal_v1",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=20,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        entropy_cost=5.0e-3,
    ),
    # Formal bounded-action profile.  MuJoCo Playground's current T1 PPO path
    # uses the tanh-normal default; keeping the decoder at 0.5 rad preserves
    # the competition runtime while preventing unbounded exploratory targets.
    "t1_tanh_v1": PpoProfile(
        name="t1_tanh_v1",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="tanh_normal",
        unroll_length=20,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        entropy_cost=5.0e-3,
    ),
    # Current T1 soccer work converges from a low-noise motion/locomotion prior,
    # not random actions.  This profile imports the verified competition actor,
    # uses the ICRA 2026 T1 walking learning rate and initial exploration scale,
    # and adapts it with PPO while keeping a separately initialized critic.
    "legacy_warmstart_v1": PpoProfile(
        name="legacy_warmstart_v1",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.995,
        entropy_cost=1.0e-3,
        learning_rate=1.0e-5,
        normalize_observations=False,
        adaptive_kl=True,
        factory_kind="legacy_teacher",
    ),
    # Phase-aware extension of the exact legacy teacher.  The two new first
    # layer rows are initialized to zero, so bootstrap actions remain exactly
    # equal to the 78-value ONNX teacher before optimization.
    "legacy_phase_warmstart_v2": PpoProfile(
        name="legacy_phase_warmstart_v2",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.995,
        entropy_cost=1.0e-3,
        learning_rate=5.0e-6,
        normalize_observations=False,
        adaptive_kl=True,
        factory_kind="legacy_teacher",
        policy_contract="run_policy_v2",
    ),
    # Conservative reference-motion transfer profile.  Brax defaults the
    # adaptive-KL lower bound to 1e-5, which silently doubles the v2 profile's
    # nominal learning rate.  This version makes the bounds explicit, reduces
    # each rollout to one PPO pass and tightens the KL trust region so the
    # stable phase policy is not forgotten while learning the motion prior.
    "legacy_motion_track_v3": PpoProfile(
        name="legacy_motion_track_v3",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=1,
        discounting=0.995,
        entropy_cost=1.0e-4,
        learning_rate=1.0e-6,
        normalize_observations=False,
        adaptive_kl=True,
        factory_kind="legacy_teacher",
        policy_contract="run_policy_v2",
        desired_kl=0.002,
        learning_rate_min=2.5e-7,
        learning_rate_max=2.0e-6,
    ),
    # Retained so the first rejected random-head smoke run remains exactly
    # reproducible from its manifest and source revision.
    "reference_residual_v1": PpoProfile(
        name="reference_residual_v1",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="tanh_normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=2,
        discounting=0.995,
        entropy_cost=1.0e-4,
        learning_rate=1.0e-4,
        normalize_observations=True,
        policy_contract="run_policy_v3",
    ),
    # The v3 actor predicts only a bounded correction around the periodic T1
    # reference.  A zero-mean normal network therefore starts exactly on the
    # demonstrated motion instead of inheriting the legacy nominal-pose
    # decoder.  The residual authority is limited separately by the v3
    # contract to 0.15 rad.
    "reference_residual_v2": PpoProfile(
        name="reference_residual_v2",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=1,
        discounting=0.995,
        entropy_cost=1.0e-4,
        learning_rate=1.0e-5,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="run_policy_v3",
        desired_kl=0.002,
        learning_rate_min=2.5e-6,
        learning_rate_max=2.0e-5,
        init_noise_std=0.1,
        zero_mean_init=True,
    ),
    # The v2 run proved that a 0.1 standard deviation and one PPO pass keep KL
    # controlled, but they did not explore corrections as large as the exact
    # MuJoCo inverse-dynamics diagnostic requires.  This ablation retains the
    # exact zero deterministic mean while increasing exploration to 0.5,
    # matching the official BeyondMimic five-epoch update pattern and using a
    # wider adaptive trust region.  It is an experiment profile, not a release
    # default; v2 remains immutable for reproduction.
    "reference_residual_v3": PpoProfile(
        name="reference_residual_v3",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=5,
        discounting=0.99,
        entropy_cost=5.0e-3,
        learning_rate=1.0e-4,
        normalize_observations=True,
        adaptive_kl=True,
        policy_contract="run_policy_v3",
        desired_kl=0.01,
        learning_rate_min=1.0e-5,
        learning_rate_max=3.0e-4,
        init_noise_std=0.5,
        zero_mean_init=True,
    ),
    # GMR reference generation changes the pinned artifact, not the actor
    # interface or optimizer hypothesis.  Keeping a separate profile prevents
    # a v3 checkpoint from being resumed against the incompatible v4 decoder.
    "reference_residual_v4": PpoProfile(
        name="reference_residual_v4",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=5,
        discounting=0.99,
        entropy_cost=5.0e-3,
        learning_rate=1.0e-4,
        normalize_observations=True,
        adaptive_kl=True,
        policy_contract="run_policy_v4",
        desired_kl=0.01,
        learning_rate_min=1.0e-5,
        learning_rate_max=3.0e-4,
        init_noise_std=0.5,
        zero_mean_init=True,
    ),
    # The fixed-command v4 run exposed two deployment hazards: its running
    # normalizer assigned near-zero variance to command channels, and the
    # five-pass/high-entropy update moved away from its best checkpoint late
    # in training.  v5 keeps the explicit observation scales used by the
    # runtime, starts with less exploratory noise, and uses a tighter trust
    # region so checkpoints can transfer through a low-to-high-speed course.
    "reference_curriculum_v5": PpoProfile(
        name="reference_curriculum_v5",
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="normal",
        unroll_length=24,
        batch_size=256,
        num_minibatches=32,
        num_updates_per_batch=3,
        discounting=0.99,
        entropy_cost=1.0e-3,
        learning_rate=5.0e-5,
        normalize_observations=False,
        adaptive_kl=True,
        policy_contract="run_policy_v4",
        desired_kl=0.005,
        learning_rate_min=5.0e-6,
        learning_rate_max=1.0e-4,
        init_noise_std=0.3,
        zero_mean_init=True,
    ),
}


def get_ppo_profile(name: str) -> PpoProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown PPO profile {name!r}") from exc
