# RCSSServerMJ running-policy development plan

Status: v1/v2 plus motion-prior pipeline complete; open-strategy refresh has
selected periodic reference-centred residual tracking; no running policy has
passed release gates as of 2026-08-31

Owner environment: WSL2 Ubuntu 22.04, Conda `my3d-rl`

Target: Booster T1 in RCSSServerMJ 0.2.1, 50 Hz policy control

## 1. Outcome and scope

This phase produces a versioned, recoverable reinforcement-learning pipeline
for fast T1 locomotion and a checkpoint that is evaluated against an explicit
running gate. It does not replace the deterministic team strategy, ball
approach, kick, or get-up state machines.

The existing `walk.onnx` remains the competition fallback until the new model
passes all simulator, export, server, and 7v7 gates. A short PPO smoke run proves
only that the optimizer works; it does not prove that the robot can run.

## 2. Evidence-backed decision

The 2026-08-31 open search supersedes reward-only continuation of the baseline
described below. The selected next controller is
`q_target = q_reference(phase, speed) + residual_scale * action_residual`, after
an identical-action CPU/MJWarp parity test and periodic T1 reference
projection. Whole-body tracking, adaptive failure-phase sampling and
train-time symmetry follow in that order. The evidence matrix, licensing
boundaries and exact stages are in
[`open-strategy-search-2026-08-31.md`](open-strategy-search-2026-08-31.md) and
`training/locks/open_strategy_2026_08_31.yaml`.

The fixed-nominal policy remains a measured historical baseline and the stable
walk remains the runtime fallback; neither is treated as a running teacher.

Use the already pinned MuJoCo Playground/Brax PPO stack for the first formal
baseline, with MJX-Warp as the measured fast backend and MJX-JAX as a diagnostic
fallback. Train on a custom environment assembled from the installed
RCSSServerMJ soccer world and T1 XML, rather than deploying Playground's
upstream `T1JoystickFlatTerrain` policy directly.

Reasons:

- the competition client consumes a 78-value observation and emits 23 residual
  joint-position actions at 50 Hz;
- upstream Playground T1 currently uses a different observation layout,
  0.002 s simulation step, and command range whose forward maximum is 1.0 m/s;
- RCSSServerMJ uses a 0.005 s simulation step and four physics substeps per
  policy action;
- the project's current walk decoder has a distinct nominal pose, joint sign
  conversion, normalization, action scale, and PD gains;
- local measurement already selected Warp by throughput, but CPU MuJoCo and
  the actual server remain the deployment truth.

PPO is the first baseline because the pinned local toolchain already completes
real optimizer updates and checkpoint restore. FastSAC/FastTD3 is a deliberate
second algorithm experiment, not an unreviewed mid-run dependency change. The
2025 rapid-locomotion paper reports strong wall-clock gains on an RTX 4090 and
T1, but this 8 GB RTX 5060 Laptop host and exact RCSS scene need their own
measurement.

## 3. Versioned policy contract

`run_policy_v1` preserves the legacy Python runtime boundary, while
`run_policy_v2` adds gait phase. Both current contracts centre actions on a
fixed nominal pose; the next motion-tracking contract will version the moving
reference and residual scale explicitly rather than changing v2 silently:

- input: `float32[1, 78]`;
- output: `float32[1, 23]`;
- control rate: 50 Hz;
- action: normal-policy output clipped to `[-10, 10]`, matching the existing
  runtime, then decoded as a residual around the nominal pose;
- physical target: `(nominal + 0.5 * action) * train_sim_flip`, followed by
  the exact T1 joint-limit clamp already used by the competition runtime;
- PD gains: `kp=25`, `kd=0.6`;
- finite-value checking and fallback to the existing stable walk model.

The observation is one current frame, in this exact order:

1. 23 interleaved triplets of normalized joint-position offset, normalized
   joint velocity, and normalized previous action: 69 values;
2. body angular velocity: 3 values;
3. local target velocity `(vx, vy, yaw_rate)`: 3 values;
4. projected gravity in the body frame: 3 values.

Training-only root velocity, height, contacts, and exact actuator state may be
used by the critic or metrics, but never by the exported actor.

## 4. Task, reward, and curriculum

The actor tracks a body-local velocity command. The initial formal curriculum
is deliberately staged:

1. balance at a zero command until 10-second upright completion passes;
2. stand and slow forward motion, `vx=0.0..0.8 m/s`;
3. fast walk transition, `vx=0.4..1.2 m/s`;
4. running target, `vx=0.8..1.8 m/s`;
5. add `vy=-0.4..0.4 m/s` and `yaw_rate=-0.6..0.6 rad/s`;
6. add reset perturbations, pushes, observation noise, action delay, and
   bounded PD/friction randomization;
7. retain the strongest stable checkpoint and run server-side acceptance.

Commands are held long enough to reveal falls and resampled between episodes.
At least 20% of held-out episodes use a zero command to verify stable standing.

Positive reward families:

- exponential local linear-velocity tracking;
- exponential yaw-rate tracking;
- upright torso, target torso height, and episode survival;
- forward progress for non-zero forward commands.

Penalty families:

- vertical velocity and roll/pitch angular velocity;
- action rate and second difference;
- joint speed, joint-limit proximity, and control effort;
- foot slip and non-foot contact when robust contact metrics are available;
- termination on low height, inverted torso, invalid state, or timeout.

Every reward/cost term is logged independently. Total reward is never used as
the sole release criterion.

## 5. Formal training protocol

All generated data lives below `/home/win98/rl_runs/run-*`; no checkpoint,
TensorBoard event, rollout, or video is committed to Git.

Baseline settings:

- backend: Warp, with JAX smoke parity;
- parallel environments: 64 initially on the 8 GB GPU;
- policy MLP: `512, 256, 128`;
- value MLP: `512, 256, 128` with privileged state;
- PPO learning rate: `3e-4`;
- normal action distribution, with runtime-compatible `[-10, 10]` clipping;
- unroll length: 20;
- discount: 0.97;
- GAE lambda: 0.95;
- action clipping and gradient clipping enabled;
- fixed seed recorded in the run manifest;
- resumable checkpoints and machine-readable progress JSONL.

The first baseline may tune only documented hyperparameters. A formal model
candidate requires a complete run manifest, environment/config snapshot,
source commit, dependency lock, random seed, elapsed time, checkpoint hash, and
deterministic evaluation report.

Compatibility note: Brax 0.14.2's generic `load_policy` helper cannot rebuild
the saved network when its JSON contains a null default kernel-initializer
name. Project evaluation/export code therefore reconstructs the pinned network
factory explicitly, applies Brax running-statistics normalization, and then
loads the numeric parameter tree. This path is covered by checkpoint evaluation
rather than relying on the broken convenience helper.

## 6. Acceptance definition

### Gate R0: interface and physics

- contract validation, joint order, signs, scales, and shapes pass;
- reset and one step are finite on CPU MuJoCo, MJX-JAX, and MJX-Warp;
- clamped neutral targets remain within the exact T1 joint limits;
- 0.02 s control equals four 0.005 s RCSS physics steps.

### Gate R1: optimizer integration

- a real PPO update completes on the GPU;
- a checkpoint restores with all numeric leaves finite;
- no contact-buffer overflow, driver reset, out-of-memory exit, or silent NaN;
- the run manifest is written even when a later acceptance gate fails.

### Gate R2: simulated running candidate

Evaluate deterministic policies over at least 200 held-out episodes per seed.
The eventual release requires three independent seeds; a single-seed baseline
is labelled `candidate`, never `release`.

- commanded `vx=1.5 m/s` for 10 s;
- at least 95% upright completion;
- median achieved forward speed at least 1.2 m/s after the first 2 s;
- median forward tracking RMSE at most 0.35 m/s;
- median lateral drift at most 0.25 m over the 10-second rollout;
- no NaN, invalid action, joint-limit violation, or unsafe motor packet;
- at least 80% of exact-CPU episodes contain a two-foot flight interval lasting
  at least two consecutive 5 ms physics frames. A candidate missing this gate
  is `fast locomotion`, not running.

The command suite also includes `vx=0.0, 0.5, 1.0, 1.5`, lateral commands, yaw
commands, and abrupt command changes. A policy that only sprints straight is
not competition-ready.

### Gate R3: deployment parity

- exported ONNX input/output names and shapes match the manifest;
- source and ONNX actions agree within `atol=1e-5`, `rtol=1e-4` over recorded
  held-out observations;
- 30-minute repeated inference produces only finite outputs;
- model selection is behind a feature flag and automatically falls back on
  load, shape, or inference failure.

### Gate R4: RCSSServerMJ and 7v7

- deterministic single-player server runs at all command-suite points;
- measured server speed and fall rate remain within the R2 thresholds or the
  candidate is rejected;
- get-up, kick, and transition regressions remain green;
- a 600-cycle acceptance match has 14 connections, `PLAY_ON`, no client
  failure, clean shutdown, and observed movement from both teams;
- a bounded real-time 7v7 visual run is recorded after headless acceptance.

## 7. Search and source provenance

Search performed on 2026-08-30 and refreshed on 2026-08-31 against current
primary sources and local pinned source trees. Design claims are accepted only
from original papers, official framework/vendor repositories, official
RCSSServerMJ documentation, or direct local measurements. The complete refresh
is in `docs/open-strategy-search-2026-08-31.md`; exact commits and reuse rules
are in `training/locks/sources.yaml`.

- MuJoCo Playground official repository, pinned local tag `v0.2.0`, commit
  `124a73fa3303f75a62f8fe04d329b829ed0ebdfb`;
- Booster Gym official repository, searched current `main` commit
  `da396a06d6eed99e2de72d7749c48ee8748950f9`;
- Holosoma/FastSAC official repository, searched current `main` commit
  `fb835ec8cb6ee48f483ce567586625e5fae1ae1f`;
- RCSSServerMJ official GitLab/PyPI release 0.2.1 and the locally hashed assets;
- Wang et al. (2025), *Booster Gym: An End-to-End Reinforcement Learning
  Framework for Humanoid Robot Locomotion*, arXiv:2506.15132;
- Seo et al. (2025), *Learning Sim-to-Real Humanoid Locomotion in 15 Minutes*,
  arXiv:2512.01996;
- Radosavovic et al. (2024), *Real-World Humanoid Locomotion with
  Reinforcement Learning*, Science Robotics, DOI:10.1126/scirobotics.adi9579;
- ApolloCodebase, pinned submodule commit
  `71018c968969d6e55130b0e1987cd5b4f5c3b4df`, used only to compare the 78->23
  inference boundary and not copied into the permissive trainer.

The strongest counterargument is that the 78-value single-frame actor may be
insufficient for high-speed state estimation. That is tested rather than
assumed: if R2 plateaus after reward and physics audits, `run_policy_v2` may add
short observation history or a recurrent encoder. The v1 contract is not
silently changed.

## 8. Deliverables checklist

- [x] evidence search, source pins, contract decision, and acceptance gates;
- [x] `run_policy_v1.yaml` legacy contract and phase-aware `run_policy_v2.yaml`;
- [x] exact RCSS-physics locomotion environment and tests;
- [x] formal/resumable PPO entry point and manifests;
- [x] deterministic MJX/CPU evaluators and machine-readable reports;
- [x] v1/v2 ONNX export and parity test;
- [x] pinned Holosoma/LAFAN import, exact RCSS replay and motion-reference gate;
- [x] bounded-KL motion curriculum and CPU true-flight/contact evaluation;
- [x] left/right policy-reflection diagnostic and involution tests;
- [x] open reliable-strategy refresh, evidence matrix and staged decision lock;
- [x] identical-action CPU/MJWarp parity trace and regression test;
- [ ] periodic exact-T1 reference projection;
- [ ] reference-centred residual tracking contract and environment;
- [ ] feature-flagged runtime integration with fallback;
- [ ] single-player RCSS gate, headless 7v7 gate, and visual 7v7 gate;
- [ ] three-seed release evaluation after the first candidate succeeds.

## 9. Measured milestone result

Reward-only teacher adaptation did not pass R2. The best phase-aware v2
checkpoint (`run-phase-v2-formal-s71.../000003538944`) achieved 100% upright
completion, 1.499 m/s median CPU speed and 0.092 m/s tracking RMSE, but drifted
5.45 m and had 0% qualifying aerial-phase episodes. A straight-recovery run
reduced MJX drift further to 4.77 m but remained outside the 0.25 m gate.

No candidate has been copied into the runtime and the original `walk.onnx`
remains the competition default. Detailed commands, rejected runs and source
choices are recorded in `rl-experiment-log.md` and
`robot-soccer-action-research.md`.

The motion-prior stage improved normal-start MJX survival to 490/500 at a
1.8 m/s command, but exact CPU evaluation rejected the exported policy for
87.5% completion, 7.18 m median drift and only 12.5% qualifying-flight
episodes. A mathematically reflection-equivariant two-pass diagnostic reached
64/64 upright and reduced drift to 1.15 m, but erased the qualifying aerial
phase and still exceeded the 0.25 m drift gate. It is not a deployable model.

The next implementation milestone is a CPU-versus-Warp trajectory parity
suite followed by a periodic robot-native reference with endpoint, bilateral,
stance-foot and exact-contact constraints. More reward-only continuation is
not supported by the evidence from this stage.
