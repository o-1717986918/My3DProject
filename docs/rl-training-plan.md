# Reinforcement-learning motion plan

Status: implementation started 2026-08-30
Target platform: WSL2 Ubuntu 22.04, Booster T1, RCSSServerMJ 0.2.1
Primary task: a stronger upright kick that fits the existing competition FSM

## 1. Decision

Train low-level skills separately and keep the deterministic team controller.
Do not train an end-to-end soccer policy. The first new policy is a ball-aware
kick; robust walking follows only after the kick has crossed its acceptance
gate. Apollo get-up remains the current recovery backend and is not used as a
training initialization.

Priority:

1. ball-aware kick;
2. robust turning and perturbation-resistant walking;
3. dribble/push control;
4. independently trained get-up only if the project needs to remove its GPL
   runtime dependency.

## 2. Why the kick comes first

The current learned-walk burst completes `ALIGN -> KICK -> RECOVER` reliably,
but produces only small ball displacement. The previous hand-authored
high-energy keyframe moved the ball and then fell. A contact-rich kick policy is
therefore the clearest case where reinforcement learning can improve the main
competition metric without replacing the stable decision and recovery layers.

## 3. Host constraints

Measured host resources:

- NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB VRAM;
- AMD Ryzen AI 7 H 350, 8 cores / 16 threads;
- 15 GiB WSL memory plus 4 GiB swap;
- approximately 893 GiB free in the WSL virtual disk;
- approximately 68 GiB free on the Windows C drive.

Training runs and checkpoints must live under `/home/win98/rl_runs`, not
`/mnt/c`. Start with 512 parallel environments and increase only after recording
peak device memory and simulation throughput.

## 4. Environment isolation

The competition environment `my3d-team` stays on Python 3.13 and must not gain
training dependencies. Training uses a separate Python 3.12 environment named
`my3d-rl`.

The initial stack is pinned MuJoCo Playground with JAX PPO and CUDA 12. This
preserves a MuJoCo-to-MuJoCo transfer path and already provides a Booster T1
locomotion baseline. MJX-JAX and MJX-Warp are both installed; the main backend
is selected from local 256/512-environment speed, memory, contact, and NaN
benchmarks rather than from generic framework claims.

Repository layout:

```text
training/
├── README.md
├── contracts/          # observation/action/model manifests
├── envs/               # MJX task definitions
├── evaluation/         # deterministic policy and physics checks
├── export/             # checkpoint-to-ONNX tools
└── tests/              # training-interface tests
```

Large generated files go to `/home/win98/rl_runs` and are never committed.

## 5. Physics-parity gate

No policy training starts until a direct MuJoCo scene and RCSSServerMJ agree on
the action contract:

- exact 23-joint order and radian/degree conversions;
- joint limits, actuator force limits, PD gains, and action scaling;
- 50 Hz policy control with the server-equivalent physics substeps;
- T1 root height and neutral-pose equilibrium;
- ball radius, mass, friction, restitution, and ground contact;
- quaternion convention and projected-gravity calculation.

The parity test replays a fixed open-loop joint-target sequence in both systems
and compares joint trajectories, torso pose, contacts, and ball displacement.
Differences are documented rather than tuned away silently.

## 6. Kick task contract

### Actor observations

Only deployable signals are allowed:

- body angular velocity and projected gravity;
- 23 joint positions and 23 joint velocities;
- previous 23-value action;
- ball position and velocity in the torso/yaw-local frame;
- target shot direction in the same frame;
- observation age/mask for ball freshness;
- optional phase sine/cosine.

The critic may receive simulator-only contact and exact velocity state during
training. These privileged values must never enter the exported actor.

### Actions

The actor outputs 23 bounded residual joint targets. Targets are applied through
the same position/PD protocol as the competition client. The initial policy is
residual around a stable nominal pose or stable locomotion target, not an
unbounded torque policy.

### Episode initialization

- upright T1 with randomized joint and base perturbations;
- stationary ball in a reachable region near either foot;
- randomized target heading;
- randomized friction, ball properties, PD gains, action latency, sensor noise,
  and light external pushes;
- a curriculum that widens these ranges only after the preceding stage passes.

### Reward families

Positive terms:

- ball velocity and displacement along the requested direction;
- useful contact followed by separation;
- upright torso and successful post-kick stabilization.

Penalties:

- falling, unstable angular velocity, and low torso height;
- support-foot slip and self-collision;
- joint-limit proximity, actuator effort, and action-rate spikes;
- lateral/backward ball motion and repeated weak contacts.

Reward components must be logged separately. A high total reward is not an
acceptance result by itself.

## 7. Curriculum and teacher/student stages

1. fixed ball, fixed target, no randomization;
2. small ball-position and base-pose randomization;
3. variable target direction and both-foot opportunities;
4. physics, latency, sensor, and external-push randomization;
5. moving ball and transition from a slow approach;
6. optional nearby static/dynamic obstacle, only after the isolated kick works.

If the deployable actor is limited by delayed or missing ball observations,
continue with a corrected version of the 2026 T1 striker pipeline:

1. train a privileged chase/kick teacher;
2. collect an actual aggregate dataset with a gradual teacher-to-student mix;
3. distil into a student using measured observation latency, field of view,
   dropouts, and noise;
4. only then evaluate constrained RL with non-negative safety costs.

The public reference implementation is not run unchanged: its released
configuration has a 21-versus-23 privileged-input mismatch, its DAgger beta
collapses after one iteration, and its stated goal occlusion does not match the
code. The method is valuable; the release is not a ready-to-run baseline.

## 8. Acceptance gates

### Simulator evaluation

- at least 200 held-out randomized episodes per seed;
- at least three independent training seeds;
- upright success rate >= 95%;
- median shot-direction error <= 15 degrees;
- forward ball displacement >= 0.8 m within two seconds for the first MVP;
- no NaN, joint-limit violation, or invalid motor packet;
- exported ONNX output matches its source checkpoint within a documented
  numerical tolerance.

### RCSSServerMJ evaluation

1. deterministic single-player kick matrix across ball offsets and headings;
2. no fall in the post-kick stabilization window;
3. repeatable ball displacement and direction from the server state/log;
4. four-direction get-up regression remains green;
5. `scripts/run_acceptance_match.sh 600` still observes
   `ALIGN -> KICK -> RECOVER`, 14 connections, and clean shutdown.

The old policy remains selectable until the new policy passes every gate.

## 9. Export and runtime integration

Each model release includes:

- ONNX model and SHA-256 checksum;
- observation names, order, units, normalization, and clipping;
- action order, scale, PD gains, and limits;
- training code revision, environment revision, seeds, and curriculum stage;
- training and held-out metrics;
- licence and source provenance;
- RCSSServerMJ validation record.

The runtime adapter validates tensor shapes and finite output. Any loading or
inference failure falls back to the existing stable kick.

## 10. Stop conditions

Stop or revise the experiment when:

- training exploits simulator artifacts rather than kicking;
- performance collapses after modest physics randomization;
- the actor requires state unavailable to the competition client;
- ONNX deployment materially changes actions;
- stronger displacement reduces upright success below 95%;
- direct MuJoCo and RCSSServerMJ parity cannot be bounded.

## 11. Implementation sequence

- [x] record hardware and current competition baseline;
- [x] define task scope, policy contract, and acceptance gates;
- [x] create and lock `my3d-rl`;
- [x] verify the JAX GPU backend;
- [x] record JAX/Warp throughput and initial allocator telemetry;
- [x] hash the exact permitted T1 and soccer physics inputs;
- [x] implement the direct-MuJoCo scene and action-contract tests;
- [x] implement the fixed-ball kick environment;
- [x] run a short PPO smoke training and persist a checkpoint;
- [ ] export ONNX and integrate behind a feature flag;
- [ ] execute the full simulator and match acceptance matrix.
