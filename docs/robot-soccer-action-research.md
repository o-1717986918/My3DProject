# Robot-soccer action research and adoption record

Last source refresh: 2026-09-01

Rule: refresh repository HEADs and primary papers before every training
milestone; do not turn remembered hyperparameters into project facts.

## Decision

The open-strategy refresh is recorded in
[`open-strategy-search-2026-08-31.md`](open-strategy-search-2026-08-31.md), with
an executable stage decision in
`training/locks/open_strategy_2026_08_31.yaml`.

The actionable stack for this project is:

1. exact RCSSServerMJ physics and deployment contract as the acceptance truth;
2. verified locomotion teacher warm-start instead of random-policy PPO;
3. phase-aware gait observation and conservative foot-edge contact geometry;
4. a retargeted T1 running-motion tracking prior;
5. low-noise velocity/football task fine-tuning;
6. teacher/student adaptation only after the privileged policy passes motion
   and server gates.

Steps 1–4 are implemented. The refreshed search changes the next control
architecture, not the acceptance truth: build a periodic, straight,
dynamically feasible T1 reference and learn a small residual around that moving
reference. The current action is centred on a fixed nominal pose, so the policy
must reproduce the full clip and stabilise it simultaneously. The first
imported clip contained real aerial phases but also a large turning component;
a straight sub-window removed that source bias but did not make the learned
policy airborne.

## Primary-source comparison

| Source | Current evidence | Adopt | Boundary |
|---|---|---|---|
| Chasing Autonomy / robot_rl | G1 running library at 1.2–3.6 m/s, periodic hybrid-domain trajectory optimisation, CLF-guided RL, released configs and trajectory consumers | periodic half-cycle-symmetric reference design and later CLF/library concepts | G1/Isaac gap; whole-body generator not released; repository and asset licensing are not safe for copying |
| FC Portugal / RuN | exact RoboCup 3D analytic-step residual controller; separate G1 conditional-generator-plus-residual result | reference-centred residual action is the immediate next architecture | GPL isolation for FCP; RuN has no official implementation located |
| BeyondMimic / MJLab | whole-body tracking, reference reset and adaptive failure-phase sampling; maintained MJWarp implementation | adaptive phase curriculum and root/foot/velocity rewards | G1 tasks; independently port algorithms into the exact T1 environment |
| MuJoCo Playground current T1 | official MJWarp task and ONNX; local CPU replay is stable at 0.4–1.0 m/s | regression baseline and implementation reference | no repeated flight and large lateral displacement; not a run teacher |
| Daffan et al., ICRA 2026 humanoid striker | T1 teacher PPO, explicit cosine/sine gait phase, geometric foot-edge contact, later DAgger/P3O adaptation | phase observation, 1–2 Hz gait scheduling, swing reward, teacher-first curriculum | Isaac Gym code is not the competition physics; evaluate independently in RCSS |
| Booster Gym | T1 foot-edge height contact and sim-to-sim deployment pipeline | oriented foot-box lowest-point proxy and CPU contact calibration | reference settings only; do not transplant an unverified policy |
| Holosoma / Fast humanoid locomotion | Apache-2.0, T1 support, MJWarp, velocity and whole-body tracking, LAFAN retargeting pipeline | implemented as the T1 retargeting source with a pinned local patch | upstream robot-only qpos, LAFAN order and MuJoCo 3.12 Jacobian defects required an audited patch; derived LAFAN data stays local |
| RoboNaldo | motion tracking prior followed by staged task adaptation; recommends small noise on resume | prior → task training boundary, immutable stage checkpoints, small exploration | released motion is a G1 kick, not a T1 run; reuse method, not motion |
| PAiD HumanoidSoccer | released 13-motion G1 dataset, motion/phase failure sampling, retained motion rewards during perception-action adaptation, recurrent policy and physics-aware transfer | immediate clean-room K0--K2 design reference and local attributed non-commercial retargeting candidate | CC BY-NC 4.0; G1 29-DoF/160-input checkpoint is not T1-compatible; do not vendor source/assets |
| `wbc_fsm` | G1 C++ FSM, ONNX runtime, state history, projected-gravity termination and deterministic transitions | strengthen Apollo skill lifecycle, contract validation and fallback concepts | no top-level licence found; no ball perception or team strategy; no source/model copying |
| ApolloCodebase | GPL C++ behavior-tree and strategy-network runtime with the familiar 78→23 walk boundary | isolated architecture and inference reference | GPL separation; no code copied into the permissive trainer |

Current pinned commits and reuse restrictions are machine-readable in
`training/locks/sources.yaml`.

## What the experiments established

The verified teacher reaches 1.6 m/s and survives ten seconds, but turns in a
large arc. Increasing flight reward generated a misleading training metric and
reduced variable-command stability. After replacing the proxy and adding a
phase-aware 80-value actor, the selected policy reaches 1.50 m/s with full
held-out survival and materially lower drift, but exact CPU contacts still show
no sustained flight. This falsifies the assumption that reward shaping alone
will convert the existing walking teacher into a run on the current budget.

## Implemented motion-reference contract

The motion-prior stage must produce a versioned T1 trajectory at 50 Hz with:

- root position and quaternion;
- all 23 joints in `run_policy_v2` order and radians;
- root linear/angular velocity and joint velocity, derived reproducibly if not
  supplied;
- left/right contact labels from the exact RCSS foot geoms;
- at least one ≥10 ms two-foot aerial interval per cycle;
- source URL, source commit/dataset version, license, conversion command and
  SHA-256;
- RCSS replay report: finite state, joint limits, no non-foot collision, foot
  slip and tracking error.

The pipeline now imports 30 Hz Holosoma qpos, performs quaternion-safe 50 Hz
resampling, canonicalizes forward direction, grounds on the exact RCSS T1 foot
boxes, derives velocities, replays exact contacts, validates running
morphology, and hashes provenance. The accepted straight local reference is
34 frames/0.66 s at 3.20 m/s with ten contact frames per foot, an eight-frame
aerial interval, zero non-foot pitch collision, and mean root yaw rate
0.0031 rad/s. Its SHA-256 is
`2ad294330d7d7fc19e236169bdc862079c8228fd38a544703c38f698fee09820`.

## Revised next-stage sequence

1. add a short CPU-versus-Warp trajectory parity suite for identical actions,
   contacts, torso pose and yaw before more long training;
2. create a periodic robot-native reference: enforce endpoint pose/velocity,
   bilateral half-cycle symmetry, stance-foot velocity and exact contact, not
   merely a visually plausible human segment;
3. centre the action on the phase/speed reference so zero residual replays the
   selected motion frame, starting with a small residual scale;
4. track root orientation/velocity, feet and contact timing in addition to
   joint position/velocity, with adaptive failure-phase reset sampling;
5. add a train-time mirror-consistency loss; enable mirrored transition
   augmentation only after a one-step next-state equivariance test.
   Inference-time averaging reduced drift but also erased the weak aerial phase
   and remains diagnostic only;
6. require deterministic exact-CPU tracking and ONNX parity before velocity
   or football rewards;
7. then add command diversity, perturbations and action delay, followed by
   three seeds, RCSS single player and 7v7.

Reward-only continuation and action-scale calibration are closed negative
branches. They are not reasons to promote a model that misses contact or drift
gates.

## Search-quality policy

- Academic claims require an original paper or official project page.
- Implementation claims require the official repository at a recorded commit.
- Community/Bilibili videos may reveal a useful implementation lead, but a
  video is not evidence of observation layout, physics parity or reproducible
  performance. Promote a lead only after locating source/config/checkpoint and
  license information.
- A model is called "running" only when exact contact evaluation demonstrates
  the aerial-phase gate; visual appearance or speed alone is insufficient.
