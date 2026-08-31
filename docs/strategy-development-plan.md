# Strategy development and 2D migration plan

Status: accepted next-stage direction

Planning date: 2026-08-31

Runtime: `runtime/apollo/` on WSL2 Ubuntu 22.04

## Mission

Turn the current safe, complete 7v7 match loop into a coordinated football
team. The next stage prioritizes high-level strategy, parameterized ball
actions, teammate coordination, and evidence-based migration of mature
RoboCup2D planning methods.

Apollo remains the 3D execution and world-model foundation. RoboCup2D code is
used as an algorithmic reference, not as a runtime dependency. Every migrated
method must be rewritten against measured 3D robot and ball dynamics.

Locomotion development is not abandoned, but it is no longer the critical
path. The validated Apollo walk, get-up, path planner, and stable contact kick
remain the release baseline. A new motion-training task is opened only when a
measured strategy requirement cannot be met by alignment, timing, and the
existing safe action primitives.

## Stage definition of done

This stage is complete only when the release runtime can demonstrate all of
the following in repeatable RCSSServerMJ tests:

1. select among `Shoot`, `Pass`, `Dribble`, `Clear`, and `Move/Hold` using a
   logged, deterministic action evaluator;
2. execute a target-directed short pass inside an empirically validated
   distance and angular envelope;
3. coordinate passer and receiver through a loss- and delay-tolerant intent
   protocol;
4. generate and reject direct, leading, and through-pass candidates using 3D
   ball travel and humanoid reach-time estimates;
5. perform off-ball support, unmarking, defensive marking, and phase-dependent
   formation changes;
6. complete strict 7v7 matches without regressing connection, shutdown,
   finite-action, fall-recovery, or illegal-defense gates;
7. preserve enough decision and outcome telemetry to replay why each tactical
   action was selected and whether it succeeded.

The stage does not require end-to-end multi-agent reinforcement learning,
11v11 formation files, or an unbounded multi-step planner.

## Current gap analysis

| Layer | Available now | Required next |
|---|---|---|
| World state | Canonical team frame, filtered ball, visible/shared players, play mode | Score, time context, possession confidence, observation freshness, tactical phase |
| Team structure | Seven roles, dynamic assignment, formation positions | Stable tactical ownership, support/mark assignments, role-change hysteresis |
| Navigation | Opponent-aware A* walk planning | Reach-time estimates and action-specific arrival poses |
| Ball action | Safe fixed forward-contact `KickCommand` | Target, receiver, requested speed/range, action ID, result |
| Communication | Eight-byte periodic pose/ball/opponent/role packet | Multiplexed intent, acknowledgement/readiness, sequence and expiry |
| Decision | Behavior tree, fixed goal push, set-play diagonal relay | Candidate generation, legality/executability filters, field evaluation, action selection |
| Verification | Unit tests, deterministic probes, strict and visual 7v7 | Tactical fixtures, pass scenarios, A/B self-play metrics, decision replay |

The first dependency is the strategy-to-execution contract. Porting a pass
planner before extending `KickCommand`, communication, and telemetry would
create decisions that the robot cannot faithfully execute or verify.

## Target architecture

```text
WorldSnapshot + team packets
          |
          v
TacticalStateBuilder
  possession / phase / freshness / score-time context
          |
          v
Candidate generators
  shoot / direct pass / leading pass / through pass / dribble / clear / move
          |
          v
Hard filters
  rule legality / field bounds / action envelope / teammate readiness
          |
          v
3D predictors
  ball trajectory / robot reach time / opponent interception margin
          |
          v
FieldEvaluator + ActionPlanner
  deterministic score / risk / hysteresis / one-step commitment
          |
          +---------------------+
          |                     |
          v                     v
PassCoordinator            Behavior tree
 intent / ready / expiry    priority and safety fallback
          |                     |
          +----------+----------+
                     v
            HighLevelCommand
                     |
                     v
          Apollo motion execution
```

Planning starts at depth one and is recomputed every decision cycle. Depth two
is enabled only after one-step predictions are calibrated and stable. Longer
2D-style action chains are explicitly deferred because 3D action duration,
falls, and perception uncertainty invalidate long open-loop plans quickly.

## Source migration map

The primary algorithm reference is Cyrus2DBase at audited commit
`3459d1b2a0627231fa11f775658328d7a2de41b7`. Pyrus2D is a secondary readable
reference. Apollo, FC Portugal, and Magma remain the preferred references for
3D execution, behavior integration, and formation mechanics.

| Reference concept | Project decision |
|---|---|
| `CooperativeAction` (`Pass`, `Shoot`, `Dribble`, `Clear`, `Move`) | Reimplement as a small immutable 3D action description with target, receiver, requested ball speed, predicted duration, confidence, and provenance |
| Direct/leading/through pass generators | Reimplement geometry and candidate enumeration; replace all 2D cycle and decay assumptions |
| Strict pass interception checks | Preserve the receiver-versus-opponent race structure; use measured humanoid rotation, acceleration, travel, and fall margins |
| Field evaluator | Preserve feature-based comparison and terminal goal states; fit features and weights to the 7v7 field and telemetry |
| Receiver intention and pass communication | Reimplement as an explicit sequence-numbered state machine in the eight-byte communication budget |
| Action-chain graph | Defer; introduce depth two only after one-step calibration gates pass |
| Formation and role files | Migrate tactical zones and phase semantics, not 11v11 coordinates or role counts |
| 2D dash, turn, kick, stamina, player-type, and ball-decay code | Reject as non-transferable runtime physics |

MIT-licensed reference code may be adapted only with its notices retained.
The default approach is clean reimplementation with source attribution. Do not
vendor librcsc, the 2D server, or a full 2D team dependency tree.

Primary references:

- <https://github.com/Cyrus2D/Cyrus2DBase>
- <https://github.com/Cyrus2D/Pyrus2D>
- <https://github.com/robocup3d/op2>
- <https://github.com/m-abr/FCPCodebase>
- <https://github.com/magmaOffenburg/magmaRelease>

## Implementation sequence and promotion gates

### S0: freeze baseline and build tactical fixtures

Deliverables:

- record the current executable, policy, source, server, and scenario hashes;
- preserve a strict baseline 7v7 result and representative decision logs;
- add reusable server scenarios for 1v0 kick, 2v0 pass, 2v1 pass, 3v2 attack,
  and defensive marking;
- define a machine-readable tactical telemetry schema before changing policy.

Gate:

- the existing native tests and strict 7v7 acceptance still pass;
- each scenario has an explicit initial state, bounded duration, expected
  safety invariants, and reproducible output location.

### S1: strategy and execution contracts

Deliverables:

- add `strategy::CooperativeAction` and stable action/rejection enums;
- extend `KickCommand` with target point or direction, requested ball speed or
  range class, receiver, foot/mode hint, and action/sequence ID;
- surface canonical own/opponent score and match-time context in
  `WorldSnapshot`;
- add `TacticalState` for possession, attack/defend/transition phase, freshness,
  and current commitment;
- emit candidate count, rejection reasons, chosen utility, target, receiver,
  and fallback reason in telemetry.

Compatibility rule:

- a default-constructed kick must retain the current safe forward-contact
  behavior until the new executor is promoted.

Gate:

- unit tests cover canonical left/right score mapping, finite fields, enum
  stability, serialization, and default fallback;
- a strict 7v7 regression shows no behavioral change when strategy selection
  is disabled.

### S2: measured 3D ball and reach models

Deliverables:

- implement a target-aligned kick v1 by walking to a calibrated setup pose,
  orienting the body to the target, and executing the current safe contact;
- collect repeated kick traces by distance, heading, approach offset, and
  relevant robot type;
- fit a simple rolling-ball trajectory model with uncertainty bounds;
- fit a conservative robot reach-time model containing turn, start, travel,
  braking, and action-preparation time;
- version the calibration data and model parameters separately from code.

Initial promotion gate:

- at least one useful 2--5 m short-pass envelope is demonstrated;
- in 20 deterministic trials inside that envelope, at least 16 balls cross a
  one-metre-wide target corridor, median absolute direction error is at most
  12 degrees, and falls do not exceed the baseline by more than one trial;
- prediction residuals and the accepted uncertainty margin are reported, not
  hidden behind the mean result.

If this gate fails after alignment and timing calibration, open a narrowly
scoped directional/power kick training task. Do not restart general locomotion
training.

### S3: one-step pass candidate planner

Deliverables:

- direct-pass candidates to current teammate positions;
- leading-pass candidates based on teammate velocity/target role position;
- through-pass candidates into legal free space;
- hard filters for field bounds, play-mode legality, kick envelope, stale
  observations, blocked lanes, and insufficient receiver advantage;
- receiver and opponent reach-time comparison along the predicted ball path;
- deterministic field-value features for forward progress, goal threat,
  possession retention, boundary risk, and central defensive risk.

Gate:

- table-driven unit tests cover clear, blocked, stale, out-of-bounds, offside
  or rule-invalid where applicable, and opponent-first cases;
- canonical frame reflection produces equivalent decisions;
- no pass is selected without a positive receiver interception margin and a
  currently executable kick request;
- repeated evaluation of the same snapshot gives byte-stable chosen action
  telemetry.

### S4: passer-receiver coordination

Deliverables:

- retain the eight-byte packet size and multiplex state and tactical-intent
  packet types;
- add pass sequence, sender/receiver, quantized target, ETA/expiry, and state;
- implement `Proposed -> Ready -> Committed -> Executed/Cancelled/Expired`;
- make the receiver move to the target while continuing to face/track the ball;
- make the passer revalidate visibility, lane, receiver readiness, and action
  envelope immediately before execution;
- ignore duplicate, stale, out-of-order, opponent-team, and impossible packets.

Gate:

- codec and state-machine tests inject packet loss, duplication, reordering,
  quantization, and expiry;
- no lost acknowledgement can leave a robot permanently committed;
- at least 16 of 20 deterministic 2v0 trials complete both the protocol and
  receiver-zone ball arrival inside the calibrated pass envelope.

### S5: behavior-tree action selection

Deliverables:

- add one high-level selector below rule/fall/beam safety nodes and above
  role-specific open-play behaviors;
- evaluate `Shoot`, `Pass`, `Dribble`, `Clear`, and `Move/Hold` together;
- retain set-play restrictions, goalkeeper safety, obstacle avoidance, and the
  current fixed behavior as a same-cycle fallback;
- add hysteresis and a bounded commitment window to prevent pass/dribble
  oscillation;
- expose all hard rejections independently from utility scores.

Gate:

- 2v1 and 3v2 scenarios exercise pass selection, pass rejection, receiver
  motion, safe cancellation, shot selection, and defensive clear;
- invalid/non-finite predictions always fall back in the same cycle;
- no new rule violation, client failure, or unrecovered commitment is observed.

### S6: team tactics and bounded two-step planning

Deliverables:

- support and unmark targets derived from passing lanes and opponent pressure;
- defensive marking assignments and coverage of dangerous lanes;
- attack, defend, and transition formation variants with role-change
  hysteresis;
- score/time risk modes such as protect-lead and chase-goal;
- depth-two action evaluation for selected pass-then-shoot or pass-then-pass
  cases, with a strict computation and uncertainty budget;
- set-play templates that use the same pass executor and coordinator as open
  play.

Gate:

- assignments are deterministic, unique where required, and stable under
  small observation noise;
- depth two must beat or match depth one on a fixed tactical scenario suite
  without missing the server decision budget;
- disabling depth two remains a runtime fallback.

### S7: match validation and release promotion

Validation ladder:

1. native unit/property tests;
2. recorded-snapshot decision replay;
3. 1v0 kick calibration;
4. 2v0 pass protocol;
5. 2v1 interception and cancellation;
6. 3v2 attack and defensive transition;
7. headless 7v7 A/B runs with swapped sides;
8. visual 7v7 review;
9. source-complete competition package test.

Minimum release gate:

- 2v0 pass completion is at least 80% and 2v1 completion is at least 60% over
  at least 20 accepted trials per suite;
- five consecutive strict 7v7 runs produce at least three intentional pass
  attempts per match and at least 45% completed passes in aggregate;
- a two-pass possession chain is reproduced in at least three bounded runs;
- all 14 clients connect, join, reach `PlayOn`, and exit cleanly;
- fatal errors, non-finite motor actions, and illegal-defense fouls remain zero;
- fall and recovery results are no worse than the paired baseline beyond the
  declared tolerance;
- the candidate improves pass/possession metrics without a material regression
  in territory, goal differential, or rule safety across swapped-side A/B runs.

These are initial engineering thresholds, not claims about competition rank.
After the first calibrated data set, threshold changes require a documented
rationale and must not be made merely to admit a failing candidate.

## Planned source layout

New strategy code should be isolated from motion execution:

```text
runtime/apollo/src/strategy/
  cooperative_action.h
  tactical_state.h/.cc
  ball_trajectory_model.h/.cc
  reach_time_model.h/.cc
  pass_candidate_generator.h/.cc
  field_evaluator.h/.cc
  action_planner.h/.cc
  pass_coordinator.h/.cc

runtime/apollo/tests/
  tactical_state_test.cc
  ball_trajectory_model_test.cc
  reach_time_model_test.cc
  pass_candidate_generator_test.cc
  field_evaluator_test.cc
  pass_coordinator_test.cc
  strategy_replay_test.cc
```

`strategy_core` should depend on world and communication types. The decision
tree may depend on `strategy_core`; strategy code must not depend on behavior
or ONNX execution code. This keeps prediction and decision logic testable
without loading policies or connecting to the server.

## Telemetry and experiment record

Every strategy decision record must include:

- server cycle, player, play mode, score, and tactical phase;
- world-state freshness and possession confidence;
- candidate ID/type/source, target, receiver, and requested speed/range;
- hard-filter result and stable rejection code;
- predicted ball time, receiver time, nearest-opponent time, and margin;
- utility feature vector, weights/version, total utility, and selected action;
- commitment/protocol state and sequence ID;
- execution start/end/cancel reason;
- observed result: receiver contact, interception, out of play, turnover,
  shot, goal, or timeout.

Large raw logs remain ignored, while schemas, scenario definitions, aggregate
reports, source hashes, and accepted calibration parameters are versioned.

## Risk controls

| Risk | Control |
|---|---|
| 2D timing is too optimistic | Calibrate only from Apollo/RCSS traces and require conservative uncertainty margins |
| Fixed kick cannot produce useful passes | Use a declared calibration gate; trigger a narrow kick task only after measured failure |
| Eight-byte communication is insufficient | Multiplex packet types, quantize targets, use sequence/expiry, and preserve periodic state packets |
| Stale or partial perception creates unsafe passes | Freshness hard filter, confidence floor, opponent uncertainty inflation, immediate pre-kick revalidation |
| Players oscillate between actions | Deterministic tie-breaks, hysteresis, bounded commitments, explicit cancellation |
| Strategy introduces fouls | Rule legality is a hard filter before utility evaluation and remains above tactics in the behavior tree |
| Long action chains become invalid | Depth one by default, depth two behind a gate, replan every cycle |
| Imported code obscures provenance | Record source commit/license, retain notices, prefer independent implementation, prohibit opaque binary strategy imports |
| Sparse goals hide regressions | Track possession chains, pass completion, interception margin, territory, turnovers, falls, and rules in addition to score |

## Immediate implementation slice

The first coding slice after this plan is deliberately small and behavior
preserving:

1. add `CooperativeAction`, stable rejection codes, and decision telemetry;
2. surface canonical score/time context in `WorldSnapshot`;
3. extend `KickCommand` while preserving its default behavior;
4. add unit tests and one recorded-snapshot replay harness;
5. rerun native tests and strict 7v7 to prove no baseline regression.

Only after that slice passes does work begin on kick calibration and the first
direct-pass generator.
