# One-step passing migration: implementation and training handoff

Status: first delivery loop accepted

Implementation date: 2026-08-31

Runtime: Apollo C++17 on WSL2 Ubuntu 22.04

## Delivered scope

This increment turns a RoboCup2D planning pattern into one independently
implemented, observable 3D match action. During `PlayOn`, the attacking player
can generate direct and one-metre leading-pass candidates, reject physically
or tactically unsafe candidates, announce the selected target, wait for the
receiver's readiness message, align behind the ball, and issue a typed
targeted-pass command. The previous forward-contact action remains the
same-cycle fallback.

The delivered loop is:

```text
WorldSnapshot
  -> TacticalState
  -> direct/leading candidates
  -> field, reach-time, and interception filters
  -> deterministic utility and action ID
  -> Proposed (passer) / Ready (receiver)
  -> immediate commitment revalidation
  -> target-aligned safe contact
  -> physical ball-progress analysis
```

This is the first strategy migration closure, not the completion of the whole
strategy roadmap. It proves the interfaces, decision path, communication path,
fallback, telemetry, and physical-outcome gate in a complete 7v7 process.

## Local 2D reference audit

The audited source set is
`/home/win98/my_projects/rbc/teams`. It contains two Cyrus2DBase copies,
HELIOS base archives, TheMY, and CppDNN. These extracted directories do not
contain `.git` metadata, so no commit identity can truthfully be assigned to
them. The following file hashes pin the material actually inspected:

| Archive/file | SHA-256 | Concept used |
|---|---|---|
| `Cyrus2DBase-cyrus2d/src/player/planner/cooperative_action.h` | `a0d2f25afe617642242f604d043b66fbfcbd10064bc70997c0ed31a9f527c9c3` | typed cooperative actions |
| `Cyrus2DBase-cyrus2d/src/player/planner/strict_check_pass_generator.cpp` | `adde194b5396ddba0db561d22f876278d46b6e99c212aafcb51ce9bdb78a1510` | direct/leading/through enumeration and interception race |
| `Cyrus2DBase-cyrus2d/src/player/sample_field_evaluator.cpp` | `ac86b179865a70f42b57d4da00b13d5f241cbbabd7c23ae2fb6d16dc2d560d82` | feature-based action comparison |
| `Cyrus2DBase-cyrus2d/src/player/intention_receive.cpp` | `3aa202bd4697f6fd3fbb99121cf45195c284752d24c39522ab4d3b66e74bded4` | receiver intention |
| `TheMY/src/player/planner/pass.cpp` | `8e8eea29e9692d4044cb5d47c300ecd06810bb781fe50c77b15600e5a6c84f3e` | pass representation |
| `TheMY/src/player/planner/simple_pass_checker.cpp` | `f9f484f224702ca384f7cbf94f8b31e0516a051a59c7e287d7d9e69e0833e3b0` | pass feasibility checks |

The archive-level Cyrus2D and TheMY license files have the same MIT-license
digest, `7b5f0ac4d1b931dc9feda7b7f001f4463462842d54029ea8948192dfff78d3f7`,
but inspected source files also carry LGPL-3.0-or-later or GPL-3.0-or-later
headers. Because the archive provenance is incomplete and the notices are
mixed, no source was copied. My3D's implementation preserves only general
algorithmic ideas, uses Apollo types and measured 3D timing, names the design
influence in source comments, and remains under this repository's
GPL-3.0-or-later license.

## Exact implementation contract

Strategy code is isolated in `runtime/apollo/src/strategy/` and compiled as
`strategy_core`; it does not depend on behavior-tree or ONNX execution code.

- `CooperativeAction` carries stable action/sequence IDs, actor, receiver,
  target, pass type, requested speed, predicted times, interception margin,
  utility, and confidence.
- `KickCommand` carries the target, requested speed, receiver, action ID,
  sequence ID, and exact `KickMode`. A default command still means the former
  safe forward contact.
- `TacticalState` records possession, phase, pressure proxies, score, and time.
- The initial ball model uses the measured 1.43 m/s contact speed, 0.08 m/s²
  rolling deceleration, and 0.20 m/s minimum controlled speed. These are seed
  parameters, not a completed calibration model.
- Candidates are limited to 1.5--8.0 m and one metre inside field bounds.
  Leading targets are one metre forward of the receiver. A candidate needs at
  least 0.35 s opponent-interception margin and 0.10 s receiver timing margin.
  Teammate positions older than one second and dangerous backward passes
  behind x=-20 m are rejected.
- The ball path is sampled at 20 points. Opponent arrival is conservatively
  estimated at 1.35 m/s with turn/preparation margins rather than using 2D
  dash cycles or ball decay.
- Candidate ordering has explicit tie-breaks and a stable quantized action ID.
  The minimum selected utility is 1.0.
- Team communication remains exactly eight bytes. Version 2 multiplexes the
  original state packet with checksummed `Proposed` and `Ready` pass intents;
  target, sequence, receiver, speed, and ETA are quantized. Intent freshness is
  bounded to 30 cycles.
- Before contact, the passer cancels or replans if the receiver is invalid or
  fallen, the target leaves the field, the ball moves more than 0.75 m from the
  planned start, or the receiver moves more than 1.25 m from a direct target
  (2.0 m for a leading target). Commitment is bounded to six seconds.
- The receiver moves toward the proposed target while facing the ball. Lost or
  stale intents cannot permanently block the existing behavior.
- Telemetry includes score/time, phase, possession, candidate and rejection
  counts, pass type, action/sequence IDs, receiver, ready state, target,
  interception margin, utility, and exact kick mode.

## Verification evidence

Native unit and integration tests cover strategy determinism and filtering,
the fixed-size intent codec, target quantization/checksum/freshness, receiver
readiness, and behavior-tree selection/fallback.

The strict deterministic full-team gate was run with:

```bash
APOLLO_ENABLE_PARAMETERIZED_KICK=1 MATCH_PASS_SCENARIO=1 \
  MATCH_REQUIRE_PASS=1 MATCH_REQUIRE_KICK=1 \
  APOLLO_STATUS_INTERVAL=5 scripts/run_apollo_acceptance_match.sh 600
```

Observed result:

```text
connections=14 joins=14 play_on=14 clean_exits=14
failures=0 server_errors=0 illegal_defense=0
kick_samples=20 pass_plan_samples=48 pass_ready_samples=36
targeted_pass_kick_samples=20 pass_contact_events=2
```

The source-complete competition archive was then rebuilt, extracted to a clean
temporary directory, and tested through the same 600-cycle gate using only its
packaged binary and assets. It again completed 14/14 cleanly with zero fatal
errors or illegal defense, 11 targeted-pass samples, and one physical contact
event.

A preserved calibration trace at `/tmp/my3d-pass-scenario-final` contains one
unique targeted release with a 4.823 m requested target and 0.644 m maximum
projected progress toward that target. A second run observed only 0.186 m on
one accepted contact and a near-zero miss on another attempt. The analyzer's
0.10 m threshold distinguishes observable intentional progress from the
approximately millimetre-scale stationary noise in these fixtures.

This evidence validates command-to-contact wiring, not useful pass range or
receiver completion. The current fixed contact macro is weak and variable; it
does not yet satisfy the roadmap's 2--5 m corridor or 16/20 completion gates.

To inspect a preserved run:

```bash
conda run -n my3d-team python scripts/analyze_apollo_pass.py \
  /tmp/my3d-pass-scenario-final/My3D-*.log
```

## Deferred strategy work

The current implementation deliberately defers through passes, explicit
`Committed/Executed/Cancelled/Expired` packets, receiver first touch, pass
outcome ownership, shot/dribble/clear comparison in one selector, support and
mark assignments, low-pressure backward-pass suppression, decision replay,
and swapped-side multi-match A/B statistics. Those items stay in the staged
roadmap; none should be inferred from `TargetedPass` telemetry alone.

## Training requirements after this delivery

### T1: parameterized directional short-pass kick (blocking)

This is the next training task because the strategy path has reached the
physical-action bottleneck. Train or optimize a small target-conditioned
policy for the T1 robot, not an end-to-end team policy.

Inputs should include joint position/velocity, torso orientation and angular
velocity, feet contact state, ball position/velocity in the robot frame,
target bearing, requested range or arrival speed, and phase/time. Outputs may
be 23 residual joint targets over a guarded posture reference, or a compact
foot/hip action decoded by a deterministic controller. The exported policy
must keep an explicit observation ordering, scaling, action decoder, control
rate, asset hash, and fallback contract.

Use curriculum stages: stationary balance; contact timing; left/right
direction; 1.5--3 m range; 3--5 m range; approach-offset randomization; then
limited pose, ball, friction, latency, and sensor-noise randomization. Reward
should separately score projected range error, lateral corridor error,
direction error, useful post-contact speed, uprightness, support stability,
joint limit/velocity/torque cost, smooth recovery, and absence of double
contact. A large sparse goal-only reward is not sufficient.

Required data are at least 20 accepted trials per target bin, recording the
initial pose, ball pose, target, joint trace, contact time, ball trajectory,
fall/recovery, and simulator/config hashes. Promotion begins with 2, 3.5, and
5 m targets at -30, -15, 0, 15, and 30 degrees.

The first gate is at least 16/20 balls through a one-metre-wide corridor for
each promoted central range, median absolute direction error no more than
12 degrees, no non-finite actions, and no more than one additional fall versus
the fixed-contact control. The final gate adds ONNX parity, three seeds, both
sides, 2v0 receiver-zone arrival, and strict 7v7 fallback tests.

### T2: receiver first touch and stop (next, conditional)

Do not start this before T1 supplies repeatable incoming trajectories. First
attempt a deterministic receive pose plus walk/stop behavior. Train a receive
policy only if measured 2v0 failures are dominated by humanoid contact and
stabilization rather than communication or target selection. Its gate is
controlled possession after arrival, not merely receiver proximity.

### T3: locomotion maintenance (parallel but non-blocking)

The existing Apollo walk and get-up policies remain the release baseline.
Faster running is valuable for reach-time margins, but it is not the blocker
for this pass closure. Continue the separate R2/R3 run work only behind its
observation-shape, uprightness, finite-output, multi-seed, and 7v7 gates; do
not replace the stable walk to improve a tactical benchmark.

### What not to train yet

Do not train an end-to-end seven-agent strategy policy at this stage. First
calibrate the kick and reach models, implement outcome labels, complete 2v0
and 2v1 fixtures, and collect deterministic planner replays. Utility weights
can then be fitted offline from labeled candidate outcomes. Multi-agent RL is
appropriate only after the action interface and credit signal are stable and
after it can be compared against the deterministic planner with the same
safety fallback.

## Immediate continuation order

1. build the target-conditioned kick data collector and T1 training/export
   contract;
2. replace seed ball parameters with fitted distributions and uncertainty;
3. add `Committed/Executed/Cancelled/Expired` local state plus outcome labels;
4. add repeatable 2v0 and 2v1 trial batches and receiver-zone metrics;
5. add through-pass/free-space generation and tactical no-benefit filters;
6. implement support, unmarking, marking, and phase-specific formation;
7. run paired, side-swapped 7v7 A/B suites before any release claim.
