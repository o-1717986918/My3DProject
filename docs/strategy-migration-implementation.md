# One-step passing migration: implementation record

Status: historical first-delivery record with 2026-09-05 strategy closure
appendix

Implementation date: 2026-08-31

Runtime: Apollo C++17 on WSL2 Ubuntu 22.04

## Delivered scope

This increment turns a RoboCup2D planning pattern into one independently
implemented, observable 3D match action. During `PlayOn`, the attacking player
can generate direct and one-metre leading-pass candidates, reject physically
or tactically unsafe candidates, announce the selected target, wait for the
receiver's readiness message, align behind the ball, and issue a typed
targeted-pass command. An unsupported or rejected targeted request is now
converted to a neutral no-contact hold at the motion boundary; the previous
forward-contact macro is available only when the strategy explicitly requests
the fixed-contact capability.

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

This was the first strategy migration closure. The later code-level completion
is recorded in the 2026-09-05 appendix below and in
`team-excellence-roadmap.md`.

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
  sequence ID, exact `KickMode`, and (for restarts) an epoch/revision pair.
  A default command still means the former safe forward contact, while an
  invalid targeted request is rejected into a neutral no-contact hold.
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
- Team communication remains exactly eight bytes. Version 3 multiplexes the
  original state packet with checksummed pass lifecycle intents; the packet
  identifies passer/receiver authorship, lifecycle state, target, sequence,
  speed, and ETA. Intent freshness is bounded to 30 cycles.
- Before contact, the passer cancels or replans if the receiver is invalid or
  fallen, the target leaves the field, the ball moves more than 0.75 m from the
  planned start, or the receiver moves more than 1.25 m from a direct target
  (2.0 m for a leading target). Commitment is bounded to six seconds.
- The receiver moves toward the proposed target while facing the ball. Lost or
  stale intents cannot permanently block the existing behavior.
- Telemetry includes score/time, phase, possession, candidate and rejection
  counts, pass type, action/sequence IDs, receiver, ready state, target,
  interception margin, utility, and exact kick mode.

The currently shipped residual table narrows the experimental targeted-pass
execution envelope to a 1.45--2.55 m target, +/-2 degrees, and 1.43 m/s. The
candidate generator still evaluates the larger tactical 1.5--8.0 m space, but
the capability registry prevents a commitment outside the measured residual
envelope. Such a candidate is retained as a diagnostic rejection rather than
silently becoming a fixed forward contact.

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

An in-envelope 900-cycle trace was collected on 2026-09-04 using the same
residual table (the final shared-envelope commit followed this trace). Its
2.42 m target lies inside the now-declared envelope:

```bash
APOLLO_ENABLE_PARAMETERIZED_KICK=1 MATCH_PASS_SCENARIO=1 \
  MATCH_PARAMETERIZED_KICK_SCENARIO=1 MATCH_REQUIRE_PASS=1 \
  MATCH_REQUIRE_KICK=1 APOLLO_STATUS_INTERVAL=5 \
  scripts/run_apollo_acceptance_match.sh 900
```

It completed 14/14 agents cleanly with zero server errors or illegal defense,
12 parameterized-kick samples, 18 Ready samples, and one measured pass-contact
event. The final-envelope code has since passed the default 600-cycle 7v7
gate and all unit tests; a repeat of the parameterized fixture is still a
pending verification item because the receiver reset is timing-sensitive. The
out-of-envelope 3.65 m fixture correctly produced no targeted kick; that is an
intentional safety rejection, not a failed fixed-kick fallback.

To inspect a preserved run:

```bash
conda run -n my3d-team python scripts/analyze_apollo_pass.py \
  /tmp/my3d-pass-scenario-final/My3D-*.log
```

## Current strategy closure and handoff

The subsequent full-team increment is now wired into the decision tree. A
single deterministic `TeamPlan` is generated from the complete role
assignment, with support/unmark candidates, pressure/cover, one-owner
interception, paired centre-back marking, passing-lane blocks, and goalkeeper
goal-line interception. Phase-specific formation and the configurable
`Balanced`, `ProtectLead`, and `ChaseGoal` risk modes are also implemented.
Set plays are coordinated by `RestartCoordinator`
with taker election, receiver positioning, alignment, execution feedback,
release verification, one bounded fallback, and post-release lockout. These
paths are covered by repository unit/integration tests and preserved match
evidence.

The current implementation deliberately does not claim a competition-reliable
physical targeted pass: preserved early trials produced only 0.186--0.644 m,
and the target-directed path remains experimental. Later training delivered
narrow procedural shot and safety-clear anchors, but those contracts are not
evidence of general placement accuracy or receiver first touch.

## 2026-09-05 migration closure

The previously open code-level migration work is now implemented:

- the common planner generates and compares capability-gated Hold, Move,
  Dribble, direct/leading Pass, Shoot, and Clear actions;
- the pass lifecycle actively emits and consumes `Proposed`, `Ready`,
  `Committed`, `Commanded`, `Executed`, `ReceiverZone`, `Received`,
  `Intercepted`, `Out`, `Timeout`, `Cancelled`, and `Expired`;
- motor completion is separated from observed ball execution and terminal
  physical outcomes;
- receiver readiness has a stable dwell; receive intent persists across speech
  gaps, predicts a reachable moving-ball intercept, times out/cancels, and
  releases after local ball control so role reassignment can take over;
- `TeamPlan::plan_all()` carries stable possession owner/phase, freshness and a
  deterministic revision, assigns separated attacking support lanes and unique
  defensive jobs, and protects the second ball during a goalkeeper claim;
- goalkeeper behavior now separates hold, goal-line intercept, safe smother,
  goal-kick execution and capability-gated open-play clear;
- restarts include frozen primary/alternate/safety variants, opponent-aware
  branch selection, feedback, release verification and bounded fallback.

Through-space passes remain intentionally absent until the world model exposes
player velocity or a receiver run intent. Cross-agent plan-revision consensus
and paired opponent A/B evaluation are performance/evidence extensions rather
than an unhandled code path. The authoritative current inventory is
`docs/team-excellence-roadmap.md`.

Training details are maintained in `docs/rl-training-plan.md` and
`docs/model-free-parameterized-kick-plan.md`. End-to-end seven-agent strategy
training remains deferred until replayable labels and stable action semantics
exist.
