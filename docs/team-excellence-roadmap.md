# Complete and excellent team capability roadmap

Status: active and authoritative capability plan

Last audited: 2026-09-04

Authoritative runtime: `runtime/apollo/`

Accepted committed baseline: `b2a4bbc`

## Objective and scope

Build a complete, fluent, and strong seven-player team whose decisions make
maximum honest use of every available action. The active priority is the
team's football ability:

- shared match-state interpretation and stable role ownership;
- coherent attack, defense, transition, support, marking, and set plays;
- complete action and pass lifecycles, including reception and replanning;
- role-specific striker, midfielder, defender, and goalkeeper behavior;
- calibrated improvement from replay, opponent tests, and later learning;
- robust locomotion and ball skills exposed through one capability contract.

Visualization, launchers, Windows/WSL integration, packaging, and tournament
administration remain maintenance work. They may verify the team but do not
consume team-capability milestones.

`Excellent` is an engineering target, not a championship claim. The roadmap
does not claim a physical pass, shot, clear, or learned policy until its
measured envelope exists. Conversely, a weak physical action does not block
code-level work that can already select, coordinate, reject, or replan around
the capabilities actually present.

## Route decision after source review

The selected route is a hierarchical team stack:

```text
world evidence + communication
              |
              v
state/possession/phase consensus
              |
              v
global roles and tactical duties
              |
              v
typed action candidates + bounded evaluation
              |
              v
action lifecycle and receiver/teammate intentions
              |
              v
capability registry
  learned ONNX skill -> procedural fallback -> safe rejection
```

The route is based on the following primary-source observations:

- [FCPCodebase](https://github.com/m-abr/FCPCodebase) publicly supplies strong
  low-level skills, world/communication infrastructure, and a sample team, but
  describes the sample high-level behavior as one active player attempting a
  kick while the others hold a basic formation. This is useful motion and
  runtime reference, not a complete tactical donor.
- The [UT Austin Villa 3D base](https://github.com/robocup3d/op2) and the
  team's published architecture emphasize walk, kick, world state,
  communication, positioning, and role assignment. Public 3D bases therefore
  do not remove the need for this project's own complete team layer.
- [HELIOS base](https://github.com/helios-base/helios-base) documents online
  cooperative tree search. Its typed action generators, evaluator, and bounded
  search structure are transferable; 2D dash cycles, ball decay, and player
  physics are not.
- [SCRAM](https://www.cs.utexas.edu/~AustinVilla/details/AAAI15-MacAlpine.html)
  minimizes formation makespan while avoiding assignment collisions, and its
  3D deployment synchronizes the mapping through communication. The related
  [prioritized marking work](https://www.cs.utexas.edu/~AustinVilla/details/LNAI16-MacAlpine.html)
  extends this idea to threat-ordered defensive jobs under partial, noisy
  observations. These results directly motivate a constrained full-team duty
  assignment and plan-agreement mechanism.
- The original [pre-planned set-play description](https://www.cs.utexas.edu/~pstone/Papers/99aij/node19.html)
  includes role mapping, alternative actions, successful termination, and
  role-specific timeouts. The current restart coordinator already implements
  the skeleton; the missing work is a variant playbook and outcome-aware
  selection, not another restart state machine.
- A peer-reviewed 2025 study on
  [RL inside a classical robot-soccer stack](https://www.cs.utexas.edu/~AustinVilla/details/zhihan_wang_ICRA25.html)
  decomposes behavior into learned sub-behaviors selected by a structured
  controller. The peer-reviewed
  [FC Portugal skill-set-primitive study](https://doi.org/10.1007/s00521-025-11151-3)
  similarly supports reusable learned motor primitives. These sources support
  retaining deterministic team semantics while improving motion through
  replaceable ONNX skills; they do not justify monolithic seven-agent RL now.

The conclusions above are project inferences from those sources, not claims
that their robot parameters or tactics can be copied unchanged. Detailed
source and provenance notes remain in `docs/reference-projects.md`,
`docs/strategy-migration-implementation.md`, and
`docs/paid-wbc-fsm-audit-2026-09-01.md`.

## Audited capability inventory

Legend: **done** means implemented and covered by repository tests; **partial**
means a truthful path exists but important states or physical envelopes are
missing; **missing** means it is not yet an executable team capability.

| Area | Status | Current implementation | Remaining work |
| --- | --- | --- | --- |
| Canonical world, legal play modes, role-local state | done | canonical team frame, bounded ball tracks, legality guards, per-`DecisionManager` behavior state | add a team-consensus tactical snapshot and explicit plan revisions |
| Dynamic formation roles | done | goalkeeper/AP selection, sticky minimum-cost assignment of remaining roles, fallen-GK replacement | make assignment capability-aware, collision-aware, and agreed across agents |
| Full-team tactical plan | done | one `plan_all()` assigns formation, support, unmark, outlet, pressure, cover, mark, block, intercept, receive, and goalkeeper duties | replace sequential special cases with one constrained global duty assignment |
| Formation phases and score/time risk | done | attack/defend/transition/set-play plus `Balanced`, `ProtectLead`, and `ChaseGoal`; threshold is configurable | add possession hysteresis and explicit transition/counter-press timing |
| Open-play off-ball movement | partial | support lanes, unmarking, one interceptor, paired centre-back marking, cover/outlet points, short target latches | assign multiple support/defensive slots jointly; add switch hysteresis and path deconfliction |
| Restarts | partial | deterministic taker and receiver, positioning, Ready, alignment, execution feedback, release verification, bounded fallback, lockout | add legal variants for each restart, penalty/offside policy, branch selection, and receiver continuation |
| Pass planning | partial | direct and leading candidates, receiver/opponent arrival races, lane and field filters, deterministic evaluator | add through-space candidates, negative-value filters, uncertainty, and comparison with non-pass actions |
| Pass communication | partial | validated `Proposed -> Ready`, authorship, sequence ID, expiry window; Ready checks target distance, ball-facing angle, upright state, speed, ball validity, and `PlayOn` | drive `Committed` through terminal outcome states from physical evidence; add stable Ready dwell and first-touch feasibility |
| Motion feedback | partial | next-cycle `Running/Completed/Rejected/TimedOut`; failed matching kicks cancel and replan; restart consumes feedback | connect successful kick feedback to the general pass lifecycle and observable ball outcome |
| Receiver behavior | partial | receiver walks to the proposed target and faces the ball | persistent intention, timeout/cancel, predicted moving-ball intercept, first touch, and next-action handoff |
| Unified action choice | missing | type and capability registry exist; planner currently generates only passes | generate and compare executable hold, move, dribble, pass, shoot, and clear actions |
| Goalkeeper | partial | goal-angle hold point, reachable goal-line intercept, goal-kick participation, fallen-GK replacement | forward smother/block, recovery, boundary-safe clear/pass distribution, and rebound ownership |
| Physical ball actions | partial | stable approach/recover and forward contact; experimental targeted residual pass; active work adds one procedural short-touch fallback | calibrate useful pass/shot/clear envelopes; retain ONNX-primary/procedural-fallback/safe-reject order |
| Locomotion | partial | stable competition walk/get-up; full-body fast-walk candidate is wired opt-in | reduce lateral drift/falls and calibrate directional reach time without blocking team code |
| Decision evidence | missing | unit/integration fixtures and match logs exist | deterministic replay corpus, decision budget metrics, outcome labels, and paired opponent A/B reports |

### Completed items removed from the active plan

The following are retained as regression coverage, not repeated as future
milestones:

- creation of `TeamPlan::plan_all()`;
- support/unmark candidate generation and short target latching;
- unique reachable interception and paired centre-back marking;
- goalkeeper reachable goal-line crossing logic;
- phase-specific formations and configurable score/time risk modes;
- restart taker/receiver election, alignment, feedback, fallback, and lockout;
- target-angle/speed/distance executability checks and no-contact rejection;
- next-cycle motion execution feedback and failed-pass cancellation;
- removal of process-global AP/GK/kick state in favor of decision-instance
  ownership;
- basic `Proposed -> Ready` validation.

Detailed K0/K1/K2 training failures, smoke runs, and historical thresholds are
also removed from this roadmap. They belong in `docs/rl-experiment-log.md` and
`docs/kick-transition-development.md` so that a capability plan does not become
an experiment transcript.

## Missing capabilities added by this audit

The previous roadmap did not explicitly plan the following work.

### Shared state and plan consistency

- assign every tactical snapshot a freshness window, canonical ordering, and
  deterministic revision/hash;
- estimate an explicit ball owner, not only `Ours/Theirs/Contested`, and apply
  switch hysteresis so AP/phase/duty do not flap around equal reach times;
- report compact plan revision and critical ownership claims through the
  existing communication budget;
- measure disagreement and resolve it deterministically instead of assuming
  that seven partial world views produce the same plan;
- degrade safely when communication, ball, opponent, or teammate evidence is
  stale.

### Global assignment and movement coordination

- generate prioritized tactical jobs before assigning players to them;
- solve pressure, intercept, marking, cover, blocking, outlets, and support as
  one constrained assignment, respecting goalkeeper/AP locks and reach time;
- prevent duplicate jobs, crossing paths, teammate crowding, and formation
  deadlock;
- add duty-switch cost and minimum residence time without retaining a bad job
  after possession or legality changes;
- support fewer than seven upright/visible players and recover assignments as
  fallen players return.

### Possession, action, and outcome semantics

- add a stateful possession tracker with owner identity, confidence, loose-ball
  state, rebound detection, and authoritative reset on play-mode changes;
- model ball trajectories with calibrated uncertainty rather than one fixed
  deceleration assumption;
- give every action an ID, capability envelope, deadline, interrupt rules,
  predicted outcome, observed outcome, and next-action handoff;
- keep unavailable actions out of selection, while letting existing hold,
  walk, forward contact, short touch, and opt-in targeted pass participate
  honestly;
- protect against own-goal, sideline, goal-line, and central-turnover risk in
  pass, dribble, clear, and goalkeeper distribution.

### Perception and real-time behavior

- define an attention/head-look policy for ball reacquisition, receiver
  confirmation, opponent scanning, and goalkeeper trajectory tracking;
- expose tactical decision latency, candidate count, and communication packet
  use, with a bounded fallback when the cycle budget is exceeded;
- keep local obstacle avoidance but add teammate path intent or reservation for
  high-conflict coordinated moves.

### Evaluation and adaptation

- serialize decision inputs, candidates, rejections, selected actions,
  assignments, motion feedback, and physical outcomes into replayable records;
- add side-swapped and seed-controlled opponent fixtures instead of relying on
  one symmetric self-play run;
- track possession recovery, territorial progress, receiver advantage, pass
  completion, turnover location, shot creation, central chances conceded,
  goalkeeper stops, falls, and plan disagreement;
- fit evaluator weights only after labels exist, and compare learned weights or
  policies with the same deterministic capability and legality layer.

## Decision-first delivery sequence

Work proceeds in the following order. Motion training continues in parallel,
but it does not postpone code-level team behavior that can use current actions.

### D1: consistent team state and constrained duty ownership

Deliver:

1. a stateful `PossessionTracker` owned by each `DecisionManager`, with ball
   owner ID, confidence, hysteresis, loose-ball/rebound state, and play-mode
   reset;
2. a canonical `TacticalSnapshot` and `TeamPlan` metadata containing source
   cycle/time, revision, freshness, and ownership claims;
3. compact communication of AP/interceptor/taker and plan revision, plus a
   deterministic disagreement fallback;
4. one prioritized assignment stage for formation roles and tactical duties,
   including switch cost, fallen-player recovery, and collision/crowding cost;
5. tests where observation ordering, small noise, stale packets, falls, and
   opposite-side frames do not create duplicate critical duties.

Completion evidence is code and replay fixtures showing one AP, at most one
interceptor/taker, unique marked threats, bounded plan disagreement, and legal
fallbacks. It is not a demand that every motion model already be strong.

### D2: complete action and pass lifecycle using current abilities

Deliver:

1. an explicit local pass state machine:
   `Proposed -> Ready -> Committed -> Commanded -> Executed ->`
   `ReceiverZone/Received/Intercepted/Out/Timeout`, with
   `Cancelled/Expired` from either participant;
2. success transitions driven by matching motion feedback plus ball evidence,
   never by command issue alone;
3. stable Ready dwell, incoming-trajectory feasibility, receiver intention
   timeout/cancel, and a moving-ball intercept target;
4. outcome ownership after contact, rebound, interception, out-of-bounds, or
   lost observation;
5. typed generators for `Hold`, `Move`, current procedural `DribbleTouch`, and
   executable passes; `Shoot` and `Clear` remain excluded until their physical
   envelopes exist;
6. a single evaluator with risk-mode, boundary, pressure, uncertainty, and
   capability terms, plus deterministic tie breaking;
7. direct, leading, and conservative through-space pass candidates when a
   receiver run and opponent race make them executable.

This milestone maximizes current motion value: the team can already choose to
hold shape, move, touch forward, or attempt the explicitly enabled pass path
without pretending that unavailable shot/clear skills exist.

### D3: complete positional role behavior

Deliver:

- **Active player:** select among available hold, dribble/touch, pass, goalward
  contact, and retreat/recover; avoid unsafe backward or boundary actions.
- **Striker:** create width/depth, time a run, receive/intercept a pass, turn or
  relay with available actions, and recover the second ball.
- **Central midfielders:** assign distinct support/outlet/cover jobs, preserve
  rest defense, and become a receiver or ball challenger through explicit
  ownership transfer.
- **Defenders:** threat-rank all known opponents, assign pressure/cover/mark/
  block globally, delay rather than double-chase, and choose safe distribution
  or goalward clearance only when executable.
- **Goalkeeper:** hold angle, intercept reachable crossings, smother reachable
  loose balls, recover, own rebounds, and distribute through the common action
  planner.
- **Degraded team:** maintain a legal useful shape with fallen, stale, or
  temporarily missing players, then restore roles without oscillation.

Role completion is demonstrated in snapshot/replay sequences, not only one
static target per role.

### D4: coordinated attack, defense, transition, and set plays

Deliver:

1. multiple jointly assigned support slots for width, depth, outlet, third
   player, and rebound balance;
2. controlled overloads and pass-run coordination without vacating rest
   defense;
3. possession-loss counter-pressure with a short deadline, otherwise compact
   retreat; possession-gain outlets and forward support;
4. threat-ranked marking, central-lane protection, defensive line depth, and
   explicit press triggers;
5. a set-play playbook with at least two legal variants where meaningful,
   opponent-aware branch choice, role-specific timeout, and open-play handoff;
6. explicit kickoff, goal-kick, kick-in, corner, free-kick, offside, and penalty
   behavior, reusing the same action lifecycle and capability checks;
7. bounded depth-two evaluation only for a small executable subset such as
   pass-then-contact or pass-then-pass. No long open-loop action chain is
   allowed.

### D5: evidence-driven team improvement

Deliver:

1. deterministic decision replay and a labeled action-outcome corpus;
2. paired sides/seeds against at least a passive, pressing, deep, and direct
   opponent profile;
3. calibrated field-evaluator and reach/ball models with held-out reporting;
4. opponent-style estimates that modify soft preferences, never legality or
   action executability;
5. optional offline imitation/ranking and later multi-agent learning only when
   it beats the deterministic baseline on held-out opponents without increasing
   falls, illegal actions, plan disagreement, or catastrophic turnovers.

## Parallel motion and training line

The motion route remains active but subordinate to the team interface:

1. retain the stable Apollo walk and get-up as the default;
2. keep the full-body fast-walk actor opt-in until lateral drift, fall rate,
   stop/turn transitions, and direction-dependent reach time improve;
3. retain ONNX as the preferred future parameterized ball-action path;
4. retain the deterministic procedural trajectory as teacher and fallback;
5. first calibrate short touch and directional short pass, then shot/clear, then
   moving-ball receive and contact;
6. publish only the measured sub-envelope of each action through
   `ActionCapabilityRegistry`;
7. write all training artifacts to `/home/win98/rl_runs`, never the C-drive
   repository.

The detailed action route is in
`docs/model-free-parameterized-kick-plan.md`; the ONNX and curriculum route is
in `docs/rl-training-plan.md`. Full end-to-end seven-agent RL, a direct
`wbc_fsm` port, and copying another robot's joint trajectories are explicitly
deferred because they do not close the current team-decision gaps.

## Lightweight acceptance record

Acceptance checks document whether a delivered capability works; they do not
reorder development or block unrelated team code.

For every increment, preserve only the checks relevant to the change:

- deterministic unit/snapshot/replay tests for semantics and edge cases;
- one integration sequence that proves the decision reaches an executable
  action or an honest safe fallback;
- decision latency, non-finite output, fall, legality, and process-health
  regression checks where affected;
- physical trials only for claims about movement or ball outcomes;
- side/seed/opponent comparisons only for claims about team strength.

## Target capability levels

| Level | Meaning | Current assessment |
| --- | --- | --- |
| C0 robust athlete | stable movement, stop/turn/recover, calibrated reach/fall risk | partial: stable baseline exists; fast-walk candidate is not ready |
| C1 complete individual | executable pass, dribble, receive, shot, clear with lifecycle/outcomes | partial: approach/contact and one short-touch path exist; complete ball set does not |
| C2 complete positional unit | all seven roles make useful on/off-ball choices and recover from degraded state | partial: tactical targets exist; persistent role intentions and several GK/receiver behaviors do not |
| C3 coordinated team | shared plan, full lifecycle, coherent attack/defense/transitions/set plays | partial: deterministic plan/restarts exist; consensus, global assignment, full outcomes, and playbook variants do not |
| C4 excellent team | repeatably better outcomes across sides, seeds, states, and opponent styles | missing: replay corpus and comparative evidence are not yet present |

## Immediate implementation queue

The next code batches are authoritative until this document is audited again:

1. **D1-A:** possession owner/hysteresis and tactical snapshot/plan revision;
2. **D1-B:** prioritized global duty assignment and disagreement telemetry;
3. **D2-A:** general pass lifecycle plus physical outcome tracker;
4. **D2-B:** receiver intention and moving-ball intercept/first-touch handoff;
5. **D2-C:** unified executable-action generators and evaluator;
6. **D3:** role sequences, goalkeeper smother/rebound/distribution, degraded
   team recovery;
7. **D4:** coordinated attack/defense transitions and set-play variants;
8. **D5:** replay/A-B corpus and calibrated or learned soft preferences;
9. **parallel motion:** ONNX ball-action training with procedural fallback and
   the same capability contract.

When code proves an item complete, move it to the audited inventory and remove
it from this queue. Do not preserve completed work as a future milestone and do
not let historical experiment detail grow back into this roadmap.
