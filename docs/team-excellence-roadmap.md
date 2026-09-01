# Complete and excellent team capability roadmap

Status: active project-level capability target

Target adopted: 2026-08-31

Authoritative runtime: `runtime/apollo/`

## Target and boundary

Build a complete and excellent seven-player robot football team. The target is
the capability of the team itself:

- robust and fast humanoid movement;
- complete ball interaction, including approach, first touch, dribble, pass,
  through pass, shot, clear, and recovery;
- competent goalkeeper play;
- coherent distributed attack, defense, transition, support, marking, and set
  plays;
- score/time-aware tactical choices and repeatable strength against different
  opponent styles;
- later, evidence-driven learning that improves decisions without removing the
  deterministic safety and rule layer.

Competition-machine setup, operating-system migration, launcher scripts,
packaging, and tournament administration are outside this development route.
They remain maintenance/release checks and must not regress, but they do not
drive capability priorities or consume capability milestones.

`Excellent` is a measured engineering tier, not an unsupported championship
claim. A team that connects, walks, touches the ball, or completes two halves
is operational; it is not excellent until it passes the C0--C4 capability
levels below.

## Capability promotion levels

### C0: robust humanoid athlete

- stable stand, walk, run, lateral motion, turn, brake, and stop;
- smooth transitions among movement, ball setup, contact, recovery, and get-up;
- no non-finite outputs or unguarded learned-policy execution;
- reach-time and fall-risk models calibrated from the real server;
- every learned candidate retains the existing stable walk/get-up fallback.

### C1: complete individual footballer

- target-conditioned pass, shot, and clear over useful measured envelopes;
- controlled dribble and change of direction without ball holding;
- receive, cushion/stop, and immediate next-action readiness;
- moving-ball contact and recovery from imperfect approach poses;
- action choice reflects target direction, requested range/arrival speed,
  pressure, field boundary, and fall risk;
- each skill exposes success, failure, duration, uncertainty, and physical
  outcome to the decision layer.

### C2: complete positional unit

- goalkeeper predicts goal-line crossings, positions, blocks, recovers, and
  distributes;
- striker times runs, receives, turns, and finishes;
- active player chooses among shoot, pass, dribble, clear, and hold;
- midfielders provide passing angles, retain balance, press, and protect
  transitions;
- defenders mark dangerous opponents, cover lanes, delay attacks, and clear
  safely without crowding the goal area;
- every role remains useful off the ball rather than merely walking to a static
  formation coordinate.

### C3: coordinated team intelligence

- direct, leading, and through passes use calibrated receiver/opponent races;
- pass coordination covers proposal, arrival readiness, commitment, execution,
  reception, interception, cancellation, expiry, and next action;
- attack, balanced, defend, both transitions, protect-lead, and chase-goal
  phases change formation, action risk, pressing, and support behavior;
- support, unmarking, overload, width, depth, pressure, cover, balance, marking,
  passing-lane blocking, and counter-attack anchors are assigned coherently;
- kickoffs, goal kicks, kick-ins, corners, free kicks, and penalties use the
  same action library and coordinated role logic as open play;
- bounded depth-two planning evaluates selected pass-then-shoot and
  pass-then-pass cases without long open-loop chains.

### C4: excellent team

- the team outperforms its accepted baseline across multiple opponent styles,
  sides, seeds, and match states, not only symmetric self-play;
- it creates and converts higher-quality chances while conceding fewer central
  chances;
- possession, territorial progress, pass chains, shots, goals, recoveries, and
  defensive stops improve together rather than through a single exploited
  metric;
- tactical weights are calibrated from labeled outcomes and validated out of
  sample;
- opponent modeling and late-stage self-play improve preferences while hard
  legality, executability, motion guards, and fallback remain deterministic;
- strength remains stable under perception loss, communication loss, pressure,
  falls, score/time changes, and unfamiliar formations.

## Current capability baseline

Accepted baseline: commit `793c5af`.

Available now:

- Apollo learned walk and four-direction get-up;
- canonical world state, filtered ball, obstacle-aware navigation, dynamic role
  assignment, simple ball-relative formation, and compact team communication;
- stable approach/contact/recover fallback;
- one-step direct and leading pass candidates;
- `Proposed -> Ready` communication, receiver target motion, safe commitment
  expiry, and physical contact telemetry;
- strict 7v7 evidence that the strategy-to-contact path runs without breaking
  the existing team loop.

Binding capability gaps, in priority order:

1. the current contact moves the ball only about 0.186--0.644 m in preserved
   trials, so useful pass, shot, clear, and goalkeeper distribution do not yet
   exist;
2. the active motion layer ignores requested target speed/range and always
   executes the same forward-walk contact macro;
3. the experimental run policy has not passed simulator, ONNX, server,
   transition, and multi-seed promotion gates;
4. receiver readiness does not yet prove arrival, stable ball-facing posture,
   or a feasible first-touch window;
5. the planner formally scores passes but does not compare shot, dribble,
   clear, and hold candidates in one decision;
6. the goalkeeper holds a center position or walks toward a goal-kick ball but
   lacks interception, block, recovery, and distribution;
7. formations and role behaviors use little opponent, score/time, uncertainty,
   support, marking, or passing-lane information;
8. no labeled outcome corpus yet supports evaluator fitting, opponent modeling,
   or safe self-play.

## Development principles

1. **Actions before tactics.** A planner cannot be excellent if its requested
   pass, shot, or clear cannot be physically executed.
2. **Measured envelopes.** Every action is planned only inside a server-tested
   range/direction/timing envelope with uncertainty.
3. **One skill contract.** Strategy requests target, range/arrival speed,
   receiver, action ID, and mode; motion returns lifecycle and outcome.
4. **Distributed and recoverable.** Coordination tolerates lost information,
   expires automatically, and always retains a local safe behavior.
5. **Deterministic core first.** Hard rules, fall handling, executability,
   assignment, and fallback remain interpretable before learning preferences.
6. **No reward-only promotion.** A model advances through physical and team
   metrics, never a training curve alone.
7. **Opponent diversity.** Self-play is one test source, not the definition of
   strength.

## Capability workstreams

### A. Athletic motion

- finish the phase-conditioned residual run policy;
- train and test forward speed, lateral motion, yaw, braking, stopping, command
  changes, light pushes, and recovery;
- build transition guards among stand/walk/run, approach, kick, receive, block,
  and get-up;
- fit action-specific reach time and fall probability rather than one constant
  player-speed assumption;
- retain role-aware speed limits so defenders, receiver setup, and close ball
  control do not use an unsafe sprint command.

### B. Ball skill engine

- create one target-conditioned pass/shot/clear policy instead of unrelated
  models for every distance;
- use a low-dimensional optimized kick trajectory as teacher and deterministic
  fallback, then learn residual corrections;
- add a controlled dribble controller with ball separation and reacquisition;
- implement deterministic receive/stop first, training a first-touch policy
  only when contact mechanics dominate failures;
- progress from stationary to moving ball and from guarded setup to approach
  transition;
- calibrate ball travel distributions, not only mean range.

### C. Goalkeeper

- predict the ball's intersection with the goal line and its uncertainty;
- choose hold, lateral intercept, forward smother/block, recover, clear, or pass;
- position from ball, opponent shooting angle, defenders, and reachable time;
- learn a keeper block only after walking/standing interception coverage is
  measured and found insufficient;
- distribute toward low-risk wide or central receivers using the common pass
  planner.

### D. Attack and possession

- select shoot/pass/dribble/hold through a common evaluator;
- generate support points from open passing lanes and receiver reach margins;
- use width, depth, third-player support, underlap/overlap, and penalty-area
  occupation constraints appropriate to seven players;
- coordinate striker runs with leading/through-pass windows;
- keep a counter-attack safety player and avoid sending every player toward
  the ball;
- evaluate shot quality, receiver advantage, turnover danger, and rebound
  structure rather than forward progress alone.

### E. Defense and transition

- compute opponent danger from goal angle, distance, ball access, and open
  lanes;
- assign pressure, cover, balance, marking, and lane-blocking responsibilities;
- switch immediately between possession loss/gain shapes while keeping role
  hysteresis;
- avoid double pressure and uncovered central opponents;
- clear only when possession actions are not safe;
- evaluate defensive stops, forced wide attacks, interceptions, clear quality,
  and post-clear shape.

### F. Decision learning and opponent adaptation

- record candidate features, hard rejections, chosen action, predictions, and
  physical outcome;
- fit field-evaluator weights offline with train/validation splits and
  calibration plots;
- add a contextual bandit or shallow policy only after outcome labels are
  reliable;
- introduce bounded depth-two planning for high-value cases;
- model opponent pressure, formation, preferred attack side, and restart
  tendencies from observations available to each agent;
- use multi-agent self-play only after the action interface and credit signal
  are stable, with deterministic safety projection around learned preferences.

## Capability release sequence

### R1: parameterized ball execution and motion transitions

Estimated effort: 20--40 effective engineering days plus bounded training.

Kick route:

1. optimize a low-dimensional phase/keyframe trajectory with CEM/CMA-ES;
2. validate strong upright straight contact in CPU MuJoCo and RCSSServerMJ;
3. train target-conditioned residuals for requested direction and speed/range;
4. expand to 2, 3.5, and 5 m at `-30, -15, 0, 15, 30` degrees;
5. expand to 5--10 m shot/clear actions;
6. add setup, ball, friction, mass, PD, delay, noise, and push randomization;
7. add moving-ball and walk/run-to-kick transitions.

Kick gate:

- each promoted 2--5 m central range: at least 18/20 through a one-metre
  corridor, median direction error at most 10 degrees, upright at least 95%;
- 5--8 m unopposed shots: at least 17/20 enter the goal mouth;
- clears move the ball at least 6 m away from the defended central region in
  at least 18/20 accepted trials;
- three training seeds, source/ONNX parity, no non-finite output, and safe
  fallback on every load/shape/inference failure.

Run gate:

- three seeds and at least 200 held-out episodes per seed;
- `vx=1.5 m/s` for 10 s with at least 95% upright completion;
- median achieved speed at least 1.2 m/s after 2 s;
- tracking RMSE at most 0.35 m/s and lateral drift at most 0.25 m;
- stop, lateral, yaw, abrupt command, kick transition, and get-up regressions
  all pass before any match role uses the policy.

If the full conditional kick fails, release a smaller empirically valid
envelope and keep the optimized trajectory. Honest reliable capability is
better than an unstable nominal 10 m skill.

R1 checkpoint (2026-09-01): the exact-physics teacher, residual action
contract, multi-condition dataset, BC/DAgger loop, robust parameter-table
export, and guarded Apollo C++ executor are implemented. The locked 2 m
forward-pass table made contact in 300/300 held-out trials with zero falls, but
passed the full range/direction/speed gate in only 224/300 (`74.67%`). It is an
experimental default-off integration, not a promoted team skill. Fixed-phase
CEM and the current MLP are closed at this checkpoint; the remaining R1 route
starts with adaptive ball setup/contact timing, then adds 2/3.5/5 m pass,
direction, shot/clear, moving-ball, server and three-seed gates.

Server transition checkpoint (2026-09-01): controlled 7v7 trials proved that
the same residual table node can range from no contact to 2.70 m and can change
direction error by more than 40 degrees when only the underlying walking state
changes. Fixed ball pose and a longer stable hold did not remove the variance.
The accepted continuation is therefore a versioned phase/state-conditioned
transition policy trained from randomized pre-kick states, with deterministic
setup and fallback retained. Evidence, external-method boundaries, and exact
gates are recorded in [`kick-transition-development.md`](kick-transition-development.md).

Exact transition checkpoint (2026-09-01): independent CPU optimization solves
361/368 randomized training states without falls, but observation-to-trajectory
selectors reach only about 40% on untouched states. A frozen static phase-6
release rule then passed only 3/5 released states on an independent corpus and
is closed. Sequence-level timing data confirms successful windows on 87/128
approach rollouts for one action (`67.97%`), so R1 now evaluates a compact
action bank before fitting a grouped, causal trigger. This work does not widen
the runtime envelope or change the default-off competition guard.

Closed-loop transition checkpoint (2026-09-01): four training-selected action
prototypes have 25/26 untouched oracle coverage, but learned timing/action
selectors realize at most 15/26. Direct v3 behavior cloning also fails despite
low validation loss. Exact-state DAgger improves closed-loop success from 0/92
to 10/92 and then 27/92; the second round contacts in every trial but falls in
one. R1 remains blocked at the 83/92 and zero-fall gate, and no candidate is
enabled in the competition runtime.

### R2: complete individual football actions

Estimated effort: 15--25 effective engineering days.

Deliverables:

- full pass lifecycle:
  `Proposed -> Ready -> Committed -> Executed -> Received/Intercepted/Out/Timeout`,
  with `Cancelled/Expired` from any active state;
- `Ready` requires target arrival, upright stability, ball-facing orientation,
  freshness, and first-touch feasibility;
- direct, leading, and through-pass execution;
- receive/stop, controlled dribble, turn with ball, shot, and clear;
- common action result and failure codes consumed by telemetry and planner;
- ball trajectory and reach-time distributions fitted from server traces.

Gate:

- 2v0 pass completion at least 90%;
- 2v1 pass completion at least 70%;
- unopposed receive-and-control at least 85%;
- controlled 5 m dribble corridor completion at least 80%, with no continuous
  ball-holding exploit;
- 20% intent loss and ten-cycle delay never create a permanent commitment;
- invalid or uncertain actions fall back in the same cycle.

### R3: complete goalkeeper and positional roles

Estimated effort: 15--25 effective engineering days.

Deliverables:

- goalkeeper intercept prediction, angle reduction, block, recover, and
  distribution;
- striker receive/turn/finish and timed run behaviors;
- midfield support, circulation, pressure, and transition coverage;
- defender marking, line/lane coverage, delay, interception, and clear;
- role-specific action envelopes and motion-risk budgets;
- every role has ball-near and ball-far behavior with explicit ownership.

Gate:

- goalkeeper saves at least 70% of the declared physically reachable shot
  fixture and distributes successfully at least 85%;
- striker converts at least 60% of controlled central chances inside the
  promoted shooting envelope;
- defensive fixtures reduce unopposed central shots by at least 30% versus the
  current ball-relative formation;
- role assignments remain unique and stable under small observation noise;
- no role becomes idle or blindly chases the ball for an unbounded interval.

### R4: coordinated team attack, defense, and set plays

Estimated effort: 20--35 effective engineering days.

Deliverables:

- attack, balanced, defend, transition, protect-lead, and chase-goal phases;
- opponent-aware support, width/depth, third-player options, marking, pressure,
  cover, balance, lane blocking, and counter-attack anchors;
- unified action selection with hard filters and calibrated utility;
- coordinated open play and all restart/penalty behaviors using the same skill
  library;
- bounded pass-then-shoot/pass depth-two planning;
- complete decision replay and physical outcome attribution.

Gate:

- two-pass possession chains complete in at least 70% of unopposed trials and
  45% of 3v2 trials;
- 3v2 controlled entries or effective shots improve at least 25% over the
  forward-contact baseline;
- defense reduces high-quality central chances at least 35%;
- at least 90% of team restarts reach a planned teammate or a declared safe
  target within the action envelope;
- phase/role/action changes remain bounded under noisy inputs;
- protecting a lead reduces conceded chance rate without eliminating all
  counter-attack threat; chasing a goal increases chance creation without
  uncontrolled defender collapse.

### R5: excellent-team optimization

Estimated effort: 20--35 effective engineering days.

Evaluation set:

- fixed 1v0, 2v0, 2v1, 3v2, goalkeeper, and role-transition suites;
- at least 20 paired, side-swapped matches against the accepted baseline;
- varied scripted opponent styles: deep block, high pressure, direct attack,
  wide attack, possession, and counter-attack;
- replay-derived situations and isolated black-box public opponents where
  legally and technically usable;
- score/time states including early lead, late lead, early deficit, and late
  deficit;
- perception loss, communication loss, falls, and unfamiliar formations.

Final C4 gate:

- aggregate intentional pass completion at least 60%;
- at least one controlled two-pass chain in 70% of full matches;
- effective shots average at least one per match and shot quality does not fall
  merely to increase volume;
- central chances conceded and uncontrolled turnovers both improve relative to
  the accepted baseline;
- paired evaluation materially improves at least three of possession,
  territory, controlled entries, effective shots, goals, goals conceded, and
  expected field value, without material regression in the others;
- improvements are confirmed by at least two non-self opponent styles;
- fall/recovery, finite actions, illegal behavior, and decision-time safety do
  not regress;
- no single opponent, seed, side, or monitor-initialized scenario accounts for
  the claimed improvement.

## Training and learning order

The accepted order is:

1. optimized/reproducible action reference;
2. isolated target-conditioned low-level skill training;
3. guarded runtime integration and server calibration;
4. deterministic action generation and hard safety filters;
5. labeled action outcomes and offline evaluator fitting;
6. bounded two-step planning;
7. opponent-conditioned preferences;
8. optional multi-agent self-play inside the deterministic safety projection.

Do not train an end-to-end seven-agent strategy before R2. Until actions and
outcomes are stable it would optimize an unstable credit signal and obscure
whether failures come from tactics or motor execution.

## Schedule and resource expectation

For one primary developer and the current 8 GB laptop GPU, the C4 capability
target is estimated at 90--160 effective engineering days, typically 5--8
months with training, C++ development, and evaluation overlapping. This does
not include competition infrastructure work.

R1 and R2 are the immediate capability release. R3 and R4 make the team
complete. R5 is what changes the claim from complete to excellent.

Training jobs and raw evaluation data stay under `/home/win98/rl_runs` or a
dedicated artifact directory. Formal learned candidates require three seeds,
resumable checkpoints, source/config/asset hashes, deterministic held-out
reports, and explicit stop conditions. Reward curves never substitute for
capability gates.

## Priority from the current commit

1. parameterized kick teacher/residual policy and server calibration;
2. run-policy promotion in parallel, without displacing stable walking early;
3. complete receive/dribble/pass/shot/clear lifecycle;
4. goalkeeper and role-specific capability;
5. opponent-aware coordinated attack, defense, transition, and set plays;
6. outcome-driven evaluator fitting, bounded depth two, opponent adaptation,
   and late-stage self-play;
7. opponent-diverse C4 evaluation and iterative weakness repair.

This route keeps infrastructure outside the capability objective while making
the final team substantially more than a legal or merely functional entrant.
