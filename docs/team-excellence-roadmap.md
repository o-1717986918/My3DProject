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

- stable stand, high-speed forward walk, lateral motion, turn, brake, and stop;
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

Accepted competition baseline: commit `793c5af`.

Active locomotion candidate checkpoint (2026-09-01): the complete phase-v2
actor is wired into Apollo as opt-in `FastWalkV2`, with all 21 body joints
owned by the actor and the two head joints retained by Apollo tracking. A
900-cycle combined 7v7 gate passes with 14/14 clean exits and exercises fast
walking, passing and parameterized kicking together. It is not promoted: the
locked 10-second CPU result has 32/32 upright trials and 1.499 m/s median speed,
but 5.452 m median lateral drift, while the server gate still spends too much
time in get-up. The stable Apollo walk remains the competition default.

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
3. the complete high-speed-walk actor passes wiring/coexistence but not lateral
   drift, server fall-rate, transition, and multi-seed promotion gates;
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

- adapt the complete phase-conditioned high-speed-walk actor to the RCSS server
  domain without weakening its speed;
- train and test forward speed, lateral motion, yaw, braking, stopping, command
  changes, light pushes, and recovery;
- build transition guards among stand/walk/fast-walk, approach, kick, receive, block,
  and get-up;
- fit action-specific reach time and fall probability rather than one constant
  player-speed assumption;
- retain role-aware speed limits so defenders, receiver setup, and close ball
  control do not use an unsafe fast-walk command.

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
7. add moving-ball and walk/fast-walk-to-kick transitions.

Kick gate:

- `18/20` for a 2--5 m central range is a screening gate only; it permits the
  condition to enter formal validation but does not establish a 90% skill;
- each promoted static 2--5 m central range: three seeds and at least 200
  untouched trials per seed, raw success at least 90%, one-sided 95% Wilson
  lower bound at least 85%, median direction error at most 10 degrees and
  upright at least 95%;
- randomized position, rolling-ball and light-disturbance envelopes: raw
  success at least 80% and one-sided 95% Wilson lower bound at least 75%;
- `17/20` for 5--8 m unopposed shots is screening only; formal promotion
  requires at least 85% over the same three-seed/200-trial protocol;
- clears move the ball at least 6 m away from the defended central region in
  at least 18/20 accepted trials;
- three training seeds, source/ONNX parity, no non-finite output, and safe
  fallback on every load/shape/inference failure.

High-speed-walk gate:

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

Dense-transition checkpoint (2026-09-01): safety-weighted DAgger restores zero
falls but regresses to 16/92, and every conservative physical-residual PPO
checkpoint remains at or below 15/92. Byte-identical single-pass capture then
scales timing evidence to 512 approaches. A four-action bank has 101/102 blind
oracle coverage, while the best causal all-bank selector reaches 66/102 with
zero falls and `95.65%` release precision. The missing capability is therefore
not another prototype or threshold: open-loop outcomes are poorly identifiable
at the switch boundary. R1 now ports the official ICRA 2026 Booster T1
long-horizon privileged-teacher/history-student curriculum to MuJoCo/Warp;
the default-off table remains only a regression fallback.

Long-horizon striker checkpoint (2026-09-01): the official T1 curriculum
structure is now represented by a versioned 20-second MuJoCo/Warp task, a
102-to-23 deployable actor contract, a 138-value privileged teacher boundary,
the preserved exact-CPU kick prior, and an independent exact CPU evaluator.
The deterministic settled controller contacts in 64/64 with zero falls, but
passes only 29/64 in Warp and 23/64 in exact CPU. A 32,768-step residual PPO
teacher is worse than the prior under the accepted controller; normalization-
induced saturation and prior-free exploration were also explicitly rejected.
R1 therefore remains open. The next gate is a target-range-conditioned
teacher that beats 23/64 on untouched exact-CPU rollouts, followed by history
distillation, three seeds, source/ONNX parity and server replay. No long-
horizon model is enabled in the competition runtime.

PAiD/source-audit correction (2026-09-01): later exact-CPU evidence supersedes
the preceding next-experiment sentence. A five-action bank covers 928/1023
rollouts by oracle (`90.71%`), but its best current-state selector realizes only
153/205 (`74.63%`) frozen-validation successes; privileged and 50-frame-history
variants reach 146/205 and 147/205. A continuous outcome regressor reaches only
128/205. On an independent frozen 256-rollout set, the fixed 5 m prior reaches
172/256 (`67.19%`), while 0.1- and 0.5-scale residual PPO candidates each reach
171/256 (`66.80%`). All have zero falls, but no learned model improves the
baseline and none is promoted.

The official PAiD release now provides thirteen G1 motions, training code and a
G1 recurrent checkpoint. Its progressive motion-tracking then perception-action
design exposes the current route's invalid assumption: a short fixed keyframe
bank on top of walking is not a stable kick-motion prior. R1 therefore stops
formal training on the current fixed-prior objective and executes four gates:

1. `K0`: licence/provenance-locked local PAiD motion audit and G1-29 to T1-23
   retargeting against exact RCSS geometry;
2. `K1`: phase-conditioned T1 whole-body motion tracking with adaptive
   motion-by-phase failure sampling and safety termination;
3. `K2`: resume that skill with egocentric ball/target input, rolling-ball
   starts, correct-foot contact, post-contact stability and commanded
   direction/arrival-speed rewards;
4. `K3`: RCSS-calibrated physics/noise randomization, Warp/CPU/ONNX parity,
   three seeds and server replay.

PAiD is CC BY-NC 4.0 and its released 29-DoF/160-input G1 ONNX is incompatible
with the T1 contract. Assets remain local, attributed and non-commercial; no
source, motion or weight is vendored. `wbc_fsm` contributes only lifecycle,
model-contract, projected-gravity termination and fallback patterns to the
existing Apollo executor. The full evidence and stop rules are in
[`paid-wbc-fsm-audit-2026-09-01.md`](paid-wbc-fsm-audit-2026-09-01.md).

K0 completion checkpoint (2026-09-01): all thirteen pinned PAiD motions pass
the new local-only T1 soccer-reference schema and exact RCSS kinematic gate by
both methods. The preserved semantic A baseline passes 13/13. The calibrated
GMR body-IK B candidate also passes 13/13, reduces aggregate joint-limit clips
from 1,987 to 157, reduces maximum correction from 0.303 to 0.10 rad, retains
about 4.99 m/s mean labeled-foot peak speed, has zero non-foot pitch contacts
and preserves a minimum 1.45 kick-foot/other-foot peak-speed ratio. B becomes
the K1 primary reference; A remains the required ablation and fallback. K0
proves source integrity and kinematic feasibility only. Dynamic tracking,
ball contact and target outcome remain open. The machine-readable evidence is
`training/locks/paid_k0_2026_09_01.yaml`; no external or derived motion asset is
committed.

K1 implementation checkpoint (2026-09-01): the non-periodic 110-input/23-output
multi-motion contract, padded hash-bound corpus loader, Apollo-gain MJX/Warp
environment, failure-phase curriculum, resumable PPO trainer, fixed-seed Warp
evaluator and exact-CPU fixed-grid evaluator are complete. Dynamic evidence
corrects K0's provisional choice: semantic A has better equal-protocol tracking
and contact than body-IK B, so A and B are now parallel K1 candidates with A
first. Neither open loop completes any of 13 full clips. The best retained
residual checkpoint improves semantic exact-CPU phase completion from 31/104
to 35/104, with four paired improvements and zero regressions, but its one-sided
exact McNemar result is `p=0.0625`; it is not promoted. GMR transfer improves
31/104 to 34/104 with four improvements and one regression (`p=0.1875`). More
reward-only PPO steps on this branch are stopped. The next K1 experiment is a
phase-level optimized correction teacher, behavior cloning, then PPO resume
only if the teacher first improves the same exact-CPU grid. Evidence is locked
in `training/locks/paid_k1_2026_09_01.yaml`.

K1-A teacher checkpoint (2026-09-01): a clean-revision exact-CPU phase teacher
now optimizes a bounded 40-parameter correction on training phases and evaluates
untouched validation phases. On the weakest retained motion,
`football_stylized-001`, the formal 64-by-8 CEM run improves training survival
from 0.427 to 0.541 with four improvements and zero regressions. Held-out
survival improves from 0.503 to 0.553; completion remains 1/4 with no completion
regression. The predeclared K1-A teacher gate passes, supporting the searched
teacher -> behavior cloning -> DAgger route. It remains non-promotable because
one motion cannot train or validate a thirteen-motion deployable actor. Live
TensorBoard scalars and exact MuJoCo `qpos` replay are now emitted without
rendering in the accelerated loop. Evidence is locked in
`training/locks/paid_k1a_2026_09_01.yaml`; next is full-corpus teacher generation,
then BC and closed-loop DAgger before any PPO resume.

K1-B full-corpus checkpoint (2026-09-01): all thirteen selected teachers now
form a 6,834-frame hash-bound corpus. A PPO-compatible behavior clone passes a
new 388-trial blind exact-CPU grid, improving completion from 111 to 117 with
seven paired improvements, one regression and one-sided exact McNemar
`p=0.0352`; mean survival improves from 0.5410 to 0.5638. It is the retained
K1 actor, but remains outside the competition runtime because the finite
motions do not yet constitute a parameterized ball skill or pass three seeds.

Closed-loop DAgger exposed and corrected a teacher-definition defect: the
fixed phase correction must be added to the original teacher-base actor, not
to the already distilled student. All v1 aggregate/candidate evidence using
the double correction is invalidated. Corrected beta-zero collection labels
20,682 student-state frames. Conservative output-head-only retraining raises
mean survival from 0.5517 to 0.5704 on a separate 666-trial grid, but completion
changes only 183 to 184 (three improvements, two regressions, `p=0.5`). It is
not promoted and PPO remains gated. The next teacher must use exact-CPU
short-horizon state feedback on observed failure windows; more fixed-phase BC
or PPO steps are not justified. Evidence is locked in
`training/locks/paid_k1b_2026_09_01.yaml`.

K1-B state-feedback checkpoint (2026-09-01): the exact-CPU teacher now queries
bounded student-relative actions over a two-frame horizon and retains only
labels whose local cost improvement is at least `0.001`. It selected 12,717
student-state labels. A conservative 1,000-step output-head clone was first
selected on a 679-trial tuning grid, then evaluated once on the predeclared
untouched 777-trial grid that excludes both the state-feedback dataset and the
tuning report. Completion improves from 256 to 266 (13 candidate-only, three
baseline-only, one-sided exact McNemar `p=0.01064`) and mean survival improves
by `0.02134` with 367 improvements versus 70 regressions (exact sign
`p=5.73e-50`). Tracking tolerances pass. This closes the predeclared gate for
PPO initialization and makes the state-feedback clone the retained K1 actor.
It does not authorize Apollo runtime deployment: three-seed PPO, ball-contact,
direction/arrival-speed and RCSSServer replay gates remain open.

Formal PPO resume is now evidence-bound and fail-closed. The trainer requires
the passing paired comparison alongside its restored checkpoint, verifies the
candidate report and checkpoint identity, records all hashes, requires a clean
Git tree and an external new run directory, and emits continuous TensorBoard
metrics. A separate exact-CPU checkpoint viewer shows the actual closed-loop
policy at declared evaluation boundaries without slowing vectorized training.

K1-C first PPO diagnostic (2026-09-01): the aggressive v3 optimizer does not
survive a new 634-trial grid excluded from all state-feedback training and
selection phases. Its first checkpoint changes completion from 174 to 180 but
reduces mean survival and has 179 survival regressions versus 131 improvements;
the second reaches only 177 and has 173 regressions versus 159 improvements.
Neither gate passes, so the state-feedback clone remains retained. A v4 resume
is predeclared with one PPO pass, one tenth the learning rate and a tighter KL
region. Exact-CPU selection now supports deterministic joint/root-velocity/yaw
perturbation seeds included in the paired key, preserving untouched grids after
the finite phase-start set has been used repeatedly. Protocol and hashes are in
`training/locks/paid_k1c_2026_09_01.yaml`.

The predeclared v4 selection run passes: on 518 perturbed exact-CPU pairs it
changes completion from 177 to 186 (12 improvements, three regressions,
`p=0.01758`) and raises survival by 0.01031 with 161 improvements versus 97
regressions. The result selects the conservative optimizer protocol only. Two
additional locked training seeds, a common disjoint final perturbation grid,
a median-seed rule and a second confirmation perturbation grid are required
before the checkpoint can advance to ball-contact work.

The three-seed family does not pass: all seeds avoid net completion loss, but
the median delta is 0.00965 rather than the locked 0.01 and only one of three
passes paired promotion. The confirmation seed remains untouched. An equal
average of the three aligned one-step parameter deltas is now the bounded
variance-reduction experiment, with separate selection and confirmation
perturbation seeds. Failure leaves the state-feedback clone unchanged.

The equal average also fails its selection gate: 178 to 184 completions with
eight improvements and two regressions gives `p=0.05469`; mean survival gains
only 0.00442. Its confirmation seed is left untouched. K1-C closes with no PPO
promotion, and further reward-only continuation is stopped. K1-D returns to
longer-horizon exact-CPU state feedback under reset perturbations and admits
only cross-fitted, locally advantageous student-state labels before any new
policy evaluation.

K1-D is now implementation-complete and predeclared in
`training/locks/paid_k1d_2026_09_02.yaml`. The shared reset generator derives a
signed-64-bit case seed from motion/start coordinates, so collection and blind
evaluation use identical perturbation semantics with hashable provenance. The
teacher searches four frames from the actual student state, then independently
perturbs a copied state and requires the chosen action to improve a six-frame
validation rollout as well. DAgger may reuse a nominal phase only under a
non-zero disjoint reset seed; whole episodes, rather than frames, are assigned
to the five-fold validation split. A prior DAgger aggregate is accepted only
when its completed manifest binds the exact dataset hash. Selection and a
conditional confirmation use untouched perturbation seeds; neither authorizes
runtime deployment or ball work by itself.

K1-D closes successfully on 2026-09-02. Formal beta-zero collection covers
416 perturbed student episodes and admits 15,002 labels; the 34,553-frame
aggregate contains 11,626 new training labels and 3,376 new validation labels
with zero episode split leaks. The output-head clone passes two disjoint,
full-phase 3,076-pair exact-CPU grids. Selection improves completion from 939
to 972 with 34 candidate-only versus one baseline-only completion
(`p=1.05e-9`) and raises mean survival by 0.01124. Confirmation improves 939
to 973 with 35 versus one (`p=5.38e-10`) and raises survival by 0.01153. Both
tracking gates pass and no motion has a net completion loss. The K1-D clone is
therefore the retained training actor; the prior K1-B clone remains a fallback.
This advances work to R1 ball contact and motion transition only. It still does
not authorize Apollo runtime deployment, reward-only PPO, or any claim of a
complete football skill.

K2-A contact/recovery checkpoint (2026-09-02): exact RCSS-ball replay shows why
K1-D alone is not a kick: the full 13-motion/lead-time screen produces useful
impulse but zero stable trials. A one-frame handoff after correct-foot contact
to Apollo's retained zero-command walk controller changes the outcome. Motion
12 frames 113--118 pass three predeclared 120-trial perturbation seeds with
360/360 correct-foot contacts, 360/360 stable screening completions, zero falls,
zero wrong-foot contacts and a 3.66--4.77 m progress range. This becomes the
fixed K2-A training baseline only. It is not target-conditioned, does not cover
approach/moving-ball entry and remains outside the runtime. Next is a versioned
ball/target-conditioned contract initialized from K1-D, followed by controlled
2/3.5/5 m direction and arrival-speed training. Evidence is locked in
`training/locks/paid_k2a_2026_09_02.yaml`.

K2-B bootstrap checkpoint (2026-09-02): the deployable observation contract
now appends 16 ball/target command values to the unchanged 110-value K1-D
prefix, producing a 126-to-23 actor and a 134-value critic boundary. A formal
zero-row checkpoint transfer at clean revision `5659e62` preserves all K1-D
parameters and yields exactly zero actor/critic output difference on 4,096 CPU
states. The resulting checkpoint tree SHA-256 is
`782ae53676aaca1884d6d6867535544436b6840aaffd0415de6384da0f67bb47`.
It is retained only as the initialization for K2-B physical-outcome training;
it has learned no range or direction response and has no Apollo runtime status.
The next gate is fixed 2 m contact/recovery followed by central 2/3.5/5 m
training, with the K2-A composition kept as an independent fallback. Evidence
and stop rules are locked in `training/locks/paid_k2b_2026_09_02.yaml`.

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
- striker receive/turn/finish and timed off-ball movement behaviors;
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
2. high-speed-walk adaptation and promotion in parallel, without displacing
   stable walking early;
3. complete receive/dribble/pass/shot/clear lifecycle;
4. goalkeeper and role-specific capability;
5. opponent-aware coordinated attack, defense, transition, and set plays;
6. outcome-driven evaluator fitting, bounded depth two, opponent adaptation,
   and late-stage self-play;
7. opponent-diverse C4 evaluation and iterative weakness repair.

This route keeps infrastructure outside the capability objective while making
the final team substantially more than a legal or merely functional entrant.
