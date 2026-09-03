# Phase-robust kick transition development

Status: active R1 engineering record

Last updated: 2026-09-01

## Decision

Stop calibrating a fixed joint residual on top of an arbitrary walking-policy
state. Keep deterministic ball setup and the distributed pass handshake, but
replace the current open-loop transition with a phase/state-conditioned kick
policy trained from a motion prior and randomized pre-kick states.

The parameter table and C++ residual executor remain default-off experimental
infrastructure and a guarded fallback candidate. They are not a promoted team
skill until the server gates in `team-excellence-roadmap.md` pass.

## Checkpoint verification

The 2026-09-01 guarded-infrastructure checkpoint passed:

- the Apollo build and all seven native C++ tests;
- all 47 repository Python tests;
- a normal 600-cycle 7v7 run with 14/14 clean exits, connections, joins, and
  `PlayOn` observations, zero client/server failures, and zero parameterized
  kick samples;
- a forced 900-cycle fallback-pass run with 14/14 clean exits, six
  `TargetedPass` samples, a matching physical contact event, and zero
  parameterized kick samples.

The first forced-pass run correctly failed: a 0.60 s continuous setup hold and
two-degree yaw gate admitted 27 matching `Ready` observations but no contact.
Server telemetry showed the stopped walk policy oscillating around two degrees
and continually resetting the timer. The accepted fallback guard therefore
uses a 0.25 s debounce and a three-degree yaw window. This restores the
formally accepted contact path without claiming to solve gait phase; the
versioned transition policy below still owns that requirement.

Reproduce the two match gates from WSL with:

```bash
APOLLO_BINARY=runtime/apollo/build/ApolloCodeBase \
  scripts/run_apollo_acceptance_match.sh 600

APOLLO_BINARY=runtime/apollo/build/ApolloCodeBase \
  APOLLO_ENABLE_PARAMETERIZED_KICK=1 MATCH_PASS_SCENARIO=1 \
  MATCH_REQUIRE_PASS=1 \
  scripts/run_apollo_acceptance_match.sh 900
```

## Server evidence

All trials below used the same online Apollo strategy path, a full 7v7 server,
the `Proposal -> Ready -> TargetedPass` contract, condition 60, a nominal 2 m
forward target, and a single modified `strike_hip_yaw` parameter. The
calibration scene first established a post-`PlayOn` ball track and then used the
official monitor protocol to remove approach-position variance.

| Trial | Change | Contact | Forward progress | Signed direction error | Finding |
|---|---:|---:|---:|---:|---|
| s6934 | table value | yes | 1.17 m | -13.59 deg | weak baseline |
| s6933 | hip yaw -0.30 | yes | 2.70 m | -3.58 deg | one strong result before phase lock |
| s6936 | hip yaw -0.15 | no | 0.03 m | not meaningful | non-monotonic response |
| s6938 | hip yaw -0.30, fixed contact pose | no | 0.00 m | not meaningful | pose control did not make the action repeatable |
| s6940 | hip yaw -0.30, 0.60 s stable hold | yes | 0.24 m | +44.18 deg | identical table node remains phase sensitive |

The action identity, target, ball-relative release pose, and requested speed
were preserved in telemetry. The remaining uncontrolled variable is the
low-level walking state. The server runner resets observation history at kick
entry, but the physical joint pose and velocity still come from an arbitrary
point in the standing/walking limit cycle. The current CPU kick task normally
starts from one deterministic `qpos0`, so its held-out score does not cover this
deployment distribution.

Two integration defects found during these trials have already been fixed:

- pass generation now consumes the world model's TTL-validated ball position
  instead of requiring a camera observation on the exact decision cycle;
- committed proposals survive short receiver occlusion but still require the
  matching `Ready` acknowledgement before contact.

## External evidence and reuse boundary

1. FC Portugal's skill-set-primitives work reports kicks that transition from
   walking and sprinting, with the primitive responsible for smoothing the
   transition from arbitrary gait points. Reuse the unified reference/residual
   and trained-transition method, not its NAO model or GPL runtime code:
   <https://arxiv.org/abs/2312.14360> and
   <https://doi.org/10.1007/s00521-025-11151-3>.
2. Pufe's 2026 RoboCup 3D study reports that learning the walk-to-kick switch
   time can match a handcrafted baseline, while jointly learning placement and
   switching performs substantially worse. This supports retaining the
   deterministic setup controller and learning only the low-level transition:
   <https://doi.org/10.60643/urai.v2025p29>.
3. RoboNaldo uses a motion-guided curriculum: stable whole-body tracking prior,
   stationary-ball task adaptation, then moving-ball shooting through a
   locomotion-command and kick-trigger interface. Its official code is a G1
   implementation, so reuse the staged curriculum, reward separation, critical
   frame sampling, and export checks rather than the model weights:
   <https://arxiv.org/abs/2606.11092> and
   <https://github.com/OpenDriveLab/RoboNaldo>.
4. ApolloCodebase online main at commit
   `71018c968969d6e55130b0e1987cd5b4f5c3b4df` contains the T1 walk and get-up
   runners but no kick policy or kick asset. Its value here is the maintained
   server, robot, behavior-tree, ONNX, and fallback boundary; it cannot close
   the kick transition by direct code reuse:
   <https://github.com/XiangruiJiang/ApolloCodebase>.

## 2026-09-01 transition-state checkpoint

K1 and the simulation part of K2 are now implemented. `kick_policy_v3` has a
98-value actor input: the v2 fields plus an explicit locomotion phase, while
retaining the dynamic three-value support hint. The exact-CPU corpus generator
randomizes approach position, yaw, ball pose and setup hold length, captures
whole `qpos`/`qvel` transition states, and performs whole-rollout
phase-stratified splitting. Seed 7003 accepted 122/128 rollouts, has all eight
phase buckets in both splits, 97 training states, 25 validation states, 121
contacts and zero falls. The NPZ SHA-256 is
`416ca5f6447c750ca018ff83c80a8b56fa2cf87f06cfa5888da48ae373471bba`;
its manifest SHA-256 is
`9a4be1db2796c76731527c8b2822afa1e00b765a51169765f3532654f331a75a`.

The first accelerated residual-policy route was rejected by exact evidence.
Warp passes an identical-control 60-step comparison with maximum joint error
`8.85e-6 rad`, root error `1.38e-6 m` and ball-position error `6.13e-6 m`.
JAX does not: it first exceeds the joint-velocity limit at step 11 and reaches
`0.501 rad` joint, `0.064 m` root and `0.123 m` ball-position errors. Formal
training therefore requires a hashed, passing backend report and defaults to
Warp; JAX is diagnostic-only.

That open-loop control comparison was necessary but not sufficient. A
16,384-step Warp PPO run with the full gate reward changed validation fall rate
from 62.5% to 87.5% and retained zero gate successes. Its ONNX export matches
JAX CPU to `1.40e-9`, yet exact CPU replay passed 0/25 and fell in 17/25. An
independent 150-step closed-loop comparison located the reason: the CPU and
Warp targets match initially, but small contact-state differences feed back
through the walking network; target error first exceeds `0.071 rad` at cycle
15 and reaches `1.49 rad`. Applying Warp's targets to CPU, as the first parity
tool did, hides this independent-controller amplification. The policy SHA-256
is `0c82759235f6f1ed9fad00553931fbe977339bffc23fbf25bf05a7e86db5e507`
and is explicitly rejected.

The reliable route now generates phase/state-conditioned labels in exact CPU
MuJoCo before fitting a deployable selector. A single-state proof improved
condition 60 from `0.450 m` to `1.952 m` progress and passed the complete
distance, lateral, speed, contact and posture gate in 10.9 seconds. The first
eight-bucket exact-CPU table improved training success from 5/97 to 19/97 and
held-out success from 0/25 to 3/25 with zero held-out falls. It is useful
evidence, but rejected at 12% versus the 90% gate. Bucket-level errors show
that gait phase alone is not a sufficient selector: ball-local position, root
velocity and torso state must condition the next model. The exact table
manifest SHA-256 is
`1c8747abf87032a35766b598628af16c88294204abda66abb40f1d34f7eaee1c`.

Per-transition exact-CPU teacher generation is now complete for the enlarged
training corpus; the findings and the switch-window continuation are recorded
below. Accelerated PPO remains blocked until an independent closed-loop corpus
parity gate passes; more steps on the rejected seed are not authorized by the
evidence.

## Exact state labels and switch-window pivot

Seed 7901 enlarged the randomized transition corpus to 460 accepted states,
with a 368/92 whole-rollout train/validation split. A 20-by-8 exact-CPU CEM
pass plus a 32-by-12 repair pass produced successful trajectory parameters for
361/368 training states (`98.1%`) with zero training falls. This establishes
that the kick trajectory family is physically capable from almost every
sampled state. It does not establish a deployable mapping from observation to
trajectory.

That mapping is the actual blocker. Nearest-neighbour selectors plateaued at
36--37/92 on the untouched split and a cross-evaluated teacher bank did not
generalize under a phase or global lookup. A fixed phase-6 prototype passed
29/57 states in independent seed 8501. A release gate selected on two
development corpora was then frozen before seed 8901; the blind result released
only 5/41 phase-6 states and passed 3/5 (`60%`) with zero falls. The static
threshold route is closed rather than retuned on its test set. Preserving the
captured walk action history and blending toward a neutral pose were also
tested and rejected because neither improved held-out success.

The replacement dataset treats transition timing as a sequence problem. For
each randomized approach, it deterministically replays increasing consecutive
alignment periods, captures each exact `qpos`/`qvel` switch state, and evaluates
the frozen action from that state in CPU MuJoCo. Every train/validation decision
is made for a complete approach rollout, never adjacent frames. A capture-only
path stops simulation immediately after copying the transition state; a
regression test proves that this optimization is bit-identical to the previous
full rollout at the ownership boundary.

Seed 9301 contains 3,579 candidate switch frames from 128 approach rollouts.
The single fixed prototype passes 839 frames and has six fallen frames. At
least one successful window exists in 87/128 approaches (`67.97%`), split as
72/102 training and 15/26 untouched validation approaches. This independently
confirms that timing is material, while also proving that one action prototype
cannot meet the 90% release gate.

A ten-action bank was declared from earlier teacher evidence. A greedy subset
chosen on training approaches only selected prototypes 65, 107, 84 and 117.
Its oracle coverage is 98/102 on training and 25/26 (`96.15%`) on untouched
validation; all ten actions cover 26/26. This passes the physical-existence
test, but not execution: a grouped kNN selector succeeded on 10/26 validation
approaches, and the best regularized MLP released 16 times, succeeded 15 times,
and had zero falls. The MLP has useful `93.75%` release precision but only
`57.69%` total approach success, so neither selector is deployable.

The next controlled branch distils full closed-loop actions rather than open-
loop trajectory parameters. All 361 successful exact teachers were replayed
into 54,511 samples, split by whole episode into 289 training and 72 validation
episodes over every phase bucket. The first v3 BC model has ONNX/source parity
`4.32e-7` and validation MSE `5.53e-4`, but exact closed-loop physics passes
0/92 untouched states (83 contacts, zero falls). The low MSE therefore cannot
serve as a promotion metric. DAgger round one raises this to 10/92; round two
raises it to 27/92 with 92 contacts but introduces one fall. The trend justifies
another safety-constrained iteration, not runtime integration. Promotion still
requires at least 83/92 on this split, zero falls, a third frozen seed, and
exact server replay.

Immutable local evidence:

- repaired per-state teacher manifest SHA-256:
  `ce692307b73ac25eb2c8f7d0122b1a74b9135166eee87033598a55f1b7ed1b4c`;
- repaired per-state teacher NPZ SHA-256:
  `b1761811a84f75a5c69cb646ccff476d9544d1f313fd508f0633e39a814f0418`;
- blind seed-8901 fixed-gate report SHA-256:
  `63e288445367c8afea75533fd07778e8e0fd7530cd75724584fe059c693f09e0`;
- seed-9301 switch-window manifest SHA-256:
  `0acebc40b56285075a35e02ea4407dc907e085f8421b513e614e450d6227f282`;
- seed-9301 switch-window NPZ SHA-256:
  `1e0e4fdf190b61735bf88d768961d5b45d797fa70da12884a4a2276f11cf1994`.
- seed-9401 prototype-bank NPZ SHA-256:
  `afb2bcfe40dd6d8f331bd41ef5b2256736ce92117c26b4254e13d4c4fea88988`;
- seed-9601 closed-loop teacher dataset NPZ SHA-256:
  `b49387323a1fe7982dfc10b405fbca8283bc88532b0397fa4488c764723b6dea`;
- seed-10002 DAgger-round-two ONNX SHA-256:
  `b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1`;
- seed-10002 untouched exact-CPU report SHA-256:
  `978e8c5682dbc0001cf88b39dbc711cd4250d959672df9e07a866b0baf6e65c4`.

## Safety aggregation, dense timing and route closure

Failure-weighted DAgger round three assigned weight 2 to failed learner
rollouts and weight 6 to falls. It aggregated 185,365 samples, including
43,576 new learner frames. The resulting policy restored zero falls and made
contact in 92/92 untouched states, but passed only 16/92. This is below the
unsafe round-two peak of 27/92 and closes fixed-teacher DAgger as the primary
route. Its ONNX and exact report SHA-256 values are
`543e02671e0cc39034b08e4910156a64abf52668313834e0caa3e205dd8a3260`
and `fb69f70405cf0c0c897a2c964b0b464adce56358f2536c1a67b22425a58ba808`.

A bounded PPO correction was then trained above that safe behavior clone. An
aggressive 16,384-step smoke run regressed to 10/92 and introduced three
falls. A conservative 131,072-step run used a 0.02 correction scale, learning
rate `2e-5`, noise `0.01`, gate reward 100 and fall penalty 200. Every one of
its eight checkpoints was exported and evaluated in exact CPU MuJoCo. The
safe checkpoints passed between 12 and 15 states; two checkpoints introduced
one fall. None exceeded the frozen 16/92 base, so this residual-PPO branch is
also rejected. The environment and evaluator retain base-policy composition
support for future controlled experiments, but no correction is deployable.

The switch-window generator now captures all requested alignment cycles in
one setup rollout. A regression test compares cycles 1, 3 and 5 with three
independent legacy captures. A seed-9000 end-to-end replay produced the exact
same NPZ SHA-256, `7f62251d625a9c3dadcd278f56566d67796bddc1ec84491a8b77006179abe951`.
This allowed seed 10601 to scale to 512 approach rollouts and 13,752 exact
candidate states. One prototype has a successful window in 78/102 untouched
approaches. The corpus NPZ SHA-256 is
`e6b537deb86d201f54683d28f618a0714a238a730b52c58d7d501b73ad8665d8`.

All ten predeclared actions were replayed over that corpus: 137,520 exact CPU
trials. Four actions selected only on the 408 training approaches cover
383/408 (`93.87%`) there and 101/102 (`99.02%`) on validation. Thus action
capacity is not the remaining blocker. The bank NPZ SHA-256 is
`a81d5d80f633248079b0b066733d81fc815554c1a3b66301dbdbe5f7060b9cc6`.
A causal ONNX selector was trained with a disjoint 326/82 fit/calibration
split and the original 102 approaches left blind. The strongest safe all-bank
configuration passed 66/102 with no falls; its 69 releases had `95.65%`
precision. Anchor-history, grouped kNN, trajectory-level nearest-neighbour
planning and a calibrated late fallback did not improve total success. The
selector is rejected: narrow open-loop outcome windows are not sufficiently
identifiable from a single transition state.

The replacement route follows the official ICRA 2026 Booster T1 striker
pipeline rather than adding another switch classifier. The upstream source
`Daffan/humanoid-soccer` was frozen at commit
`378a12ac7446cd175f973c04e32912eb9acbee10` and is Apache-2.0. Its relevant
structure is a 20-second closed-loop task, 4,096 parallel environments,
privileged approach-and-kick teacher, directional ball-velocity reward,
50-frame student history, DAgger, then constrained adaptation. It matches this
robot's 23 actions and 50 Hz control. Isaac-Gym-specific code will not be
vendored; the task curriculum, observation ownership and teacher/student
staging will be reimplemented against the existing MuJoCo/Warp stack. The
short 3-second open-loop transition task remains an exact regression gate, not
the primary learning task.

## Long-horizon striker baseline and exact-CPU gate

The first port is now implemented as the versioned `striker_policy_v1`
boundary. Its deployable actor has 102 values and 23 bounded joint
corrections; the training-only teacher/critic view has 115 values. A
20-second task keeps Apollo walking in closed loop toward a target-relative
contact pose, runs the frozen 60-frame exact-CPU kick prior after a guarded
trigger, and permits recovery/re-approach. The actor sees current ball and
target state, approach command, kick activation/progress, locomotion phase and
support hint. It does not receive privileged base velocity or goal-relative
teacher fields.

Four implementation failures were resolved before interpreting learning:

- Brax timeout bootstrapping required an explicit `time_out` state field;
- online Welford normalization saturated features that are constant before
  contact and change afterwards, so this contract now retains bounded SI
  units and disables online observation normalization;
- the preserved kick trajectory requires Apollo's versioned
  `[0.50, -0.04, 0]` walk command for its first 33 frames; removing that
  concurrent baseline removed contact even at the nominal pose;
- attenuating small approach commands by kick activation put Apollo below its
  gait deadband and stranded the robot outside the trigger envelope.

After removing the command attenuation, a strict 7 cm trigger reached 55/64
seed-11203 Warp rollouts, contacted in all 55, succeeded in 19 and fell in
none. The nine non-triggering robots had safely converged to 8.2--10.2 cm
contact error. The deterministic controller now triggers immediately inside
the strict envelope, or after 25 consecutive frames inside an 11 cm/0.10 rad
settled envelope. On the same frozen rollouts this reaches and contacts in
64/64, succeeds in 29/64 (`45.31%`), and has zero falls. The detailed Warp
report SHA-256 is
`9d61cf3c1b56b496ad0b10f93d328c0e855e71a2223646e7ab606fef8b399b29`.

An independent double-precision CPU MuJoCo evaluator reimplements the same
geometry, confirmation state, Apollo ONNX feedback, kick-walk ownership,
prior playback, rearm and terminal rules. Seed 12203 contacts in 64/64,
succeeds in 23/64 (`35.94%`) and has zero falls. Its report SHA-256 is
`b601785d2dcb0eea58fb2cd58b6a8c35da844d80f4d2b39840af5d7b394ef3f0`.
The accelerated/exact difference and the low absolute success rate both block
promotion.

Random residual exploration without the physical prior never discovered a
contact through the retained seed-11103 checkpoints. A seed-11102 run with
online normalization appeared to contact, but inspection showed near-unit
action saturation as soon as post-contact fields left their pre-contact
zero-variance distribution; it is invalid. With the prior restored and
normalization disabled, a 32,768-step teacher checkpoint regresses to 28/64
under the accepted settled controller, versus the deterministic prior's
29/64. Its accelerated report SHA-256 is
`404cc55f444eec7361abedcb22a689d97717af95ce126e586173c03b1d2aea25`.
Later continuation also regressed, so none of these checkpoints is retained
as a candidate.

The accepted continuation is not more PPO steps on this reward. It is:

1. freeze the 64/64 contact, zero-fall controller as the measurable baseline;
2. add identical-state independent-controller CPU/Warp comparison and retain
   exact CPU as the promotion authority;
3. make kick strength/range a supervised target and train a privileged
   correction against physical terminal outcomes, not exploration-only
   contact discovery;
4. distil a 50-frame history student only after the teacher beats the
   deterministic prior on untouched exact-CPU rollouts;
5. require three exact seeds, ONNX/source parity and server replay before any
   C++ runtime integration.

## K2-A motion-guided contact and Apollo recovery baseline

K1-D was first replayed against the exact RCSS ball using PAiD's endpoint
placement rule. This corrected an important interpretation error: relative
motion-tracking promotion did not imply a complete kick. Across 91 full-corpus
motion/lead-time trials, the actor produced eight correct-foot contacts, but 86
trials fell and five started in collision. No trial passed contact plus 8 cm
progress plus upright completion. The strongest motion nevertheless moved the
ball 1.98 m, establishing useful impulse before a stable transition existed.

The failure was isolated to post-contact ownership. Blending directly to the
default joint pose at five different rates still allowed the torso to sink
below 0.35 m. Switching one control frame after correct-foot contact to the
retained Apollo walk actor with a zero command instead stabilizes motion 12.
The nominal contiguous entry window is frames 113--118; it keeps the original
K1-D motion actor before contact and uses Apollo only for dynamic recovery.

That controller composition passes the predeclared exact-CPU robustness gate.
Seeds 20260986, 20260987 and 20260988 each evaluate 120 unique cases over the
six entry frames, with joint, root-velocity, yaw, ball-radius and ball-arc
perturbations. All three produce 120/120 correct-foot contacts, 120/120 stable
screening successes, zero falls, zero wrong-foot contacts and zero non-finite
states. Median progress is 4.13, 4.18 and 4.14 m; the complete observed range is
3.66--4.77 m. Reports bind clean revision `6d361d3`, the canonical K1-D tree
hash and unique reset/ball seeds. The earlier `v1` reports are invalidated only
because their checkpoint fingerprint algorithm was noncanonical and they did
not record Git cleanliness; canonical `v2` reports reproduce the physics.

This is the retained K2-A *training baseline*, not a runtime kick. It proves a
fixed roughly 4 m contact-and-recovery envelope under small perturbations. It
does not accept requested range, direction, arrival speed, walking entry or a
moving ball. The next policy contract must preserve the K1-D motion features,
add egocentric ball/target commands, initialize new input columns without
destroying the retained actor, and learn target-conditioned contact before any
server integration. Immutable paths and hashes are in
`training/locks/paid_k2a_2026_09_02.yaml`.

## K2-B ball/target contract and lossless bootstrap

The conditioning boundary is now implemented as
`soccer_ball_motion_policy_v1`. It preserves the complete 110-value K1-D actor
prefix and appends 16 values: egocentric ball position and relative velocity,
ball-to-target direction and distance, requested launch and arrival speeds,
pass/shot/clear mode, and ball observation age/validity. Invalid perception
maps the extension to all zero, but the deployment contract still requires a
same-cycle Apollo walking/search fallback rather than executing a blind kick.

K1-D is transferred by copying every existing actor/critic parameter and
zero-initializing only the new first-layer rows. The critic mapping inserts the
new command before its old eight-value root/upright suffix, preserving the old
semantics. On clean revision `5659e62`, 4,096 deterministic CPU states produce
exactly zero actor and critic output difference. The new 126-to-23 checkpoint
tree SHA-256 is `782ae53676aaca1884d6d6867535544436b6840aaffd0415de6384da0f67bb47`.

This closes only the K2-B bootstrap gate. The retained checkpoint is an
initialization for physical outcome training, not a target-conditioned skill
and not a runtime candidate. The locked order is fixed 2 m contact/recovery,
central 2/3.5/5 m range, direction bins, arrival speed and modes, then contact
randomization, rolling ball and walking entry. Full provenance and stop rules
are in `training/locks/paid_k2b_2026_09_02.yaml`.

## Implementation route

### K1. Version the transition contract

- Add a `kick_policy_v3` contract rather than silently changing v2.
- Preserve the deployable observations: angular velocity, gravity, joint
  position/velocity, previous action, ball state, target direction/range,
  requested and arrival speed, action mode, observation validity, and kick
  progress.
- Add an explicit pre-kick support/locomotion phase or a learned trigger value.
  Never infer phase only from elapsed kick time.
- Decode around a versioned motion reference or teacher target. Do not compose
  an unbounded residual with an unrelated gait output.

### K2. Build the pre-kick state distribution

- Generate exact-CPU setup rollouts from randomized approach position, yaw,
  walk history, and gait phase.
- Capture the full `qpos`, `qvel`, walk history/action, torso state, foot
  contacts, ball-relative pose, and accepted trigger frame.
- Add representative server release snapshots to the corpus for sim-to-sim
  calibration; server observations are labels/evaluation inputs, not hidden
  privileged actor inputs.
- Split train/validation by rollout and phase bucket, never by individual
  frames from the same rollout.

### K3. Train in three stages

1. Track a stable right-foot kick reference from every accepted transition
   state, emphasizing upright posture, support-foot slip, and smooth action.
2. Adapt to stationary balls over the measured contact-pose distribution and
   target direction/range; keep motion imitation strong at the start and anneal
   only after contact reliability passes.
3. Add moving-ball, walk/run entry, pushes, latency, PD, friction, mass, ball,
   perception-age, and command randomization. Train the trigger separately or
   as a bounded high-level output.

### K4. Deploy behind deterministic guards

- Implement a finite-shape-checked `96/98 -> 23` ONNX runner and an explicit
  decoder matching the contract.
- Require valid ball state, supported target envelope, posture/upright state,
  acknowledged pass identity, and a bounded transition timeout.
- On missing asset, shape mismatch, non-finite inference, posture violation,
  or timeout, return to the accepted fixed-contact/walk/get-up path in the same
  cycle.
- Emit policy version, trigger state, phase, contact, fall, target error,
  duration, and fallback reason in telemetry.

### K5. Promotion gates

Before widening range or angle, the central 2 m action must pass:

- at least 18/20 contacts through the one-metre corridor on each of three
  independent training seeds;
- median direction error at most 10 degrees and upright completion at least
  95%;
- randomized gait-phase/start-state results within five percentage points of
  fixed-pose results;
- source/ONNX trajectory parity and finite-output fault injection;
- at least 20/20 clean 7v7 process completions and the complete pass identity
  chain on the server.

Only then expand to 3.5/5 m, angle bins, shot/clear, and moving-ball entries.

## Immediate engineering backlog

- [x] add a versioned pre-kick-state corpus and generator;
- [x] add gait/support phase to the training reset and actor contract;
- [ ] add server release-state telemetry/export;
- [x] generate and repair exact-CPU per-state teachers;
- [x] prove that a four-action switch-window bank has more than 90% untouched
      oracle coverage, and reject selectors that do not realize it;
- [x] run safety-weighted DAgger and bounded residual PPO; reject both after
      exact CPU replay fails to improve the safe 16/92 base;
- [x] scale switch timing to 512 approaches, prove 101/102 bank oracle
      coverage, and reject current-state, history, kNN and fallback selectors;
- [x] implement the long-horizon privileged approach-and-kick task, frozen
      kick prior, deterministic settled trigger and independent exact-CPU
      evaluator;
- [ ] beat the deterministic prior with a privileged teacher on untouched
      exact-CPU rollouts, then distil a 50-frame history student;
- [x] retain a three-seed exact-CPU motion-12 contact baseline with immediate
      Apollo post-contact recovery and zero falls;
- [x] add a ball/target-conditioned K2 contract and losslessly bootstrap it
      from K1-D;
- [ ] train range/direction above the retained contact-and-recovery baseline;
- [ ] add the guarded C++ kick-policy runner and same-cycle fallback tests;
- [ ] rerun the central 2 m CPU and 7v7 server gates;
- [x] update this record and the R1 checkpoint with immutable K2-B bootstrap
      artifact hashes;
