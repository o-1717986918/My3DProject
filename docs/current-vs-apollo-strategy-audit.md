# Current team versus pristine Apollo: strategy and action audit

Status: superiority not established; repeated comparison and action work active

Audit date: 2026-09-06

Compared upstream revision: `71018c968969d6e55130b0e1987cd5b4f5c3b4df`

## 1. Bottom line

The developed strategy is structurally safer and more expressive than pristine
Apollo, but repeated full matches do **not** yet show it to be stronger. The
exact procedural path now executes end to end: in
`apollo-vs-base-contract-sync-s20261235-v1` a natural dribble request entered
`ProceduralKickExecute` and completed. Its observed post-visibility ball
displacement was only about 0.15 m, however, and the match still ended 0:1.

The subsequent goalkeeper/possession run
`apollo-vs-base-gk-occlusion-s20261237-v1` also ended 0:1. It improved visible
opponent-half occupancy to 41.5% and narrowed known possession to 4771:5593,
but conceded to a fast direct shot after the goalkeeper declared the full
goal-line crossing unreachable and held its current pose. The response is to
repair concrete decision-to-motion and goalkeeper branches, while retaining
the deterministic kick bank and forward contact as explicit fallbacks. Tactical
value still requires matched-seed, side-swapped evidence rather than source
complexity or one favourable score.

The latest execution audit found a separate offensive regression: an ordinary
AP push with no selected cooperative action still inherited the precision
controller's 0.34 m contact target and centimetre-scale settle loop. Across
recent full matches this produced roughly 8,000--10,500 sampled `forward`
approach states but only a handful of actual fallback contacts. The ordinary
path is now separated from exact ball actions and restores Apollo's permissive
0.60 m setup, 0.25 m latch tolerance, 0.40 m lateral envelope and continuous
walk one metre through the ball. Two post-fix comparisons both drew 0:0 and
kept the ball predominantly in the opponent half. That is a repeatable recovery
from the preceding loss pattern, but two draws still do not establish
superiority.

## 2. Evidence from the retained match

Run directory:

`/home/win98/rl_runs/apollo-vs-base-web-match-20260905-175550`

The server log is authoritative for score and referee events.  The developed
side was left/blue and pristine Apollo was right/red.

| Observation | Developed team | Pristine Apollo | Interpretation |
| --- | ---: | ---: | --- |
| Final score | 0 | 1 | A real loss, but one match is not a strategy ranking |
| Illegal-defense penalties | 0 | 31 | Current restart/defensive legality is materially better in this run |
| Exact kick motion samples | 0 | unavailable | Current exact release was starved |
| Kick setup samples | 291 | unavailable | Decisions reached setup but did not become contact |
| Strategy samples | 89 dribble, 22 pass, 16 clear | unavailable | Rich intent existed; execution did not realize it |
| Get-up status samples | 25 across six players | unavailable | Current fast motion still imposed recovery cost |
| Risk mode | 2205 Balanced | unavailable | The old late-match threshold never became active |

Pristine Apollo does not emit equivalent decision telemetry.  Its internal
intent must not be invented from the score.  The only safe behavioral inference
is from audited source: one active player approaches and walks through the ball
toward goal, off-ball players return to role positions, and the goalkeeper holds
or approaches a goal kick.  That simplicity has low coordination value but also
few opportunities for release-state deadlock.

Reproduce the retained summary with:

```bash
scripts/analyze_apollo_vs_base_match.py \
  /home/win98/rl_runs/apollo-vs-base-web-match-20260905-175550
```

## 3. Logic-by-logic audit

| Area | Pristine Apollo | Developed implementation | Audit verdict |
| --- | --- | --- | --- |
| Role assignment | Dynamic seven-role formation | Freshness filtering, stable owner, full-team plan revision | Better contract; performance still needs A/B evidence |
| Open-play attack | AP always pushes toward goal | Capability-gated hold/move/dribble/shoot/clear/pass evaluator | Better action vocabulary; utility weights are not empirically calibrated |
| Off-ball attack | Formation position only | Unique support, outlet, receive and unmark targets with spacing and offside constraints | Plausible 2D-to-3D migration; not yet proven to create more possession or shots |
| Defense | Formation plus limited clipping | One pressure owner, reachable intercept, paired marking, lane block and cover | Better assignment consistency; generic reach-time parameters remain approximate |
| Goalkeeper | Hold and goal-kick approach | Hold, reachable goal-line intercept, smother, rebound handling and bounded clear | Better state coverage; no learned dive or first-contact model |
| Restarts | Special kickoff/goal-kick branches | Revisioned taker/receiver plan, legality, feedback, release detection and lockout | Better rule handling; one-match 0 versus 31 illegal-defense evidence supports it |
| Pass protocol | None | Proposed through physical terminal outcome with identity and expiry checks | Better observability; physical pass remains narrow and experimental |
| Failure behavior | Keep walking through the ball | Previously exact setup could wait indefinitely | Regression fixed by explicit timeout fallback and local-action retry |
| Score/time risk | None | Balanced, ProtectLead, ChaseGoal | Code path exists; old 300 s late threshold was dead in a 300 s match and is now 240 s |
| Motion use | Stable Apollo walk | Stable walk plus bounded FastWalk and mirrored RapidTurn specialists | Faster in-domain, but current match still showed nontrivial fall/recovery cost |

## 4. Regressions found and corrected

### 4.1 Contact-release starvation

The exact setup required torso speed below `0.20 m/s`, centimetre-scale ball
placement, and a debounce while the walk still carried the torso through the
slot.  The retained match produced setup phases but zero exact-kick samples.

The release speed is now `0.50 m/s`, the procedural debounce is two 50 Hz
cycles, and a wider pre-settle corridor commands neutral before the body crosses
the release slot.  These changes relax an unrealistic transition boundary; they
do not remove the ball-distance, lateral, heading, posture, or joint guards.
The short-dribble dispatch angle is now `6 degrees` instead of `1 degree`:
at its 0.55 m target this is about 5.8 cm of lateral geometry error, a bounded
trade-off reserved for ball-carry touches. A retained natural release was
previously rejected after only a 1.79-degree localization-yaw change between
decision alignment and motion dispatch. Shot and clear retain their narrower
one-degree contracts.

Procedural actions and coordinated passes now share a progress-aware fallback:
the broad contact corridor becomes eligible after 1.20 seconds only when setup
has made no meaningful progress for 0.50 seconds, and a hard 1.80-second bound
prevents indefinite dithering. Non-procedural local contacts retain the fast
0.45-second path. An exact procedural release slot always wins over the
timeout. Telemetry names the fallback explicitly; an ordinary unsupported
targeted request still returns `RejectedTargetedKickHold`. The fallback is
therefore a declared fail-soft action, not a silent claim that a fixed contact
is precise.

### 4.2 Terminal pass freeze and permanent retry delay

A failed pass used to keep the AP neutral while its terminal message was
broadcast.  The first correction allowed local dribble/shot/clear selection,
but exposed a second bug: the retry deadline was extended on every terminal
tick, making a new pass impossible.  The delay is now armed once per sequence;
the terminal outcome remains communicable while a different local action runs.

### 4.3 Dead late-match risk mode

The default late threshold was 300 seconds in a five-minute match, so every
sample remained `Balanced` until `GameOver`.  The default is now 240 seconds,
leaving a real final-minute window while remaining configurable.

### 4.4 Retained comparison sweep

The next seven full developed-versus-pristine runs scored `0:1`, `0:0`,
`0:1`, `0:1`, `0:2`, `0:1`, and `0:1`. They are retained under
`/home/win98/rl_runs/apollo-vs-base-*`, including the no-pass,
parameterized-off, secondary-pressure and goalkeeper ablations. The draw after
adding stale-ball goalkeeper hold is the best recent result; no run proves that
the developed team is stronger yet.

The ablations do reject several tempting explanations. Disabling pass did not
restore attack, adding a second presser worsened falls without producing a
goal, and disabling parameterized actions moved the ball farther upfield but
conceded twice. Across parameterized runs the ball usually failed to cross
midfield. The dominant bottleneck is therefore continuous chase/contact tempo
plus locomotion stability, followed by goalkeeper intervention timing; it is
not justified to tune more pass utility weights before those low-level effects
improve.

### 4.5 Goalkeeper intervention timing

The retained run with suffix `s20261224-v1` reached the intended last-line
body-block branch at
cycle 11400, with the goalkeeper about 0.40 m from the ball, but conceded in the
next 20 cycles. Earlier samples show the actual regression: the ball stayed
near `x=-25.8 m, y=2.2 m` while the goalkeeper held the correct angular cover
point, and `GoalkeeperSmother` did not arm until cycle 11380. The emergency
logic now permits an earlier ETA-gated near-post challenge within the final
3.5 metres, while the central opponent-first race rule and legal goalkeeper
area clamp remain unchanged. A regression test uses the observed match
geometry.

### 4.6 Latest comparison and release-pipeline finding

`apollo-vs-base-gk-bodyblock-s20261224-v1` still lost `0:1`. The last-line
body block did execute at about 0.40 m from the ball, but only one sampled
interval before the goal; this is why the near-post challenge now arms earlier
instead of widening every goalkeeper chase. The run also exposed a telemetry
attribution error: explicit fallback contacts were labelled `KickForward` even
though setup logs recorded 24 fallback releases. Motion provenance is now
preserved as `FallbackKick*`.

A separate 2 m release audit found that the training evaluator silently forced
a perfect `1.0` activation threshold while the declared stage trained at
`0.8`. After removing that hidden override and adding translation braking with
continued yaw alignment, the deterministic prior reaches trigger/contact in
125/128 held-out rollouts. Task success is still only 24/128 with two falls,
and the learned residual regresses to 21/128. This narrows the failure from
“cannot release” to “contact outcome is inaccurate”: it justifies retaining
the deterministic action as fallback, but not declaring either version a
reliable pass.

The first post-fix comparison,
`apollo-vs-base-nearpost-telemetry-s20261229-v1`, finished `0:0`. Current had
zero illegal-defense penalties versus one for pristine Apollo. Corrected
telemetry recorded 90 sampled fallback states (`49 Forward`, `16 Hold`,
`25 Stabilize`) and 32 setup transitions into explicit fallback, but still no
exact procedural kick sample. The goalkeeper remained in `GoalkeeperHold`
throughout because the match never entered the near-post test geometry, so the
new challenge branch is covered by regression test but not yet by this server
run. This draw validates attribution and legality, not superiority or exact
kick readiness.

### 4.7 Shared release contract and natural server execution

`apollo-vs-base-latched-transition-s20261234-v1` produced a genuine decision
release but no procedural motion. The live release pose was approximately
`ball_local_x=0.356 m`: it satisfied the decision layer's former
`0.335 +/- 0.035 m` contract but violated the runner YAML's
`0.320 +/- 0.020 m` anchor. The two layers now use one shared dribble pose
contract from `kick_contract.h`, and runner tests cover both sides of the
boundary. Sparse `MY3D_EXECUTION_EVENT` records every kick request and terminal
result so a one-cycle rejection can no longer disappear between periodic
status samples.

With that correction, `apollo-vs-base-contract-sync-s20261235-v1` recorded the
first natural full-match procedural start and completion. It still lost 0:1,
and the ball moved only about 0.15 m after visibility recovered. This proves
dispatch integration, not dribble quality or promotion of a learned kick.

### 4.8 Goalkeeper continuity and direct-shot fallback

`apollo-vs-base-nearpost-continuity-s20261236-v1` lost 0:1 after the goalkeeper
closed to about 0.3 m, its torso occluded the ball, and the global 0.75-second
freshness rule returned it to `GoalkeeperHold`. The local assigned goalkeeper
now retains its 3.5-second near-contact track during open play; field players
still fall back to formation under the same globally stale observation.

The next run, `apollo-vs-base-gk-occlusion-s20261237-v1`, confirmed that the
old occlusion failure did not recur, but exposed a different fast-shot branch.
At cycle 4975 the ball was at `(-23.685,-1.246)` travelling about
`(-3.685,+0.509) m/s`; the keeper at `(-26.242,-0.270)` could not reach the full
goal-line intersection. Holding the pose left the trajectory roughly 0.6 m to
its side. For mostly longitudinal goal-bound shots, the fallback now targets
the closest legal point on the current ball-to-line segment. Steep cross-goal
trajectories retain the body-hold fallback because turning after them can remove
a useful central block. The same-seed
`apollo-vs-base-gk-best-effort-s20261237-v1` run again lost 0:1 and showed that
the server match is not strictly deterministic: possession, falls and field
progress differed despite the same configured seed. The new intercept branch
ran for 15 status samples, but the eventual goal occurred later, after a
smother had brought the ball just behind the keeper's torso plane. Duty then
returned to Hold, the generic turn-first retreat pulled the keeper away and it
fell. A final-metre guard now holds the body whenever a goal-mouth ball is
within 0.65 m and no more than 0.10 m ahead of the torso plane, regardless of
one-cycle Smother/Hold classification. Neither change is credited as a match
improvement until a multi-run comparison supports it.

`apollo-vs-base-gk-lastline-guard-s20261238-v1` finished 0:2. The new body
guard did not cause either goal, but it also could not help: the first approach
had a correct smother target about 0.73 m away while the composed turn/walk
drifted in the opposite global direction, and the second shot found the keeper
about 2.3 m away laterally. These traces set a useful boundary on further
strategy tuning. A stable lateral step/body-block primitive, followed by a
separately evaluated dive, is now a real goalkeeper capability requirement;
more smother depth constants cannot substitute for it.

### 4.9 Pressure-push separation and first repeated recovery

The tactics-off comparison
`apollo-vs-base-tactics-off-s20261239-v1` finished 0:2. Known-possession samples
were 3079:7308 and the visible-ball opponent-half fraction was 22.37%. Because
the server stack is stochastic, this single ablation does not prove every
`TeamTactics` duty is useful, but it gives no support to deleting the whole
team layer. Its strongest diagnostic was instead 10,494 `forward approach`
samples and only 16 fallback-contact transitions.

Source comparison then found that the developed generic AP path had ceased to
be behaviorally equivalent to pristine Apollo. Even after specialist admission
failed, it attempted the exact action's 0.34 m setup, three-degree heading and
settle state. Pristine Apollo uses a broad 0.60 m setup point, latches within
0.25 m, accepts 0.40 m lateral displacement and continuously walks to a point
one metre beyond the ball. The runtime now has two explicit paths:

- an ordinary/pressured continuous `WalkCommand` push using that broad Apollo
  envelope; and
- an explicitly selected Dribble/Pass/Shoot/Clear action using the narrow
  procedural contract, transition checks and bounded named fallback.

`apollo-vs-base-pressure-push-s20261240-v1` finished 0:0. Known possession was
5259:5533; visible-ball median x was +5.52 m and 77.02% of visible cycles were
in the opponent half. The independent follow-up
`apollo-vs-base-pressure-push-s20261241-v1` also finished 0:0, with known
possession 5206:6403, visible-ball median x +14.91 m and 92.09% opponent-half
occupancy. The ball reached the opponent goal line, but near the corner rather
than through the goal. Exact kick samples remained zero in both runs, and 21
then 17 independent GetUp episodes were still observed, mostly after ordinary
Walk. The result is therefore a repeatable territory/defensive recovery and a
clear final-third/action-training target, not a claim that the team is already
better than base.

### 4.10 Hold execution, goalkeeper clear admission and first win

`apollo-vs-base-restart-control-s20261242-v12` produced an attributable
`FallbackKickForward/ForwardContact` from the restart taker at cycle 253 and
then entered `TakerLockout`. This closes one restart execution loop, but its
failure in the immediately preceding run shows that setup remains gait-phase
sensitive.

The natural comparison `apollo-vs-base-restartfix-s20261242-v13` lost `0:1`
and exposed two code-level stalls. `ActionPlanner`'s Hold reference candidate
was executed as Neutral by the only pressure player, even when an opponent
could reach the ball in 0.25 seconds. Hold remains visible in telemetry, but
execution now falls through to continuous pressure. The goalkeeper also spent
more than three seconds alternating side relocation and precision turning
around a cached ball; strong Clear setup is now admitted only from a fresh
track and a coarse behind-ball/heading corridor.

`apollo-vs-base-nohold-s20261242-v14` drew `0:0`. Twelve of thirteen Hold
samples executed as Walk rather than Neutral, and independent GetUp episodes
fell from 20 to nine. Territory was poor (1.69% visible opponent-half cycles),
so this confirms the motor-side Hold correction but not superiority.

`apollo-vs-base-dribblerelease-s20261242-v15` won `1:0`, with zero current-team
illegal-defense penalties and 33.81% visible opponent-half cycles. This is the
first retained win against pristine Apollo, not a stable advantage: the three
latest natural comparisons are one loss, one draw and one win. The run still
emitted no exact procedural kick. Its `Dribble->Neutral` samples show that most
attempts failed the shared dynamic transition contract rather than only the
final pose timer. Dribble-only pose confirmation is now one 20 ms decision
cycle; static Shot/Clear keeps 40 ms and generic fallback contact keeps 250 ms.

The winning sequence also exposed a final-third geometry error. With the ball
near `(27.08, 0.67)`, centre-only pressure aim asked a forward-facing robot for
an unnecessary turn of roughly 58 degrees. Inside the final six metres,
continuous pressure now chooses among safe `y=-1,0,+1 m` goal-mouth points by
minimum current-body turn. Outside that region, goal-centre aim remains the
default. This change is regression-tested; a natural comparison must still
establish whether it improves conversion.

`apollo-vs-base-goalmouth-s20261242-v16` drew `0:0`. The ball reached only
`x=5.41 m`, so the final-six-metre aim branch was not exercised. Current again
had zero illegal-defense penalties versus one for pristine Apollo, but logged
21 independent GetUp episodes and no exact kick. Across v13--v16 the honest
score record is therefore one loss, two draws and one win; action stability,
not another unmeasured tactical weight, remains the leading performance gap.

### 4.11 Tactics ablation, corrected territory evidence and finishing control

`apollo-vs-base-tactics-off-s20261242-v17` lost `0:1`; the two retained
tactics-off runs are now losses of `0:2` and `0:1`. The corresponding
tactics-on `apollo-vs-base-tactics-on-s20261242-v18` drew `0:0`. This supports
retaining the coordinated duty layer, but does not isolate support, marking,
or goalkeeper logic individually because the switch removes all of them.

The old analyzer then understated v18. It grouped staggered clients by local
cycle and counted only direct visual ball observations. Direct vision ended
near server time 80 s, while fresh/near-contact estimates continued to follow
the same attack. Schema v3 now groups by 100 ms server-time buckets and reports
visible, fresh (visible or at most 0.75 s old), and bounded near-contact tracks
separately. Under the fresh definition, v18 had median ball x `+17.92 m`,
`60.99%` opponent-half occupancy, and maximum x `+27.30 m`: it was a wide
goal-line attack that failed to convert, not a match trapped in the own half.

The v18 terminal sequence carried the ball around `(22.3,-8.3)` toward the
goal line, repeatedly interrupting continuous pressure with uncompleted exact
Dribble/Pass setup before the ball went out wide. Open play now uses an
explicit finishing cut-in whenever the ball is inside the final 8 m but more
than 1 m outside the post. It aims only 1.5 m farther forward and toward
`y=+/-1 m`; experimental short-touch/pass setup cannot interrupt that urgent
carry. Once inside the goal corridor, the existing low-turn goal-mouth aim and
exact Shot selection resume.

`apollo-vs-base-cutin-s20261242-v19` drew `0:0`, with zero developed-team
illegal-defense penalties, fresh median x `+17.61 m`, `66.78%` opponent-half
occupancy, maximum x `+27.49 m`, and seven independent get-up episodes. The
important new evidence is 60 sampled Shoot selections after the cut-in, but
zero releases. The first live shot exposed a shared controller bug: exact
coarse relocation reused the formation navigator, whose 0.30 m stop radius
declared the canonical stance reached while the strong-kick release slot was
only 1--2 cm wide. Exact actions now use a dedicated turn-first forward crawl
until lateral error enters the fine controller. A regression reproduces the
v19 pose.

`apollo-vs-base-precision-relocate-s20261242-v20` also drew `0:0`. Its random
trajectory stayed entirely in the developed half, so it did not naturally
repeat a Shot. A strong Clear nevertheless progressed from coarse relocation
through `precision-position` to `pre-settle`, confirming that the former
0.30 m dead stop was crossed. The run exposed a second shared composition bug
during OurGoalKick: with a mostly lateral ball, the far approach overwrote its
travel heading with the eventual kick heading and asked the forward-dominant
walk to strafe for 15 seconds. Long-range exact-action approach now faces the
actual waypoint; only the near-field controller restores final contact yaw.
The observed v20 goalkeeper pose is retained as a restart regression.

`apollo-vs-base-action-chain-s20261242-v21` drew `0:0`. Goal-kick approach
samples fell from 751 to 285 and the restart reached later precision phases;
the developed side completed one explicit Clear fallback and one DribbleTouch
fallback, with zero illegal-defense penalties versus two for pristine Apollo.
Fresh opponent-half occupancy was `43.0%` and maximum x `+11.50 m`; twelve
independent get-up episodes and zero exact releases remain material blockers.
This is evidence that the composition fixes shorten stalls and preserve
fallback execution, not evidence of match superiority.

The learned-transition integration hypothesis was also checked against the
actual call graph. Fixed-2 m TargetedPass never used the procedural
leg-velocity/tilt gate, so that gate was not the claimed blocker. It did,
however, inherit a 250 ms neutral pose dwell that removes most of the gait
phase represented in the transition corpus. Fixed-2 m residual/learned entry
now has its own one-cycle pose confirmation while retaining its measured
0.50 m/s and ball/yaw limits. Static range pass, Shot, Clear, and procedural
Dribble retain the stricter static-trajectory transition contract.

### 4.12 Static-release integration and danger-memory evidence

The first retained 4 m shot replay,
`procedural-shot-post-relocate-s20261242-v1`, selected Shoot and reached the
centimetre-scale contact slot, but then switched directly from a
`0.28--0.62 m/s` gait into the Neutral keyframe. The torso continued through
the ball and fell; no kick command was emitted. A zero-demand Walk command now
owns the high-speed braking phase, preserving the learned walk's balance and
gait phase. Neutral takes over only after measured planar speed falls below
the static handoff threshold. The release latch is also evaluated after this
pre-settle phase rather than before it.

The intermediate replay `procedural-shot-controlled-brake-s20261242-v2`
reduced GetUp samples from 261 to 106 and made one attributable Shot contact,
but only through the timed forward-contact fallback. After moving the latch
behind controlled braking, `procedural-shot-latch-after-brake-s20261242-v3`
passed with 62 `ProceduralKickExecute/Hold` samples, zero fallback samples,
and one physical Shot contact. Release occurred at 7.59 s with ball-local
position `(0.321, 0.040) m`, essentially zero yaw error, zero planar speed,
`0.08 deg/s` tilt rate, and `0.3 deg/s` maximum leg rate. The ball then crossed
the target goal line inside the posts near `y=-0.78 m`. This validates one
static strong-shot execution, not a dynamic striker policy.

`apollo-vs-base-static-release-fix-s20261242-v22` lost `0:1`. It exposed a
separate frame-contract defect: the only exact Dribble release was centred in
the requested-target frame, but the torso still differed by 5.6 degrees. The
procedural runner correctly rejected the resulting body-frame ball position,
which was about 3.3 cm away from its measured lateral anchor. Procedural action
admission remains broad, but body-fixed static trajectories now finish to a
1 degree release alignment before dispatch. The corresponding isolated
dribble replay passed with three physical contacts and no decision-to-runner
frame rejection.

The v22 concession also disproved the former 1.5 s lost-ball search bound in a
specific defensive state. Player 7 was already at `x=-23 m` outside the
goalkeeper area when its last local ball observation in the final eight metres
expired; it then retreated toward the default `x=-7 m` formation while the
opponent carried through the vacated second line. A genuine open-play
coordinate in this danger region is now retained for at most 12 s strictly for
movement, never contact. Exactly one AP stays at the legal goalkeeper-area
boundary and its remembered lateral target decays toward goal centre as
uncertainty grows. Ordinary midfield search still expires after 1.5 s and all
memories reset across play-mode transitions.

`apollo-vs-base-bodyframe-danger-memory-s20261242-v23` then won `1:0`, with
fresh-ball median x `+14.65 m`, `74.85%` opponent-half occupancy, maximum x
`+28.18 m`, and zero developed-team illegal-defense penalties versus seven for
pristine Apollo. The goal at 170.91 s was a continuous pressure/body-contact
carry across the line, not an exact Shot; the 11 Shoot samples remained in
setup. Fourteen independent GetUp entries still followed ordinary Walk. The
paired v22 loss and v23 win therefore support the repaired integration paths
but still do not establish stable superiority.

### 4.13 External-reference status

Apollo is a qualified and useful 2026 baseline, but no official result
currently supports calling it the 2026 champion. The official awards page
lists FC Portugal as 2025 champion and Apollo3D third in 2024; the official
2026 page presently provides qualification status. The 2026 league also moves
to MuJoCo and Booster T1, making the official ICRA 2026 T1 striker repository's
four-stage chase-teacher, kick-teacher, DAgger student, and constrained-P3O
route directly relevant to the remaining action work. Sources:

- <https://ssim.robocup.org/3d-simulation/3d-awards/>
- <https://ssim.robocup.org/2026/02/16/robocup-2026-soccer-simulation-3d-qualification-results/>
- <https://ssim.robocup.org/2025/12/16/robocup-2026-soccer-simulation-3d-call-for-participation/>
- <https://github.com/Daffan/humanoid-soccer>

### 4.14 Static-shot admission versus learned transition

The official ICRA 2026 striker pipeline confirms the architectural direction:
long-distance chase, directional kick training, noisy-observation DAgger, and
constrained student adaptation form one continuous learned skill. It does not
justify applying a static keyframe's release conditions to that dynamic policy.
The runtime therefore keeps three contracts distinct:

- strategy admission decides whether an action is worth attempting;
- a static procedural trajectory owns braking, exact body-frame pose, tilt,
  joint-rate, and final release checks;
- the fixed-2 m learned/residual transition consumes live gait phase after its
  own one-cycle pose confirmation and never uses the static-shot admission
  estimate.

Natural-match v23 provided three concrete counterexamples to the old Shot
admission. Two began with `0.64--0.73 m` of lateral relocation, and another
occurred while the ball moved at about `1.6 m/s`; one opponent was only
`0.25 s` from the ball. These were not releasable shots hidden behind a strict
threshold. They were static actions selected from states that could not finish
their setup before the opportunity disappeared.

Shot commitment now requires a target still inside the calibrated 4 m range,
at most `0.35 m` initial lateral setup error, at most `60 deg` initial yaw
error, no observed ball speed above `0.60 m/s`, and an estimated
translate/turn/acquire time that fits before the nearest opponent arrives.
These are coarse attempt conditions, not relaxed or duplicated release
conditions. A committed static shot is rechecked until the motion request
actually starts, so a newly moving ball or lost race returns immediately to
continuous pressure.

All 17 C++ tests pass, including positive calm-shot admission and negative
moving-ball, opponent-first, and large-lateral-error cases. The retained
physical scene `procedural-shot-static-admission-s20261242-v5` also passed with
62 procedural Shot samples, one attributed contact, zero fallback contacts,
and zero GetUp samples. The first scene run was invalidated by monitor rebeam
timing and is retained as a fixture-reliability warning rather than counted as
action evidence.

The first two natural comparisons with the same code both finished `1:0`
against pristine Apollo. In v24 the fresh-ball median x was `+3.21 m`, 68.8%
of buckets were in the opponent half, and only five Shot status samples
remained. In v25 the corresponding values were `+12.13 m`, 78.5%, and 109 Shot
samples. The latter included one 2.62 s setup that reached the ball but failed
to settle yaw/lateral pose and leg rate before the bounded forward-contact
fallback. This is evidence that infeasible commitments were reduced without
removing real opportunities; it also identifies dynamic locomotion-to-shot
transition quality as the next action bottleneck.

The next two retained comparisons both finished `0:0`, making the current
small series `2W-2D-0L`, goals `2:0`. This is encouraging but not yet a
statistically stable superiority claim. V26 was a materially different game:
fresh-ball median x was `-13.16 m`, only 25.5% of buckets were in the opponent
half, 183 status samples selected Clear without one exact contact, and the
late own goal kick spent 251 setup samples in approach plus 45 in lateral
relocation. Its coordinator had frozen the ball where it crossed the line
(`-27.5,-3.04`) before the referee placed the official restart at about
`-25.5,0`; because the mode and epoch did not change, taker alignment and the
motion setup then used different anchors.

Restart geometry now reacquires a confirmed large, slow referee relocation
before any contact has been authorized, increments the plan revision, and
keeps the taker and selected variant frozen. The test covers both primary and
already-fallback safety plans and rejects a rolling ball as a false
placement. Exact play mode and restart anchor are now present in periodic
runtime telemetry. A monitor-injected physics replay is not counted: the
installed RCSSServerMJ monitor parses `playMode` but its command implementation
does not apply it. A future natural goal kick must provide the physical check.

V27 had fresh-ball median x near `0 m`, 49.0% opponent-half occupancy, no
exact or fallback kick, and 24 independent GetUp entries. It also exposed an
action-ownership bug independent of release thresholds: several players began
a Dribble precision setup and then lost AP on the next locally inconsistent
role assignment. For example, player 6 started setup at 61.64 s, lost the
Pressure/AP duty roughly one second later, and restarted a different action at
65.22 s. A committed local action or pass now refreshes a 0.35 s rolling AP
lease. It is overridden by a fall, expires immediately after refresh stops,
and avoids a long global ownership lock.

V28 then finished `0:0` with fresh-ball median x `+26.28 m`, maximum x
`+27.21 m`, and 94.8% of fresh buckets in the opponent half. Ten independent
GetUp entries followed ordinary Walk. Three fallback Shot contacts executed;
there was still no exact targeted pass. The retained player stayed Pressure/AP
through the observed Dribble setup episodes, so the lease repaired the v27
handoff failure. Those same episodes exposed the next, different fault: the
static short-touch actor was retained while the ball accelerated away at about
`1.9--2.4 m/s`. Static Dribble admission now rejects observed ball speed above
`0.45 m/s`, and an existing commitment is cancelled after more than `0.12 m`
of ball displacement. This returns to continuous pressure without disabling
the imperfect but useful locomotion actors.

The score series is now `2W-3D-0L`, goals `2:0`. It is still too small and
one-sided to claim stable superiority, but v28 supplies strong territorial
evidence and isolates release quality—not AP ownership—as the remaining final-
third bottleneck.

The fixed-2 m ONNX transition and deterministic parameterized bank now also
have separate executable envelopes. Active ONNX may start only in its actual
fixed-distance corpus (`1.90--2.10 m`, requested speed `1.23--1.63 m/s`), with
target yaw up to `12 deg` and its trained body-frame ball slot. The static bank
keeps its `2 deg` yaw limit. In the overlapping slice both runners are prepared
and the deterministic residual remains a same-cycle fallback; outside the
static slice, ONNX failure holds/rejects rather than converting a directional
request into fixed forward contact. Shadow mode never broadens live decisions.

The first active-ONNX comparison, v29, lost `0:1`; fresh-ball median x was
`+2.68 m` and opponent-half occupancy was 60.1%. This run contained no exact
TargetedPass release and no `LearnedKickExecute` sample, so it is not evidence
for or against the actor. The decisive event at 145.4 s instead exposed a
keeper-state defect. From 142.0 s the local keeper continued an old
`GoalkeeperSmother` target while its near-contact ball track was already
1.1--1.7 s old; by the next fresh observation the keeper had moved laterally
outside the live shooting lane.

Goalkeeper action freshness is now stricter than movement-only tactical
memory. A visible or at-most-0.20 s old ball may authorize Smother/Intercept.
A still-close torso-occluded track can only preserve the keeper's current body
block, never continue a race; a displaced stale track returns the keeper to a
central goal-line hold. Unit coverage includes the exact v29 stale/displaced
shape.

The next natural comparison, v30, finished `0:0`. Fresh-ball median x was
`-3.248 m`, opponent-half occupancy was 37.23%, and the observed range was
`-27.49..+25.62 m`. The ball naturally reached our goalkeeper area without
recreating the stale lateral chase: the keeper remained in
`GoalkeeperHold`, with no stale `GoalkeeperSmother` sample. Eleven independent
GetUp episodes still followed ordinary Walk, so locomotion stability remains
a match-result variable. Including v30, the retained natural record is
`2W-4D-1L`, goals `2:1`; stable superiority remains unproven.

Controlled full-team diagnostics v31--v35 then isolated pass acquisition,
retention and release without replacing the production strategy. Initial
proposal still requires a calm, self-owned ball and a safe opponent reach-time
margin. Once a pass is committed, a temporary gait-speed transient no longer
cancels it; a genuinely near-contact but torso-occluded ball may retain the
same commitment for at most 1.5 s. Release itself is not widened and continues
to require the selected static or learned executor's exact pose contract.

V35 produced the first real server-side `LearnedKickExecute`: a 2.046 m,
1.43 m/s request moved the ball 1.052 m with `+4.70 deg` signed direction
error, 0.087 m lateral error and no fall. The pass reached `Commanded` and
`Executed`, but ended in `Timeout` rather than receiver possession. This proves
that the active ONNX route and its separate learned-transition guard are wired
through real server physics. It does not promote the actor: one underpowered
contact and no completed reception are insufficient, while its frozen exact-
CPU evaluation remains 27/92 with one fall. Natural comparisons therefore keep
the model in shadow mode unless an active ablation is requested explicitly.

The first natural comparison after that retention repair, v36, finished
`0:0`. Fresh-ball median x was `+10.499 m` with 69.25% of fresh buckets in the
opponent half; strictly visible-ball median x was `+2.290 m` with 65.84% in the
opponent half. This is territorial evidence, not finishing evidence: one
natural pass was proposed and later cancelled without Ready, no exact or
fallback kick executed, and 21 independent GetUp entries followed ordinary
Walk. Our side committed zero illegal-defense events while pristine Apollo
committed three. The retained natural record is now `2W-5D-1L`, goals `2:1`.
Strategy can hold useful territory, but action acquisition, ordinary locomotion
stability and completed contact still prevent a superiority claim.

### 4.15 Tactical-churn ablation and conservative phase handling

V36 also exposed a separate strategy cost. With the complete `TeamTactics`
layer enabled, periodic telemetry contained 832 duty switches, 6,673 plan-
revision switches, 829 target jumps above 1 m and 534 facing changes above
45 degrees. These counts include legitimate ball and play-mode changes and are
not physical-event counts, but they are much larger than the same measurements
with only dynamic open-play duties disabled. The difference matters because the
deployed walk frequently falls and every unnecessary target reversal consumes
the scarce stable gait time needed to reach a pass or shot release.

Two independent natural ablations retained the developed role assignment,
formation, restart coordinator, goalkeeper fixes and full action stack, while
setting only `APOLLO_ENABLE_TEAM_TACTICS=0`. Both beat pristine Apollo `1:0`.
V37 placed 83.0% of fresh-ball buckets in the opponent half and recorded nine
independent GetUp episodes; its goal followed continuous pressure/forward ball
progress rather than an exact kick. V38 placed 63.5% of fresh-ball buckets and
84.3% of strictly visible-ball buckets in the opponent half, with 13 independent
GetUp episodes and three attributed fallback contacts. Neither run executed an
exact targeted kick. The paired result is strong evidence that the former
high-frequency duty layer could obstruct the currently usable simple pressure
game; it is not yet proof that all marking or support concepts are inferior.

The repair keeps emergency behavior and removes only unsupported urgency:

- a transient `Contested` or `Unknown` observation now receives the same 0.40 s
  grace as an ordinary weak possession change, rather than bypassing phase
  hysteresis for one frame;
- sustained ambiguity still becomes `Transition`, but off-ball players retain
  their phase-aware formation instead of simultaneously switching to mark,
  cover, block and outlet targets;
- a reachable moving-ball intercept, goalkeeper duty and AP pressure continue
  to bypass the conservative hold immediately;
- attacking support targets are held for 0.9 s, long enough for the current
  walking stack to settle onto one lane, without becoming action ownership.

V39 exercised that conservative repair with dynamic tactics enabled. It reduced
plan-revision changes from 6,673 to 5,659, but still produced 861 duty switches
and 689 target jumps above 1 m. The match finished `0:0`; fresh-ball median x
was `-10.27 m` and only 4.7% of fresh buckets reached the opponent half. Ten
independent GetUp entries followed ordinary Walk, and no exact or fallback
contact executed. The state repair is valid and retained, but it did not make
the complete dynamic layer competitive with the two simple-shape ablations.

The full dynamic orchestrator remains available behind
`APOLLO_ENABLE_TEAM_TACTICS=1` for diagnosis, but is no longer the production
architecture. The default now extracts only independently useful cooperation:
Support/Unmark for the two attacking lanes, unique centre-back Mark targets,
pass communication/receiving, bounded lost-ball search and goalkeeper safety.
Broad Cover/BlockLane/Intercept/Outlet retasking is discarded from the default.
AP is forced back to Pressure whenever the ball is actionable, so a whole-team
plan cannot pull the only ball player away from a walk, turn or contact action.
Stable superiority is still unproven, but it is no longer a prerequisite for
continuing action-layer development.

The same motion-first rule applies inside the AP action selector. A planner
`Move` candidate remains visible in telemetry, but no longer executes a generic
walk whose stop radius is centred on the ball. The direct approach/push
controller owns that translation and keeps its target beyond the ball. Exact
Dribble/Pass/Shoot/Clear actions may still take ownership when their concrete
capability contract is satisfied; failed or unavailable releases fall back to
continuous pressure instead of a team-strategy hold.

The first right-side comparison, v40, then lost `0:1`; fresh-ball median x was
`-4.55 m` and only 21.6% of fresh buckets were in the opponent half. Its first
analysis incorrectly reported 31 current-team Illegal defense events because
the parser interpreted the robot id prefix `r` as the right side. The referee
message carries the actual area at the end of the line. Re-parsing that field
attributes 30 events to left-side pristine Apollo and one to the right-side
current team. The analyzer now reports both physical side and developed-team
counts explicitly.

V40 still revealed a real, smaller legality risk: with dynamic duties disabled,
player 6 received ordinary Formation targets as deep as `x=-26.8 m`, while the
server permits only two own-team bodies in the goalkeeper area. A first repair
reserved the area for the goalkeeper alone. V41 exercised that configuration
from the right and also lost `0:1`, but current-team Illegal defense fell to
zero while left-side Apollo received 30 events. Fresh-ball median x improved to
`-0.22 m`, strictly visible-ball median x was `+0.74 m`, and only five
independent GetUp entries followed Walk. No exact or fallback contact executed,
so the result still exposes finishing and deep-defense weakness rather than an
action success.

The final geometry contract uses the legal capacity instead of abandoning it:
GK and the fixed CBL role may enter, while every other formation and generic
walk target stays at least 0.8 m beyond the inclusive boundary. A fixed role
avoids transient overlap when the ball changes side. Unit and decision-level
tests cover the v40 deep-defense geometry; the two-player version still needs a
right-side natural replay before promotion evidence is updated.

### 4.16 First motion-first collaboration match

The first full match from commit `763ef64` reached `GameOver` without a server
or client failure. My3D-Current played left and lost `0:1` to pristine Apollo.
This one result is diagnostic rather than a promotion gate. It does verify that
the filtered production plan is live: telemetry recorded 304 Support, 364
Unmark, 555 Mark and 1,052 Pressure samples, while all 342 near-ball samples had
zero Formation duty. Only five near-ball samples were truly neutral/idle;
another 125 were pure turns, so slow orientation remains an action-layer cost
rather than a formation override.

The action stack executed 129 FastWalkV2, 744 RapidTurnV1 and 17 fallback-contact
samples, but still produced zero exact kick samples. Twelve Pass planning
samples reached Proposed and then Cancelled without Ready/Committed/Executed.
Sixteen independent GetUp entries followed ordinary Walk. Fresh-ball median x
was approximately midfield (`-0.014 m`) with 45.7% of buckets in the opponent
half. My3D-Current committed zero illegal-defense events; pristine Apollo
committed 31. The next implementation priority is therefore contact acquisition,
turn/walk stability and receiver readiness, not restoration of the discarded
broad orchestrator.

## 5. What is actually better, and what is not yet proven

The following improvements are supported by code invariants and tests rather
than tactical taste: deterministic unique defensive jobs, stale-observation
rejection, restart legality, explicit pass identities and terminal outcomes,
motion rejection feedback, goalkeeper reachability checks, and safe rejection
outside an action envelope.

The following are hypotheses until match metrics support them: support-point
utility, pass-versus-dribble weights, risk-mode formation offsets, generic
humanoid reach-time constants, and the benefit of frequent role/duty changes.
Through-space passing remains deliberately absent because the runtime does not
yet expose a trustworthy teammate-velocity or receiver-run contract.

The mixed score record is primarily evidence that the action-release contract
and locomotion stability still dominate outcome variance, not proof that
formation, marking, or pass geometry is inferior. Once contact is available,
the most likely remaining tactical failure modes are overvalued passes,
excessive target churn, and decisions whose estimated reach margin does not
match the deployed locomotion speed.

## 6. Controlled comparison protocol

The comparison launcher keeps role assignment, formation, lightweight
cooperation and restart legality constant. The switch below adds the discarded
broad duty orchestrator; it does not disable support, marking or pass comms in
the default path:

```bash
# Experimental full dynamic-duty stack versus pristine Apollo
APOLLO_ENABLE_TEAM_TACTICS=1 \
  scripts/run_web_match_vs_apollo_base.sh 120000

# Motion-first collaboration default with the same developed action stack
scripts/run_web_match_vs_apollo_base.sh 120000

# Put the developed team on the right while preserving it as kickoff side
MATCH_CURRENT_SIDE=right scripts/run_web_match_vs_apollo_base.sh 120000
```

Action attribution can then be separated without changing team strategy:

```bash
APOLLO_ENABLE_FAST_WALK=0 APOLLO_ENABLE_RAPID_TURN=0 \
  scripts/run_web_match_vs_apollo_base.sh 120000

APOLLO_ENABLE_PARAMETERIZED_KICK=0 APOLLO_LEARNED_KICK_MODE=off \
  scripts/run_web_match_vs_apollo_base.sh 120000
```

The comparison launcher keeps the unpromoted learned kick loaded in `shadow`
by default. This is deliberate: its fixed evaluation is weaker than the 2 m
residual path, so allowing it to own actuators would hide the stronger fallback.
Use `APOLLO_LEARNED_KICK_MODE=active` only for the explicit model ablation.

For each retained run, report score, legal penalties, independent falls,
time-to-ball/role target, physical contacts, ball progress, pass terminal
outcomes, shots, and possession chains. The current web server exposes no seed
argument, so full matches must use repeated independent runs and side swaps;
matched seeds remain applicable to deterministic single-action evaluation.
No single score, training reward, or status-sample count is sufficient by
itself.

### 6.1 Permissive action-release experiment (2026-09-06)

The current production experiment reverses the earlier static-release bias:
the team now accepts a wide contact corridor, normal walking phase, up to
`15 deg` directional mismatch and immediate release after one complete legal
sample. Deterministic fallback begins much earlier when exact acquisition does
not converge. The aim is to test whether action starvation, rather than
trajectory precision, is the dominant match weakness.

This change supersedes earlier passages in this audit that describe the
centimetre-scale static envelope as the current default. Those passages remain
historical evidence for why the old path was precise but rarely triggered.
The widened mode must be judged by attributed contacts, ball progress,
direction/range error, independent falls and opponent dispossessions. If it
acts frequently but loses control, the correct successor is the documented
phase-conditioned transition training—not restoration of an action-starving
one-degree release gate.

### 6.2 Natural goal-kick coarse-relocation failure (2026-09-06)

The first natural own goal kick after referee-placement reacquisition exposed
a different controller failure.  The coordinator correctly froze goalkeeper 1
as taker, reacquired the official ball at about `(-25.5, 0)`, and entered its
safety-plan `Aligning` phase.  The keeper then remained near
`(-26.38, 0.50)` for roughly 12 seconds with the ball about `0.99 m` away.
Telemetry showed `kick_mode=None` and the same local `(0.25, 0)` Walk request
throughout: no contact action had yet been released.

That command came from coarse lateral relocation but inherited the final
restart-precision speed cap.  On the deployed stable walk actor it became a
low-speed hold rather than observable translation.  Coarse relocation now has
its own `0.45--0.50 m/s` forward command while it faces the canonical setup
waypoint; the existing `0.25/0.20 m/s` caps remain in force for final
longitudinal/reverse correction near the release slot.  A regression preserves
the exact live pose and requires the stronger local forward command.  This is
a code-level correction; a fresh natural goal kick remains required to confirm
physical release and ball travel.

## 7. Immediate development order

1. Complete the motion-owned ball path: direct pressure must reach and cross
   the ball; exact Dribble/Pass/Shoot/Clear must either acquire and release or
   return immediately to pressure. A planner Move/Hold may not stop the AP.
2. Improve the learned transition's distance and complete the receiver
   lifecycle. V35 reached only 1.052/2.046 m and timed out; the next corpus must
   include those underpowered server outcomes as hard negatives/DAgger states.
3. Validate referee-placement reacquisition on the next natural own goal kick,
   and verify that boundary smother actually reaches the ball and hands off to
   Clear instead of leaving the keeper parked outside engagement distance.
4. Preserve continuous pressure, wide finishing cut-in, danger-memory guard,
   and low-turn goal-mouth aim over repeated natural runs; `2W-5D-1L` remains
   a small, one-sided sample and is not a stable superiority result.
5. Make procedural dribble and all fixed-distance releases start from a real
   walking gait phase. The next data task is phase-conditioned approach-to-
   contact BC/DAgger, including v31--v35 acquisition and retention failures,
   not another wider static-trajectory pose gate.
6. Build a phase-conditioned BC/DAgger striker student from successful complete
   approach-release trajectories; do not repeat unsupervised residual PPO.
7. Train and promote stable long-forward, rapid-turn, and later lateral skills
   with explicit fall, drift, speed and transition tests; ordinary Walk is also
   implicated in current falls and must remain in the audit.
8. Keep the 2/3.5/5 m deterministic bank and original forward contact as
   explicit fallbacks while collecting server outcome traces.
9. Calibrate reach time and action utility from deployed FastWalk/turn logs.
   Reintroduce a discarded broad duty only when it cannot pre-empt an available
   action and its local value is clear; otherwise delete it from production.
10. Use periodic full matches as diagnosis, not as a gate on action or
   collaboration implementation. Retain losses and draws when comparisons are
   run so regressions remain visible.
