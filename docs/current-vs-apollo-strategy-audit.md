# Current team versus pristine Apollo: strategy and action audit

Status: code audit complete; opponent-diverse match evidence pending

Audit date: 2026-09-05

Compared upstream revision: `71018c968969d6e55130b0e1987cd5b4f5c3b4df`

## 1. Bottom line

The developed strategy is structurally safer and more expressive than pristine
Apollo, but it is not yet demonstrated to be stronger overall.  The retained
comparison match ended `My3D-Current 0 - 1 Apollo-Base`.  That result exposed a
specific action-integration regression: the developed side selected passes,
dribbles, shots, and clears but never entered an exact kick motion, while the
base side could still score through its simple walk-through-ball behavior.

The correct response is therefore neither to remove the new team strategy nor
to hide the loss behind more tactical complexity.  This revision restores a
bounded, explicit forward-contact fallback, loosens the exact release contract,
and adds a team-tactics ablation switch.  Tactical value will be decided with
repeatable A/B evidence after the action path can actually realize its choices.

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

After 0.45 seconds of continuous near-ball setup for dribble/shot/clear, or
1.20 seconds for a coordinated pass, a request inside a separate broad contact
corridor may explicitly use pristine Apollo's forward-contact macro. An exact
procedural release slot always wins over this timeout. Telemetry names the
fallback explicitly; an ordinary unsupported targeted request still returns
`RejectedTargetedKickHold`. The fallback is therefore a declared fail-soft
action, not a silent claim that a fixed contact is precise.

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
three metres, while the central opponent-first race rule and legal goalkeeper
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

The present loss is primarily evidence against the old action-release contract,
not evidence that formation, marking, or pass geometry is inferior.  Once
contact is available, the most likely remaining tactical failure modes are
overvalued passes, excessive target churn, and decisions whose estimated reach
margin does not match the deployed locomotion speed.

## 6. Controlled comparison protocol

The comparison launcher keeps role assignment, formation, and restart legality
constant while allowing only the new open-play duty layer to be disabled:

```bash
# Full developed stack versus pristine Apollo
APOLLO_ENABLE_TEAM_TACTICS=1 \
  scripts/run_web_match_vs_apollo_base.sh 120000

# Same developed action stack, without new open-play TeamTactics duties
APOLLO_ENABLE_TEAM_TACTICS=0 \
  scripts/run_web_match_vs_apollo_base.sh 120000
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
outcomes, shots, and possession chains.  The final judgment should use matched
seeds and side swaps.  No single score, training reward, or status-sample count
is sufficient by itself.

## 7. Immediate development order

1. Validate the earlier near-post goalkeeper challenge in one retained match,
   with corrected `FallbackKick*` attribution rather than inferred contacts.
2. Build a phase-conditioned BC/DAgger striker student from successful complete
   approach-release trajectories; do not repeat unsupervised residual PPO.
3. Keep the 2/3.5/5 m deterministic bank and original forward contact as
   explicit fallbacks while collecting server outcome traces.
4. Calibrate reach time and action utility from deployed FastWalk/turn logs,
   then repeat tactics-on/tactics-off and side-swapped comparisons.
5. Decide superiority only from repeated full matches; retain every loss and
   draw instead of selecting favourable scores.
