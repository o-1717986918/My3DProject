# Complete and excellent team capability roadmap

Status: strategy code closure delivered; physical calibration and competitive
improvement remain active

Last audited: 2026-09-05

Authoritative runtime: `runtime/apollo/`

Accepted committed baseline before this delivery: `27695b4`

The developed-versus-pristine evidence and logic-by-logic assessment are kept
in `current-vs-apollo-strategy-audit.md`. This roadmap defines intended team
capability; the audit distinguishes implemented invariants from tactics whose
competitive value is still unproven.

## Objective and definition of completion

The project target is a complete, fluent, and strong seven-player team. The
strategy layer is complete when every legal match state produces a coherent
team plan and every selected action is either inside a deployed capability
contract or rejected into a safe non-contact command. This definition covers
decision semantics and code closure; it does not turn an experimental physical
kick into a reliable pass, shot, or clearance.

The implemented hierarchy is:

```text
canonical world evidence + eight-byte team communication
                         |
                         v
freshness + possession owner + stable tactical phase/risk
                         |
                         v
role assignment + one full-team TeamPlan and revision
                         |
                         v
typed Hold/Move/Dribble/Pass/Shoot/Clear candidates
                         |
                         v
utility + legality + exact action capability envelope
                         |
                         v
motion lifecycle and observed ball outcome
  learned ONNX -> procedural implementation -> safe rejection
```

This preserves deterministic football semantics while allowing better learned
motion to replace an executor without rewriting team decisions.

## Delivered strategy surface

| Area | Delivered implementation | Evidence boundary |
| --- | --- | --- |
| World and legality | canonical left-to-right team frame, fresh/stale observations, complete play-mode guards, GameOver neutral | no tactic uses a ball older than 0.75 s |
| Possession and phase | explicit owner ID/team, reach-time evidence, 0.40 s weak-turnover confirmation, immediate strong turnover, 1.20 s counter-press, attack/defend/transition/set-play | local agents remain limited by their observations |
| Match risk | configurable late-match `Balanced`, `ProtectLead`, and `ChaseGoal` modes | no guessed match duration when the clock is unavailable |
| Roles | one goalkeeper and active player, sticky minimum-cost remaining roles, fallen/stale-player degradation and goalkeeper replacement | assignments are deterministic for equivalent snapshots |
| Team plan | one `plan_all()` result containing every player, source time, freshness and deterministic revision | plan revision is logged; the eight-byte protocol does not yet carry a consensus digest |
| Attack without ball | jointly allocated striker/central support or unmark lanes, outlet/cover, conservative offside line, hard teammate spacing and short target residence | no velocity-based through run is claimed |
| Defense | unique pressure owner, reachable single interceptor, lane block, cover, threat-ranked non-duplicated centre-back marking | reach time uses calibrated conservative constants, not a learned opponent model |
| Active-player choice | one evaluator compares executable Hold, Move, Dribble, Pass, Shoot and Clear candidates with deterministic tie breaks | experimental actions appear only when explicitly enabled and inside their exact envelopes |
| Passing | direct/leading candidates, lane and arrival races, stable receiver Ready dwell, complete state machine from proposal through physical terminal outcome | current targeted pass remains a narrow experimental physical skill |
| Receiving | persistent intention across speech gaps, moving-ball reachable intercept target, timeout/cancel and local-control handoff to the next role cycle | there is no learned first-touch controller |
| Motion feedback | next-cycle Running/Completed/Rejected/TimedOut feedback; command completion alone never counts as ball execution | success requires observed ball displacement, receiver control, interception, out, or timeout |
| Goalkeeper | angle hold, reachable goal-line interception, safe walk-based smother, AP rebound cover, goal-kick participation, and contracted open-play safety clear | no dive or hand-catch capability is claimed |
| Restarts | deterministic epoch/revision, taker/receiver, positioning, Ready, alignment, primary/alternate/safety variants, opponent-aware lane choice, feedback, release verification, one fallback, taker lockout and PlayOn handoff | penalty/corner variants use only existing contact capability; tactical intent is stronger than physical placement accuracy |
| Telemetry | duty/target/mark, plan revision/freshness, possession owner, lifecycle, candidates/rejections, selected action, restart variant/target and decision latency | match logs are evidence, not a performance claim |

## Pass and action lifecycle contract

The passing path is explicitly:

```text
Proposed -> Ready -> Committed -> Commanded -> Executed
                                         |        |
                                         |        +-> ReceiverZone -> Received
                                         +----------> Intercepted / Out / Timeout
Proposed or Committed ------------------------------> Cancelled / Expired
```

`Ready` requires the receiver to remain near the target, upright, slow and
facing the ball for a stable residence interval. `Completed` motion feedback
only means the motor request ended. `Executed` requires ball movement;
`Received`, `Intercepted`, and `Out` require world evidence. A terminal state
is broadcast briefly while local non-pass planning resumes; the pass may be
recommitted only after its one-shot retry delay.

The common action planner never silently changes semantics. In particular, an
out-of-envelope targeted pass cannot become a fixed forward kick. Hold and Move
remain available through the stable walk contract; experimental Dribble,
TargetedPass, Shot and Clear are admitted only by their exact distance, speed
and orientation contracts. Once such an action has been admitted and has spent
1.20 seconds continuously attempting a valid near-ball setup, it may explicitly
authorize a wider-corridor forward-contact fallback. That path is logged as
`FallbackKick*` and never counted as target-accurate execution.

## Role behavior closure

- The active player approaches the ball, compares all executable actions,
  aligns to a frozen target, executes, consumes feedback, and replans.
- The striker and midfielders create separated receiving/support/outlet lanes,
  retain receiver intent across communication slots, and hand local control
  back to normal role selection after receiving.
- The defensive unit assigns one pressure/intercept owner, distinct threats to
  the two centre-backs, a lane blocker and cover so players do not all chase.
- The goalkeeper prioritizes a reachable goal-line crossing, otherwise holds
  its angle, claims only a safe loose ball in the actual goalkeeper area, asks
  the active player to protect the second ball, and clears through the same
  capability-gated action path after a successful smother.
- Every opponent restart returns the team to legalized formation behavior. Our
  restarts freeze roles and direction, wait for required preparation, execute
  once, verify release, then lock out the taker until another touch or distance
  evidence permits open play.

## 2D migration closure

The local reference tree `/home/win98/my_projects/rbc/teams` was used only for
algorithmic structure. No source was copied because extracted archives contain
mixed licence notices and incomplete provenance. The useful concepts now
implemented are:

- `bhv_basic_move`: intercept/block before unmark/support before formation;
- `bhv_basic_block`: reach-time race, one owner and short target residence;
- `bhv_unmark`: candidate points, lane space, offside constraint and latching;
- `intention_receive`: persistent intent, interruption/timeout and moving-ball
  interception;
- `role_goalie`: separate hold, intercept, smother and distribution decisions;
- set plays: frozen roles, alternatives, readiness, deadlines, outcome and
  bounded fallback;
- HELIOS-style typed generators and deterministic evaluator.

The following 2D mechanics were deliberately not migrated: dash cycles,
2D ball decay, stamina/body models, and `PredictState`. They would produce
incorrect timing and physics in RCSSServerMJ 3D.

## External-source rationale

- [FCPCodebase](https://github.com/m-abr/FCPCodebase) is valuable for low-level
  skills and runtime structure, but its sample high-level team is intentionally
  basic.
- [UT Austin Villa 3D](https://github.com/robocup3d/op2) supports the separation
  of walk/kick/world state, positioning and role assignment.
- [HELIOS base](https://github.com/helios-base/helios-base) motivates typed
  cooperative actions and bounded evaluation, not copied 2D dynamics.
- [SCRAM](https://www.cs.utexas.edu/~AustinVilla/details/AAAI15-MacAlpine.html)
  and [prioritized marking](https://www.cs.utexas.edu/~AustinVilla/details/LNAI16-MacAlpine.html)
  motivate unique, deterministic formation and defensive job allocation.
- [Pre-planned set plays](https://www.cs.utexas.edu/~pstone/Papers/99aij/node19.html)
  motivate role mapping, alternative branches, termination and timeouts.

These are architectural inferences, not claims that another robot's parameters
or tactics can be copied unchanged.

## Remaining work: performance improvement, not an unimplemented match path

The strategy runtime now has a safe code path for ordinary open play, degraded
play, all supported restarts, receiving and every deployed action category.
The remaining items improve strength or evidence and must not be described as
already solved:

1. Add a low-bandwidth plan digest only after measuring real cross-agent plan
   disagreement; the current speech packet is already fully allocated to state
   and pass lifecycle data.
2. Add explicit path reservations if match traces show support/defensive route
   crossings that local obstacle avoidance cannot resolve.
3. Add velocity-conditioned through passes only after temporal player velocity
   or an explicit run intent exists. A fixed +x offset is not a through pass.
4. Calibrate direction-dependent player reach time and ball uncertainty from
   server traces, then tune soft evaluator weights on held-out data.
5. Consider bounded depth-two pass continuations only when the first pass and
   receiver control are physically repeatable. Long open-loop action chains
   remain prohibited.
6. Build paired side/seed fixtures against passive, pressing, deep and direct
   opponents before making any claim of tactical superiority.

## Parallel motion and training line

Strategy completion does not discard training. The stable Apollo walk/get-up
remain the default safety layer; full-body fast walk stays opt-in until its
drift/fall/transition evidence improves. ONNX remains the preferred ball-action
executor, deterministic procedural trajectories remain teacher/fallback, and
only measured sub-envelopes are published through
`ActionCapabilityRegistry`. Training artifacts belong under
`/home/win98/rl_runs`, never in the C-drive repository.

The detailed motion work is tracked in `docs/rl-training-plan.md` and
`docs/model-free-parameterized-kick-plan.md`.

## Delivery verification

Completed checks for this strategy delivery are:

- clean WSL CMake build and 16/16 CTest targets;
- deterministic snapshot/replay coverage for phase hysteresis, unique duties,
  pass terminal outcomes, receiver persistence, goalkeeper smother/clear and
  restart variants/fallbacks;
- 59/59 main Python tests and 5/5 training deployment-contract tests;
- shell syntax and `git diff --check`;
- a 900-cycle headless 7v7 targeted regression with fourteen connections,
  fourteen PlayOn clients, fourteen clean exits, no client/server/legality
  error, 124 pass-plan samples, six Ready samples, nine targeted-kick samples
  and one observed contact event;
- 2,520 decision timing samples at 67 us median, 2,412 us p99 and 3,279 us
  maximum on the local WSL host.

These checks establish implementation closure. Physical pass/shot/clear
accuracy and team excellence require the separate multi-trial evidence above.
