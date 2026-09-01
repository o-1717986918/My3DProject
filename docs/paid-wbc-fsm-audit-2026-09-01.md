# PAiD and `wbc_fsm` adoption audit

Audit date: 2026-09-01

Decision status: binding correction to the R1 striker route. It does not
change exact RCSSServerMJ acceptance or authorize external source/assets to be
copied into this repository.

## Evidence boundary

Primary sources:

- [PAiD project page](https://soccer-humanoid.github.io/);
- [PAiD arXiv preprint](https://arxiv.org/abs/2602.05310), submitted
  2026-02-05; arXiv describes a conference submission but does not establish a
  named peer-reviewed venue;
- [PAiD official code and assets](https://github.com/TeleHuman/HumanoidSoccer),
  pinned at `e72e470230047dedaf66df0983f1d0ab746faeb5`;
- [`wbc_fsm` source](https://github.com/ccrpRepo/wbc_fsm), pinned at
  `b352409d73bed469169334040ee9ea70bc28a5f1`.

PAiD's `91.3%` real-world success rate and eleven consecutive randomized shots
are author-reported results, not results reproduced on Booster T1 or
RCSSServerMJ. All adoption decisions below separate released implementation
facts from transfer hypotheses.

## Reference value

| Source | Value to this project | Direct reuse | Binding decision |
|---|---|---|---|
| PAiD | high for motion data structure, progressive training, soccer reward staging, recurrence and physics-aware randomization | no direct checkpoint reuse: released policy is a 29-DoF G1 two-layer LSTM with a 160-value observation, while `striker_policy_v1` is T1 23-DoF/102-value | adopt the training decomposition immediately; audit and retarget motions locally before any use |
| `wbc_fsm` | medium for controller lifecycle, ONNX contract checks, state history, projected-gravity termination and safe fallback | no: G1 29-DoF/Unitree SDK2, no top-level licence found, and no soccer perception or tactics | use as architecture-only evidence for Apollo `SkillExecutor`; do not replace Apollo or copy source/models |

`wbc_fsm` is not a football team base. Its manually selected Passive,
FixedStand, Loco, AMP/MJAmp and WBC states provide a compact deployment example,
but replacing Apollo with it would discard the behavior tree, team decision
interfaces and current guarded execution path.

PAiD is materially more relevant than the prior survey recorded. The official
repository now publishes training code, thirteen labeled motion files and a G1
ONNX checkpoint. Its CC BY-NC 4.0 licence permits attributed non-commercial
research use but is not treated as a permissive software licence. Source,
motions and weights remain outside this repository unless a separate licence
review explicitly changes that rule.

## Why the current R1 route must change

The current experiments were useful falsification work, but they over-invested
in selecting fixed open-loop trajectories before acquiring a stable kick
motion skill:

- the five-action exact-CPU bank has an oracle success rate of 928/1023
  (`90.71%`), yet the best deployable current-state selector reaches only
  153/205 (`74.63%`) on its frozen validation split;
- adding privileged trigger values reaches 146/205 and a 50-frame causal
  history reaches 147/205, so missing a few trigger features is not a sufficient
  explanation;
- a continuous outcome regressor fits 90% of its training rollouts but selects
  only 128/205 (`62.44%`) successful validation actions;
- on a separate frozen 256-rollout long-horizon set, the fixed 5 m zero-residual
  prior reaches 172/256 (`67.19%`), while residual-scale 0.1 and 0.5 PPO
  candidates each reach 171/256 (`66.80%`), with zero falls. Neither is
  promotable.

The engineering foundations remain valid and should be retained: strict
contact-plus-target-plus-arrival-speed success, an independent CPU evaluator,
formal Warp/CPU parity, frozen evaluation sets, zero-fall gating and explicit
rejection of regressions. The invalid assumption was that Apollo walking plus a
short fixed keyframe bank constituted a sufficiently smooth motion prior for a
long-horizon approach-and-kick policy.

PAiD directly addresses this failure mode: acquire a unified reference-tracking
skill first, preserve most motion rewards when ball perception is introduced,
and only then widen position, ball-motion and physics distributions. Its
high-speed shot reward must not be copied literally: this project also requires
controlled pass arrival speed, so reward must track a commanded ball trajectory
instead of monotonically preferring speed.

## Corrected R1 execution route

### K0: provenance and T1 reference construction

1. Keep the PAiD clone and all CC BY-NC assets outside the project repository.
2. Parse all thirteen motion files, preserving fps, body transforms and
   `kick_leg` labels.
3. Build a reproducible G1-29 to T1-23 mapping through GMR/IK, then optimize
   root, feet and contact against exact RCSS geometry rather than dropping hand
   joints alone.
4. Validate joint limits, finite derivatives, support/contact timing,
   non-foot collisions, balance recovery and mirrored left/right labels.
5. Hash every input, mapping, command and local output. Failed retargets cannot
   enter K1.

The released G1 ONNX is useful only as an upstream reproduction and visual
reference. Its weights cannot pass the T1 contract gate.

### K1: motion-skill acquisition

Train a reference-centred recurrent or privileged teacher whose zero residual
tracks the T1 motion phase. Use adaptive motion-by-phase failure sampling,
reference/root/body/foot/velocity rewards, terrain/contact perturbations and
safety termination. Ball perception noise and pass-range randomization remain
off until held-out motion tracking, contact timing and recovery gates pass.

### K2: perception-action integration

Resume K1 while retaining whole-body, velocity and foot tracking. Relax global
anchor position, add egocentric ball/target observations, randomized arc and
rolling-ball starts, correct-foot one-time contact, post-contact stability and
a short target-trajectory window. Freeze the proximity reward after first
contact so it cannot reward chasing the departed ball. For passes, reward
direction and commanded arrival speed; for shots, select a distinct high-speed
command rather than sharing one objective.

### K3: robustness and deployment

Randomize friction, restitution, mass, PD gain, delay and observation noise
using RCSSServerMJ measurements. Require Warp/exact-CPU parity, three seeds,
frozen 2/3.5/5 m evaluation, source/ONNX parity and server replay before an
artifact can replace the default-off fallback. Distill history only after the
privileged teacher passes these physics gates.

### Runtime boundary

Apollo remains the match runtime. Adopt only the `wbc_fsm` concepts that
strengthen the existing executor: explicit lifecycle transitions, projected-
gravity/finite-output termination, exact model input/output validation,
timeout and deterministic fallback. Policy state never owns tactics, pass
selection or role allocation.

## Stop rules

- Do not spend another formal run on the current fixed-prior PPO objective.
- Do not promote a model on online-batch metrics; the frozen exact-CPU set is
  decisive.
- Do not install Isaac Lab merely to resemble an upstream recipe on the local
  8 GB GPU. Reimplement the validated method in the accepted MuJoCo/Warp stack.
- Do not vendor PAiD or `wbc_fsm` source/assets under the project's existing
  distribution assumptions.
- Do not expand to shot/clear/team tactics until K1 motion tracking and K2
  controlled 2 m pass gates are closed.
