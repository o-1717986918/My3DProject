# Open and reliable strategy search: T1 running and robot soccer

Search cutoff: 2026-08-31

This record is the decision artifact for the next training stage. It separates
source reliability from project applicability, records negative results, and
defines the gates that must be passed before a new locomotion policy can replace
the competition fallback.

## Executive decision

The next experiment must not be another reward-only continuation of the current
walking policy. The best evidence-supported, implementable route on this
project's 8 GB GPU is:

1. verify identical-action CPU MuJoCo versus MJWarp short-horizon parity;
2. construct a periodic, dynamically consistent T1 running reference on the
   exact RCSS model;
3. change control from a fixed nominal-pose action to a reference-centred
   residual action;
4. train whole-body motion tracking with adaptive failure-phase sampling;
5. add train-time bilateral symmetry, then velocity-command fine-tuning;
6. begin ball interaction only after deterministic CPU and ONNX running gates
   pass.

The decisive control equation is:

```text
q_target(t) = q_reference(phase, speed) + residual_scale * action_residual(t)
```

The current trainer instead applies the learned action around a fixed nominal
pose. That forces one network to reproduce the entire reference and correct
dynamics error at the same time. FC Portugal's exact RoboCup 3D running task,
RuN, and the current robot-rl running work independently support separating a
motion generator/reference from a smaller residual controller.

## Search protocol

The search covered original papers, official repositories, released
checkpoints or trajectories, project documentation, and community discovery
channels. A community post or video was used only to discover a candidate; no
performance claim was accepted until an original paper or official artifact
could be found.

Evidence classes:

- **A**: official source plus a runnable artifact, locally inspected or
  replayed;
- **B**: original paper plus official implementation, but a robot/backend gap
  or a missing trained artifact remains;
- **C**: original paper or official project page without a reproducible
  implementation;
- **D**: community lead, incomplete repository, unclear provenance, or another
  material reproducibility gap.

Applicability is evaluated independently:

- **direct**: T1 and close to RCSS/MuJoCo;
- **adapt**: an algorithm can be independently reimplemented in the present
  stack;
- **later**: useful only after the locomotion gate;
- **reject**: no current implementation value or unacceptable provenance.

## Candidate matrix

| Source | Evidence | Robot / backend | Public artifact | License finding | Applicability and decision |
|---|---:|---|---|---|---|
| MuJoCo Playground current T1 joystick task | A | T1 / MJWarp and MuJoCo | experimental ONNX policy | Apache-2.0 | **Direct baseline only.** Replayed locally; it is stable walking, not running. Adopt observation, contact and testing patterns, not the policy as a run teacher. |
| Booster Gym | A | T1 / Isaac Gym, MuJoCo deployment | lower-body checkpoint | Apache-2.0 | **Direct baseline only.** Useful for T1 geometry and sim-to-sim checks; local audit found no sustained flight. |
| Holosoma | A | T1 / MJWarp | retargeting and tracking code | Apache-2.0 | **Adopt.** Remains the best open T1 retargeter. Keep the pinned project patch and local-only LAFAN-derived data. |
| Chasing Autonomy / `robot_rl` | B | G1 / Isaac Lab | running configs, library consumer, policies and trajectory files | repository has no top-level LICENSE; metadata says MIT while several files say all rights reserved; trajectory repository has no declared license | **Adapt method only.** Do not copy code or distribute trajectories. The whole-body multiple-shooting generator described in the paper was not found in the repository; the CasADi trajectory optimiser present there is navigation MPC. |
| FC Portugal running task | B | NAO / SimSpark RoboCup 3D | complete training task | GPL-3.0 | **High-value method reference.** Its analytic step generator plus learned residual is the closest competition precedent. Reimplement the pattern independently to preserve project licensing. |
| RuN | C | G1 / Isaac Gym | no official code located | paper license only | **Method corroboration.** Frozen conditional motion generator plus residual policy supports the selected architecture, but training a generator is deferred. |
| BeyondMimic | B | G1 / Isaac Lab | official tracking code | MIT | **Adopt algorithms.** Whole-body tracking, exact reference reset, termination rules and adaptive failure-state sampling are implementable in the current trainer. |
| MJLab whole-body tracking | B | G1 / MJWarp | maintained training implementation, no checkpoint | Apache-2.0 | **Strongest backend reference.** Independently port its adaptive sampling and reward structure; do not migrate the project framework. |
| RSL-RL symmetry extension | B | generic / PyTorch | official PPO symmetry implementation | BSD-3-Clause | **Adopt concept.** Implement train-time mirror loss first; enable mirrored transition augmentation only after next-state equivariance tests. |
| Daffan humanoid striker | B | T1 / Isaac Gym | teacher/student curriculum code, no directly deployable RCSS policy | Apache-2.0 | **Adopt curriculum.** It remains the main T1 soccer progression reference after running passes. |
| DribbleMaster | B | T1 / Isaac Gym | locomotion and dribbling configs, no checkpoint | root MIT with inherited all-rights-reserved notices | **Later, method/config only.** Useful for Stage-II ball curriculum, delay, field-of-view and symmetry design; not a reproducible trained solution. |
| RoboNaldo / PAiD | B/C | G1 / Isaac Gym | staged kick projects | mixed; PAiD has non-commercial restrictions | **Later.** Use their motion-prior-to-task sequence, never their G1 motion as a T1 run reference. |
| SoccerDiffusion | B | heterogeneous humanoid football data | official code/data pipeline, no T1 policy | MIT | **Later.** Potential team-motion data architecture, not an immediate locomotion asset. |
| STOFT | C | humanoid kick | paper and project page | no reusable implementation located | **Later.** Foot-trajectory optimisation is relevant once stable locomotion and ball state estimation exist. |
| HTWK T1 policies | D | T1 / MuJoCo replay | incomplete replay release; one model duplicates Booster bytes | no top-level license located | **Reject for redistribution and training provenance.** Retain only as a local black-box comparison. |
| community and Bilibili results | D | mixed | mostly demonstrations or reposts | usually unclear | **Discovery only.** No candidate exceeded the evidence available from the official sources above. |

## Local artifact audit

### Official MuJoCo Playground T1 policy

The current upstream T1 joystick task was pinned at commit
`8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`. Its experimental ONNX policy has
SHA-256
`3f25f2b9c3dd49bab915577f7d5caba4e753a6cb29df2d7488ef7da14d1966b4`.
The following fixed-command runs used the official flat-terrain T1 model and
CPU MuJoCo for ten seconds:

| commanded vx (m/s) | achieved vx (m/s) | absolute lateral displacement (m) | survived | >=10 ms flight |
|---:|---:|---:|:---:|:---:|
| 0.4 | 0.372 | 0.744 | yes | no |
| 0.6 | 0.546 | 1.323 | yes | no |
| 0.8 | 0.730 | 1.086 | yes | no |
| 1.0 | 0.862 | 0.875 | yes | no |

The official task also commands at most 1.0 m/s forward and uses a
default-pose-plus-action controller. It is therefore a valuable stable-walk
reference and regression case, but it cannot satisfy this project's running
definition. Rough-terrain spot checks did not change that conclusion and were
unstable at the high end of the command range.

### Other released T1 policies

The Booster Gym lower-body policy is an official, licensed T1 artifact and is
kept as a geometry and deployment reference. Prior exact-CPU project audits
also found stable walking but no repeated aerial phase. The HTWK base model is
byte-identical to the Booster model, while its additional release lacks a
complete replay path and a license. Neither is promoted as a run teacher.

## What the strongest sources change

### 1. Reference-centred residual control

The selected design preserves a deterministic reference at zero policy output.
This sharply reduces the first task the policy must solve and creates a useful
diagnostic: an untrained residual policy should replay the reference rather
than command the nominal standing pose. Residual scale starts small and grows
only if reference tracking saturates.

This is deliberately narrower than training a conditional motion generator.
RuN demonstrates that such a generator can support smooth walk-run transitions,
but building the generator and its motion corpus would add a second difficult
training problem before the first running gate is solved.

### 2. Periodic, dynamically plausible reference

Chasing Autonomy reports a library produced from one human running stride by
whole-body multiple-shooting optimisation with explicit single-support and
flight domains, friction/contact constraints, periodic endpoints, and
half-cycle reflection. The published `robot_rl` repository can consume the
resulting library, but its whole-body generator is not available there and the
repository/trajectory licensing is not safe for reuse.

The project will therefore implement the smallest independent analogue on the
exact T1 model:

- preserve the imported run's flight and sagittal timing;
- enforce matching cycle endpoint joint positions and velocities;
- enforce half-cycle left/right reflection;
- minimise stance-foot world velocity and penetration;
- bound root yaw change, lateral displacement, joint limits and control
  discontinuity;
- replay the projected cycle in CPU MuJoCo and record contacts before training.

A full CLF-conditioned trajectory library is a later extension, not a
prerequisite for the first accepted fixed-speed runner.

### 3. Whole-body tracking and hard-phase sampling

Joint tracking alone does not constrain root heading, foot timing or the
ballistic phase strongly enough. The next environment needs reference-relative
terms for root position/orientation and linear/angular velocity, feet, joint
position/velocity and exact/proxy contact timing.

BeyondMimic and MJLab provide a practical curriculum: split a motion into phase
bins, maintain an exponential moving average of failure frequency, smooth it
across nearby earlier bins, mix it with a uniform distribution, and sample
resets from the resulting distribution. This targets failed take-off and
landing phases without permanently forgetting the rest of the cycle.

### 4. Symmetry during training

The current reflection map and involution tests are retained. Inference-time
left/right averaging is closed as a deployment route because the local
experiment reduced turning while also erasing the already weak flight phase.
The next experiment applies a mirror-consistency loss to policy outputs during
PPO. Mirrored transition augmentation is enabled only after a one-step
environment test shows that reflection commutes with dynamics within a stated
tolerance.

### 5. Simulator parity before scale

MuJoCo's CPU, MJX and MJWarp implementations are not expected to be bitwise
identical. A short-horizon, identical-action trace must nevertheless identify
whether divergence begins in contacts, action decoding, root state or solver
integration. Long training remains blocked if the same initial state and
action stream immediately disagree on take-off or stance-foot identity.

## Selected implementation stages and gates

The machine-readable counterpart is
`training/locks/open_strategy_2026_08_31.yaml`.

### R0: CPU/MJWarp parity instrument

Record an identical initial state, phase, reference frame and action sequence
in both backends. Compare decoded joint targets, torso quaternion and angular
velocity, root displacement/yaw, foot height and left/right contact state at
every policy step. The trace must identify the first mismatch and be usable as
a regression test. No claim of long-horizon equality is required.

### R1: periodic T1 reference

The projected reference must pass schema, finite-value, joint-limit,
periodicity, half-cycle reflection and provenance checks. CPU replay must have
no non-foot pitch collision, at least one two-consecutive-5-ms two-foot flight
interval per cycle, bounded yaw/lateral motion, and a reported stance-foot slip
metric. A visually plausible animation is insufficient.

### R2: fixed-speed residual tracker

Train first at 1.6--2.0 m/s. Reset to the reference state with bounded noise.
Zero residual must reconstruct the selected frame. Promote only a three-seed
candidate that survives ten seconds, is finite, retains an aerial phase in at
least 80% of exact-CPU episodes, reaches at least 1.2 m/s, has median tracking
RMSE at most 0.35 m/s, and median lateral drift at most 0.25 m. Reference pose,
velocity, foot and contact errors must also be reported.

### R3: adaptive phase curriculum and symmetry

Add failure-bin reset sampling and whole-body terms before domain
randomisation. Then add train-time mirror loss. Each addition is an ablation
against the same seeds and evaluation set; it is retained only if it improves
exact-CPU gate margin without reducing aerial-phase reliability.

### R4: command and robustness fine-tuning

Expand speed gradually, then lateral/yaw commands, transitions from standing
and walking, pushes, friction/PD variation, observation noise and action delay.
Export ONNX at immutable checkpoints and require native-versus-ONNX action and
rollout parity.

### R5: football and team closure

Only after R4 passes: restore ball approach and kick transitions, train a
privileged T1 ball policy before vision/student adaptation, run deterministic
single-player RCSS acceptance, then headless and visual 7v7. The current stable
walk remains the fallback throughout.

## Compute policy for the local machine

The 8 GB RTX 5060 Laptop GPU is sufficient for the selected single-policy
tracking path, but not a reason to begin a motion-generator or diffusion stack.
Use short parity traces and 256--512 environment smoke tests first, then the
largest MJWarp batch that leaves memory headroom. Keep fixed-speed training and
ablation runs small enough to preserve three independent seeds. Store
checkpoints, manifests, source hashes and exact-CPU evaluation together; never
select a checkpoint from training reward alone.

## Closed and deferred branches

- **Closed:** more flight reward on the existing fixed-nominal action. It
  improved a proxy metric without producing exact-CPU flight.
- **Closed:** inference-time mirror averaging as the production symmetry
  mechanism. It suppressed the desired motion as well as drift.
- **Rejected:** treating any released T1 walking ONNX as a run teacher.
- **Rejected:** copying `robot_rl` source or trajectory assets while its
  licensing remains inconsistent or undeclared.
- **Deferred:** conditional motion generation, AMP/selective AMP, diffusion
  football control and CLF libraries. They become rational only after the
  fixed-speed residual tracker passes.
- **Deferred:** framework migration to Isaac Lab. The current exact RCSS model
  and MJWarp route are closer to deployment truth.

## Primary sources

- [Chasing Autonomy paper](https://arxiv.org/abs/2603.25902) and
  [official robot_rl repository](https://github.com/Zolkin1/robot_rl)
- [FC Portugal running paper](https://arxiv.org/abs/2312.14360) and
  [official FCPCodebase](https://github.com/m-abr/FCPCodebase)
- [RuN](https://arxiv.org/abs/2509.20696)
- [BeyondMimic](https://arxiv.org/abs/2508.08241) and
  [official implementation](https://github.com/HybridRobotics/whole_body_tracking)
- [MJLab](https://github.com/mujocolab/mjlab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- [Holosoma](https://github.com/amazon-far/holosoma)
- [Daffan humanoid striker](https://arxiv.org/abs/2512.06571)
- [DribbleMaster](https://arxiv.org/abs/2505.12679)
- [SoccerDiffusion](https://arxiv.org/abs/2504.20808)
- [MuJoCo MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
  and [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/)

