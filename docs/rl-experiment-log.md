# RL implementation and experiment log

Date: 2026-08-30
Host: WSL2 Ubuntu 22.04, RTX 5060 Laptop GPU (8 GB), 15 GiB WSL RAM
Purpose: establish a reproducible training path; no smoke checkpoint is a
competition model.

## Reproducible runtime

The `my3d-rl` conda environment uses Python 3.12.  Source and package pins are
in `training/locks/sources.yaml` and `training/locks/my3d-rl-pip.txt`.

The working compatibility set is:

- MuJoCo Playground 0.2.0 at commit
  `124a73fa3303f75a62f8fe04d329b829ed0ebdfb`;
- Brax 0.14.2, JAX/JAX CUDA plugin 0.6.2, and Flax 0.11.2;
- MuJoCo/MJX 3.12.0 and Warp 1.16.0;
- RCSSServerMJ 0.2.1 competition resources, identified by SHA-256 in the
  source lock.

JAX 0.11.1 was rejected after the first real PPO call: Brax 0.14.2 still uses
`jax.device_put_replicated`, which JAX 0.11.1 removes.  JAX 0.6.2 is the version
recorded by the pinned Playground 0.2.0 `uv.lock`, and the final environment
passes `pip check`.

## Playground backend benchmark

Task: upstream `T1JoystickFlatTerrain`, zero-action batched rollout on the GPU.
The two runs use the final JAX 0.6.2 lock.

| Backend | Environments | Timed steps | Compile + first step | Rollout | Throughput | Reported allocator peak |
|---|---:|---:|---:|---:|---:|---:|
| MJX-JAX | 512 | 20 | 107.234 s | 55.484 s | 184.6 env-steps/s | 78,945,280 B |
| MJX-Warp | 512 | 100 | 12.357 s | 5.815 s | 8,804.8 env-steps/s | 69,329,152 B |

Warp was approximately 47.7 times faster for the timed rollouts on this host,
so it is the primary local physics backend.  JAX remains the compatibility
fallback.  The allocator values come from JAX and do not prove total Warp or
driver VRAM usage; full-run VRAM must also be sampled with `nvidia-smi`.

## RCSS-physics kick environment

`DirectionalKick` is built from the installed RCSSServerMJ soccer world and T1
XML rather than a visually similar replacement.  It uses:

- 0.005 s physics steps and four substeps per 50 Hz action;
- the competition ball mass, radius, and friction;
- the competition position-plus-velocity PD actuator protocol;
- a 96-value deployable actor observation, 106-value privileged critic state,
  and 23 residual joint targets;
- per-joint action scales and explicit joint-limit clipping;
- randomized fixed-ball position and shot heading for curriculum stage one.

Both MJX-JAX and MJX-Warp reset/step tests produced finite rewards with the
expected shapes.  Direct CPU MuJoCo scene tests also verify the exact asset
assembly, ball parameters, timestep, and actuator mapping.

## PPO integration result

Final smoke run:

```text
run: /home/win98/rl_runs/kick-smoke-20260830-04
backend: GPU / Warp
parallel environments: 64
environment steps: 4096
total elapsed: 28.686 s
compiled training wall time: 17.549 s
training throughput: 233.400 env-steps/s
checkpoint: checkpoints/000000004096 (commit_success.txt present)
```

The run completed a genuine PPO optimizer update with separate actor and
privileged critic inputs.  The final scalar loss was finite.  This proves the
reset -> rollout -> advantage -> optimizer -> checkpoint path; one update is
not evidence that the policy can kick.

The saved checkpoint was loaded back through Brax after the run.  It contained
the normalizer, actor, and critic parameter groups (25 array leaves), and every
numeric leaf was finite.

Initial randomized rollouts showed that `naconmax=256` dropped contact
candidates, with a measured peak request of 928.  The environment now reserves
1024, and the final run log contains no broadphase/narrowphase overflow,
traceback, or segmentation message.

A 256-environment attempt exited during compilation without a Python
traceback or checkpoint.  The cause is not established.  On this 8 GB WSL
host, 64 environments are therefore the supported starting point; higher
parallelism must earn support through an isolated VRAM/driver stress test.

## Commands

From the repository root in WSL:

```bash
conda activate my3d-rl
export PYTHONPATH=training
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python training/tools/verify_playground.py --impl warp --num-envs 512 --steps 100
python training/tools/smoke_kick_env.py --impl warp --num-envs 16
python training/tools/train_kick_smoke.py \
  --impl warp --num-envs 64 --num-timesteps 4096 \
  --run-dir /home/win98/rl_runs/kick-smoke-$(date +%Y%m%d-%H%M%S)
```

## Gate status

Passed now:

- pinned GPU training runtime and source provenance;
- exact competition asset assembly and 96/23 interface contract;
- JAX and Warp environment compilation/step;
- real PPO update and persistent checkpoint;
- checkpoint reload with finite actor/critic/normalizer parameters;
- no contact-buffer overflow in the final smoke run.

Final automated checks: 17 training-side tests and 29 competition-runtime
tests pass.  The custom environment also resets and steps with finite rewards
on both JAX and Warp under the final dependency lock.

Not passed yet:

- policy learning curve or actual kick performance;
- three-seed, 200-held-out-episode acceptance;
- high-speed MuJoCo/MJX versus RCSSServerMJ trajectory bounds;
- ONNX export and source-versus-ONNX parity;
- competition client feature-flag integration and full match regression.

## Fast-locomotion experiments — 2026-08-30/31

The running work uses the exact RCSSServerMJ soccer/T1 assets, a 0.005 s
physics step, 50 Hz control, the deployed joint order/sign table, and an ONNX
teacher whose SHA-256 is
`ece316886c2a4cde17402f0332f0f955a83db786490cf6d50cfc680e5e30434b`.
Generated checkpoints and reports remain under `/home/win98/rl_runs`.

### Findings that changed the implementation

- Randomly initialized tanh and normal PPO policies failed before learning a
  reliable stand. The current T1 soccer repositories instead start from a
  locomotion/motion prior, so all subsequent runs use verified teacher
  warm-start and low-noise PPO.
- The training sign for `Left_Ankle_Pitch` was opposite to the competition
  runtime. It is now `+1`; a regression test also checks the YAML table.
- The former 5 cm foot-site threshold was not a valid contact test. It was
  replaced by the oriented foot-box lowest-point test with 1 cm tolerance,
  following current Booster/T1 implementations. CPU calibration found zero
  cases where a true foot contact was labelled airborne.
- A global flight reward of 5 destabilized the teacher and still produced no
  real aerial phase. That result is retained as a negative experiment, not
  promoted.

### Reproducible result table

| Run/checkpoint | Purpose | Held-out result at `vx=1.5` | Decision |
|---|---|---|---|
| `run-teacher-flight-s53.../000011010048` | v1 teacher fine-tune, 11.01M steps | 64/64 upright, 1.635 m/s, 10.47 m drift; CPU flight 0% | reject as run; fast-walk baseline only |
| `run-yaw-flight-s59.../000009437184` | high yaw/flight reward, paused at 9.44M | 64/64 upright, 1.617 m/s, 9.86 m drift; CPU flight 0% | reject |
| `run-yaw-flight-s59-resume.../000001572864` | resumed final epoch | variable-command stability fell to 78%; 10.14 m drift | reject |
| `run-phase-v2-smoke-s67.../000000196608` | 80-value phase interface smoke | 100% upright; ONNX parity `1.14e-5` | pipeline pass only |
| `run-phase-v2-formal-s71.../000003538944` | phase-aware v2, selected middle checkpoint | 64/64 upright, 1.494 m/s, 0.092 m/s RMSE, 5.66 m MJX drift | best phase checkpoint; not releasable |
| same checkpoint, CPU ONNX | exact contacts and export | 32/32 upright, 1.499 m/s, 5.45 m drift, flight 0%; parity `1.43e-5` | reject as run |
| `run-phase-v2-straight-s73.../000002359296` | strong fixed-command yaw/lateral recovery | 64/64 upright, 1.504 m/s, 4.77 m drift | improvement, still reject |

The phase-aware model reduced median ten-second drift by about 52% relative to
the v1 paused candidate while preserving speed, but it remains roughly 19
times above the 0.25 m release gate and never sustains a 10 ms aerial phase.
The honest label is **stable high-speed walking research candidate**.

### Next experiment boundary

Do not raise the flight reward again. The next run must start from a retargeted
T1 running reference or a demonstrably airborne T1 motion prior, train a
tracking policy first, and only then add velocity/football task rewards with a
small exploration standard deviation. Holosoma is the preferred Apache-2.0
T1/MJWarp retargeting path; RoboNaldo supplies the staged curriculum pattern.

## Holosoma motion-prior implementation — 2026-08-31

### Source and licence boundary

Holosoma was cloned outside the repository at
`fb835ec8cb6ee48f483ce567586625e5fae1ae1f` into the isolated Python 3.11
`my3d-motion` environment. The official LAFAN1 archive is also external; its
SHA-256 is
`ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977`.
LAFAN1 is CC-BY-NC-ND-4.0, so neither source nor derived T1 NPZ data is
committed or redistributed.

Three upstream defects were reproduced and fixed by the pinned project patch:

- the robot-only path wrote a dummy seven-value object pose over T1's final
  seven joints;
- LAFAN extraction persisted BVH left-first joint order while the retargeting
  registry consumed right-first order, including a `LeftToe`/`LeftToeBase`
  alias mismatch;
- MuJoCo 3.12 returns an integer joint-type value for which enum tuple
  membership failed, producing zero hinge columns in the qdot-to-qvel
  Jacobian.

`git apply --unidiff-zero --check --reverse` passes against the patched pinned checkout. The
fixed hinge transform norm is `sqrt(23)=4.7958315`, and optimization cost fell
from roughly 8 to 1–2 with non-zero limb motion.

### Imported references

| Local-only reference | Frames/duration | Exact RCSS replay | Decision |
|---|---:|---|---|
| `t1_run2_subject4_f1940_2030_v1.npz` | 151 / 3.0 s | both feet contact; longest flight 0.20 s; no non-foot pitch contact | validated parent, SHA `4a3beb70...` |
| `t1_run2_subject4_cycle71_87_v1.npz` | 28 / 0.54 s | flight 0.06 s; speed 1.93 m/s | rejected for straight training: mean root yaw about 0.50 rad/s |
| `t1_run2_subject4_straight76_109_v1.npz` | 34 / 0.66 s | contacts `[10,10]`; flight 0.16 s; speed 3.20 m/s; no bad collision | accepted reference input, SHA `2ad29433...`; not itself a learned policy |

The importer uses quaternion slerp, canonical forward orientation, exact T1
foot-box grounding and CPU MuJoCo contact replay. The validator requires exact
dtypes/shapes, 50 Hz, per-foot stance, a bounded aerial interval, lower-body
excursion, safe height/vertical speed, no non-foot pitch contact and at most
15 mm penetration.

### Training results

| Run | Key change | Result | Decision |
|---|---|---|---|
| `run-motion-track-formal-s83...` | first joint/action/contact prior | survival 365→268/500; final falls 84.4%; KL peak 0.059 | reject; Brax default adaptive-KL minimum silently raised LR to `1e-5` |
| `run-motion-track-v3-smoke-s89...` | explicit `2.5e-7..2e-6` LR bounds; every reset from full reference | survival 111→77; final falls 93.8% | reject; curriculum start too hard |
| `run-motion-track-v3-curriculum-s97.../000000786432` | normal-state initialization, bounded KL, fall -100 | MJX survival 444→490/500, falls 28.1%→4.7% | useful prior-reward improvement; manifest confirms reference-init probability was 0.0; CPU gate still rejects |
| same exported ONNX, exact CPU, contract scale 0.5 | 64 episodes at 1.8 m/s | upright 87.5%, speed 1.865 m/s, RMSE 0.129, drift 7.18 m, flight 12.5% | reject |
| `run-motion-straight-v3-s101...` | low-yaw reference, zero-tolerance contact proxy, stronger straight constraints | five CPU checkpoints: drift 7.41–7.87 m, flight 0–15.6% | reject; reward tuning did not remove turning mode |
| checkpoint `000000786432` with reflection ensemble | exact left/right equivariance at inference | CPU 64/64 upright, 1.851 m/s, drift 1.15 m, flight 0% | diagnostic improvement only; reject |
| reflection ensemble with scale 0.45 | decoder stress test | 64/64 upright, about 1.92 m/s, drift 1.43–1.50 m, flight 0% | reject; scale override not adopted |

The ONNX export parity maximum error for the curriculum checkpoint is
`1.10e-5`. The CPU evaluator now defaults to the contract action scale instead
of a stale hard-coded 0.45 and labels any CLI override. With zero contact-proxy
tolerance, its proxy confusion report showed zero false positive/negative
frames for the symmetry test, yet no episode sustained the required 10 ms
two-foot aerial interval.

### Current gate status

The project now has a fully reproducible and licence-aware motion ingestion,
tracking curriculum, ONNX export, symmetry diagnostic and exact-contact CPU
acceptance loop. It does **not** have a release running policy. No experimental
ONNX was copied into `mujococodebase/skills/walk/`; the existing stable walk
remains the competition default.

## CPU MuJoCo versus MJX-Warp parity baseline — 2026-08-31

The new `training/tools/compare_cpu_mjwarp.py` harness copies one complete MJX
initial state into CPU MuJoCo, replays the same bounded action sequence through
the environment's single action decoder, and records per-control-step targets,
reference phase, joint/root/torso state, foot lowest point and contact proxy.
It reports each metric's first threshold crossing rather than expecting the
solvers to be bitwise identical over a full episode.

Three RTX 5060 Laptop GPU checks passed:

| Trace | Largest root-position error | Largest joint-position error | Contact-proxy mismatch | Gate |
|---|---:|---:|---:|:---:|
| five neutral steps | `6.54e-7 m` | `2.03e-6 rad` | 0 frames | pass |
| twenty 0.15-amplitude sine-action steps | `3.07e-6 m` | `5.88e-7 rad` | 0 frames | pass |
| ten steps from the complete local run-reference state | `1.60e-6 m` | `7.31e-7 rad` | 0 frames | pass |

All decoded-target errors were exactly zero, every tracked value was finite,
and no orientation, yaw or foot-height threshold was crossed. The locked
machine-readable summary is
`training/locks/sim_parity_baseline_2026_08_31.yaml`; full JSON traces remain in
`/tmp`. This closes the immediate decoder/initialisation/basic-integrator
mismatch hypothesis. It does not replace long-horizon exact-CPU acceptance for
a trained, contact-rich policy.

## Periodic contact-aware T1 reference — 2026-08-31

R1 projects the accepted 34-frame straight LAFAN/Holosoma slice onto an exact
bilateral half-cycle. A blind 50/50 average produced excessive stance slip,
while independent frame grounding broke contact symmetry and the cyclic root
velocity seam. The accepted deterministic setting retains 80% of the selected
source half, applies four circular joint-smoothing passes, reprojects the root
after exact grounding, and performs one bilateral least-squares stance
correction. This is a recorded correction to the initial branch, not a hidden
manual edit.

The first passing artifact exposed a semantic gap in those geometric gates:
its mean root yaw was approximately `-pi` while world velocity was +X, so its
initial body-local velocity was negative. It represented a backward-labelled
run despite passing symmetry, contact and continuity. The v2 projector now
measures source motion in the body frame, time-reverses this clip, rotates the
mean heading to world +X and requires the absolute yaw centre to be near zero.
At a 1.8 m/s command the corrected reset measured +1.866 m/s body-local
forward velocity rather than -2.118 m/s. The earlier artifact is superseded.

The generated NPZ remains local-only under the source dataset's
CC-BY-NC-ND-4.0 boundary. Its SHA-256 is
`ab81912570d746965162f1d84cfd6d215a1265bd28dfc2d371c72f095aa40f9a`;
the parent SHA-256 is `2ad294330d7d7fc19e236169bdc862079c8228fd38a544703c38f698fee09820`.

| R1 measurement | Result |
|---|---:|
| cycle | 34 frames / 0.68 s / 2.172 m |
| exact foot-contact frames | left 11, right 11 |
| longest exact aerial interval | 6 frames / 0.12 s |
| half-cycle joint/root/orientation/contact error | 0 / `2.22e-16 m` / 0 / 0 |
| joint seam / internal p95 step | 0.306 / 0.250 rad |
| root-velocity seam / largest cyclic step | 0.225 / 0.856 m/s |
| stable-stance slip mean / p90 / max | 0.125 / 0.262 / 0.292 m/s |
| yaw centre / deviation / lateral excursion | `5.62e-19` / 0.094 rad / 0.0294 m |
| non-foot pitch-contact frames / joint-limit violation | 0 / 0 rad |

Stable-stance metrics use exact contact runs of at least four frames. Shorter
collisions remain visible in exact replay and contact counts, but do not become
support-foot anchors. Twenty-step CPU/MJWarp replays through the actual v3
moving-reference decoder passed for both zero and full-amplitude sine
residuals. Both had target error 0 and contact-proxy mismatch 0; their largest
root errors were `2.21e-6 m` and `4.48e-6 m`. The machine-readable record is
`training/locks/periodic_reference_baseline_2026_08_31.yaml`.

This closes corrected R1 and the R2 interface gate, not policy learning. The
new v3 contract maps zero residual to the periodic reference at every phase,
uses reference-relative joint observations, scales cadence and reference
velocities to the commanded forward speed, and preserves v1/v2 unchanged.

## Reference-centred residual training — 2026-08-31

The first optimizer run,
`run-reference-residual-v3-smoke-s107-20260831-1215`, completed a real
checkpoint at 196,608 environment steps. It is rejected: the default
tanh-normal actor head did not initialise at zero residual, evaluation mean
episode length fell from 56.4 to 43.8 control steps, and the one-epoch KL was
8.61. The requested 4096 steps had been silently rounded by Brax to its
196,608-step minimum batch, so the trainer now records and passes an explicit
effective step count.

The run also exposed that the generic motion evaluation environment forced
reference initialisation probability to zero. That is useful for later
normal-start robustness, but invalid for the first fixed-reference tracking
gate. `reference_residual_v1` is retained unchanged for reproducibility. The
replacement `reference_residual_v2` uses an exactly zero mean head, 0.1 initial
standard deviation, one PPO pass, `1e-5` learning rate with bounded adaptive
KL, no observation renormalisation, and reference-state evaluation. A
regression test asserts the initial mean and standard deviation directly.

The corrected seed-109 smoke kept its distribution controlled (KL
`3.80e-4`, mean location within `6.77e-4`, standard deviation 0.100) and wrote
a valid checkpoint, but mean episode length only changed from 20.0 to 20.4
steps. A new zero-residual diagnostic then separated reference trackability
from optimisation. Across 32 random phases, the 1.8 m/s scaled cycle survived
20.3 steps on average; the original 3.195 m/s cadence survived 19.7. Neither
completed an episode. Time scaling is therefore not the primary failure;
the kinematic reference needs learned feedback/inverse-dynamics correction.
The locked results are in
`training/locks/reference_residual_baseline_2026_08_31.yaml`.

The full seed-113 v2 run then trained for 2,359,296 environment steps. Its
deterministic evaluation trajectory was 20.94 steps initially, 18.97, 19.62,
19.19, 18.31, a temporary 22.53 at checkpoint 1,966,080, and 19.97 finally.
Every final evaluation episode fell. KL remained controlled and ended at
0.00258, while the deterministic mean stayed within normalized action bounds
[-0.0713, 0.0436]. The optimizer is functioning, but this low-noise one-pass
profile produced no sustained learning improvement and is rejected.

An exact CPU diagnostic next separated PD phase lag from dynamic feasibility.
Across 32 uniformly spaced start phases at the source cadence, the raw
reference survived 20.38 steps on average. Scanning target leads from -2 to +3
frames peaked at only 21.56 steps. A MuJoCo inverse-dynamics target raised the
mean to 22.84 and the maximum to 63, but still completed zero 200-step
episodes. Its bounded target correction had mean absolute size 0.0614 rad,
p90 at the 0.15 rad limit and 24.3% saturation. More decisively, the prescribed
floating-base trajectory required p90 593 N and a maximum 1537 N of unavailable
generalized force. The inverse target is retained as a local diagnostic
sidecar, not promoted into the policy contract.

Inspection of the pinned official BeyondMimic source identified a controlled
optimizer ablation that the local v2 run has not yet tested: five PPO passes,
larger exploration and adaptive KL, while keeping the deterministic actor mean
exactly zero. `reference_residual_v3` implements the first two changes with
initial standard deviation 0.5 (about 0.075 rad before clipping), five passes,
`1e-4` learning rate and desired KL 0.01. Failure-phase sampling remains a
separate next ablation so its effect can be measured rather than bundled.

That seed-127 ablation completed the same 2,359,296 environment steps. Mean
episode length progressed 20.47, 21.00, 18.84, a temporary 24.44 at checkpoint
1,179,648, 20.75, 19.34 and 21.06. KL stayed near the wider 0.01 target, but
the final deterministic action-acceleration cost grew to 502.6 and all episodes
still fell. Greater exploration found a slightly better transient correction,
not a stable policy, so the run is rejected and only its best checkpoint is
retained for the next curriculum ablation.

The CPU tool then replayed exactly one episode from each of the 34 reference
frames and converted actual termination phases into a cyclic failure-focused
reset distribution. A three-tap exponential kernel (decay 0.8) shifts weight
toward the frames before failure and mixes in 10% uniform probability. The
distribution is symmetric, has normalized entropy 0.924 and peaks at mirrored
bins 6 and 23 with probability 0.0802 each. Its local JSON is bound to the
reference SHA and has SHA-256
`6cfecc97b517f3df4d5baf3c9a4f6d357acf6994c939a3db4fa49a3ccdb63d51`.
Training uses these weights, while held-out evaluation deliberately resets
uniformly so earlier learning curves remain comparable.

The resumed seed-127 curriculum run completed another 2,359,296 environment
steps from the earlier 1,179,648-step checkpoint. Uniform held-out evaluation
progressed from 20.94 to 20.88, 18.88, a temporary 24.28, 21.19, 19.69 and
21.25 steps. Every final episode still fell; final KL was 0.00812 and action
acceleration cost increased to 577.9. The best and final survival values do not
improve on uniform phase sampling (24.44 and 21.06), so the fixed-bin
curriculum is rejected for this reference. Together with the exact
inverse-dynamics result, this closes further PPO tuning on the current
kinematic cycle. The next artifact must be retargeted independently through
the pinned GMR T1 route and pass exact RCSS open-loop dynamics before formal
training.

## GMR T1 reference, v4 deployment loop and v5 curriculum — 2026-08-31

The independent replacement route is now implemented. Pinned GMR commit
`bb1bbe4...` retargets LAFAN `run2_subject4.bvh` directly to its Booster T1
29-DoF model. The importer maps joints by exact MuJoCo names into the 23-joint
competition order, zero-fills the two head joints, drops six wrist/hand joints,
clips only against the exact RCSS limits, carries source contact labels, and
records source/retargeter licences and hashes. The selected local parent has
SHA-256 `4037557d...`; 44 scalar values required clipping, with maximum 0.1 rad
and mean 0.00166 rad correction. No LAFAN-derived NPZ is committed or
redistributed.

The accepted 34-frame periodic projection has SHA-256 `02cd6409...`, a 0.68 s
cycle, 1.667 m forward displacement, 13 contact frames per foot and four
aerial frames. It passes exact joint-limit/non-foot collision, bilateral
symmetry, cyclic joint/root velocity, yaw, lateral excursion and stance-slip
gates. The minimal root-XY smoothing that passes the cyclic velocity seam is
three passes; this setting is recorded rather than silently increasing the
filter. A PD ablation selected `Kp=50`, `Kd=1.2`: it raised mean source-speed
open-loop survival from 22.47 to 24.29 steps, while larger gains reduced joint
error but did not improve survival.

The official Booster Gym audit explains why this ablation was warranted: its
T1 locomotion configuration uses substantially stiffer leg gains, a 50 Hz
policy, explicit action/velocity penalties and privileged base velocity for
the critic. The official BeyondMimic implementation remains the reference for
whole-body targets and adaptive phase sampling. ASAP is retained as a later
sim-to-real alignment method, not applied now because the measured failure is
within the same RCSS/MJWarp task rather than a simulator-to-hardware gap.

`run_policy_v4` binds the new reference hash and selected gains. Its zero-action
1.8 m/s MJWarp mean survival is 26.5 steps, materially above the old reference's
approximately 20 steps. Formal seed-139 PPO trained for 2,359,296 environment
steps and peaked at 28.70 steps at checkpoint 1,179,648; every evaluation
episode still fell. The standard Brax checkpoint exposed and fixed two
deployment-tool gaps: ONNX export now supports the nested standard MLP with
Swish and observation normalization, and the CPU evaluator now reproduces
reference-centred phase/velocity initialization, contract action clipping and
contract PD gains. Legacy actor export remains covered by a regression parity
run.

The best v4 ONNX has SHA-256 `a107ffe6...`; JAX CPU versus ONNX Runtime CPU
parity over 256 samples has maximum error `1.19e-7`. Exact MuJoCo CPU evaluation
at 1.8 m/s over 64 episodes reports median survival 27 steps (0.54 s), maximum
40, median forward speed 1.772 m/s, median RMSE 0.461 m/s, median lateral drift
0.160 m and zero full 10 s completions. An aerial interval occurred before
failure in 81.25% of episodes, but none survived the two-second warm-up.
Therefore this visibly produces running morphology without producing a usable
competition runner, and it is rejected for deployment.

A speed scan found a possible curriculum entrance at 0.8 m/s (36.56 mean
zero-action steps versus 26.5 at 1.8 m/s). It also exposed a transfer hazard in
v4: fixed command channels acquired near-zero running-normalizer variance.
`reference_curriculum_v5` removes that redundant running normalization, lowers
initial exploration standard deviation to 0.3, uses three PPO passes and a
tighter 0.005 KL target. Low-speed training peaked at 35.75 held-out steps;
restoring that checkpoint at 1.2 m/s started at 32.75 and ended at 32.67 after
393,216 steps. This does not exceed the zero-residual baseline, so the
low-to-medium-speed curriculum is closed rather than escalated to 1.8 m/s.

The machine-readable record is
`training/locks/gmr_reference_baseline_2026_08_31.yaml`. No experimental ONNX
was copied into the competition runtime. The next motion milestone requires a
dynamically feasible reference or a richer whole-body tracking observation;
more epochs on either current periodic reference are not justified by these
curves.

## Guarded competition posture integration — 2026-08-31

The formal `Walk` action now contains an opt-in deployment adapter for the
exact v4 actor (`a107ffe6...`) plus external GMR reference (`02cd6409...`). The
adapter reconstructs the training 80-value observation, chooses the reference
phase nearest the measured server pose, applies the v4 residual/sign decoder,
and independently calculates stable `walk.onnx` output on the same cycle. It
requires straight forward `PLAY_ON` movement by a field player, validates both
asset hashes and the 80-to-23 ONNX boundary, rate-limits reference targets,
uses a 16-cycle ramped window and two-second cooldown, and returns to stable
walk immediately on posture or inference failure.

The first real 800-cycle 7v7 actuator trial allowed the reference target to
reach full ownership. It connected all 14 players and exited cleanly, but only
2 of 18 activations completed; 16 hit the posture guard and nine get-ups were
observed versus five in a same-length stable control. Full target ownership is
therefore rejected and is not configurable in the final adapter.

The accepted integration caps the v4 contribution at 10%, making it a posture
hint while the evaluated walk model retains at least 90% authority. Three
independent 800-cycle exact-server runs completed 5/5, 5/5, and 16/16
activations with zero posture/inference aborts. All connected 14/14 players,
entered `PLAY_ON`, completed the attack loop, logged zero client failure, and
shut down cleanly. The runs observed five, eight, and three successful get-ups;
the stable control observed five. This passes the new integration gate but not
the running-policy R2 gate. `MY3D_RUN_BACKEND` therefore defaults to `stable`,
the model/reference remain external and local-only, and v4 retains rejected
release status.

## Target-conditioned kick foundation and CEM teacher — 2026-08-31

R1 now has an executable parameter boundary on both sides. `KickCommand`
target, requested speed and mode metadata can drive an opt-in bounded contact
profile; the flag defaults off and every invalid, non-finite, over-angle or
over-speed request returns the exact accepted fixed contact. The training
contract advances from the preserved 90-value direction-only v1 contract to
`kick_policy_v2`, a 96-value observation containing target distance, requested
launch speed, desired arrival speed, and pass/shot/clear mode.

A deterministic exact-MuJoCo CEM route was added before policy training. The
first neutral-pose keyframe search made upright contact but moved the ball only
0.076--0.094 m and was rejected. Re-centering the same bounded 14-parameter
trajectory on Apollo's accepted walk policy changed the result materially.
Nominal 2 m centre-target searches with seeds 1701, 1702 and 1703 produced
2.033, 2.075 and 2.429 m maximum progress, closest-target lateral errors of
0.046, 0.115 and 0.050 m, and no falls. This exceeds the earlier physical
0.186--0.644 m contact baseline in the exact-asset training simulator.

The measurement was then corrected so distance is scored at closest passage
over a three-second rollout and launch-speed error is explicit. A small
speed-aware search reached the 2 m plane within 0.0014 m, with 0.223 m lateral
error, 2.129 m/s maximum forward ball speed versus the 1.43 m/s request, and no
fall. It is useful teacher evidence but is not a promoted kick.

Held-out placement evaluation exposed the binding limitation. The nominal
teacher passed only 5/20 trials over ball offsets
`x=[-0.01, 0.08] m`, `y=[-0.08, 0.08] m`. A five-placement robust-objective
ablation passed only 1/20 on the same unseen seed. All 20 robust trials made
contact and remained upright, but direction and range varied with ball pose.
The fixed-trajectory route is therefore closed as a universal executor. Its
accepted role is to generate per-condition demonstrations for the v2
target/ball-conditioned residual policy; further single-trajectory tuning is
not justified.

All manifests and NPZ trajectories remain under
`/home/win98/rl_runs/kick-teacher`. None was copied into the competition
runtime. The next R1 deliverable is a labeled multi-condition teacher dataset,
supervised initialization, then randomized residual training and held-out
ONNX/server evaluation.

## Residual kick teacher, DAgger and guarded runtime table — 2026-09-01

The first behavior-cloning target incorrectly combined Apollo's stable walk
output with the kick offset. The walk policy carries temporal state that was
not fully observable through `kick_policy_v2`, so a small per-step regression
error compounded in closed loop: the nine-condition actor had ONNX parity
`4.17e-7`, yet fell in 20/20 exact-CPU trials. One pure-learner DAgger round
reduced validation MSE from `0.0429` to `0.0337` but still fell in 20/20.

The decoder was corrected to `Apollo walk target + bounded kick residual` and
the one-shot phase was made injective over the 1.2 s motion. This retained the
accepted walk controller as a physical baseline and changed the failure mode:
subsequent candidates made contact and remained upright rather than diverging.
A fresh 2 m, `-15/0/+15` degree, three-lateral-position teacher grid passed
9/9 exact nominal conditions. Its NPZ contained 1,359 samples. Residual BC
reached train MSE `1.40e-4`, validation MSE `0.0163`, ONNX parity `3.58e-7`,
but only 0/20 randomized closed-loop successes. Residual DAgger round one
aggregated 5,436 samples, achieved validation MSE `7.88e-4`, and improved the
randomized result to 3/20 with 0 falls and 20/20 contacts. Round two did not
improve success, so the current MLP is retained as research evidence rather
than deployed.

The reliable branch therefore exports deterministic smooth residual
keyframes. Local robust CEM was expanded from two central nodes to a 3-by-5
ball-position grid. Independent nearest-node success progressed from 13/100
for the nominal table, to 40/100 for two locally robust nodes, 63/100 for the
first position grid, and 79/100 for the repaired grid on its first held-out
seed. Stronger worst-sample CEM did not reliably cross the gate. The final
locked 15-node candidate achieved 224/300 (`74.67%`) over ball offsets
`x=[-0.01,0.08] m`, `y=[-0.08,0.08] m`, with 300/300 contacts and zero falls.
A phase-alignment scan over `-2,-1,-0.5,0,0.5,1,2 s/m` selected zero; every
fixed longitudinal timing correction reduced the held-out success rate.

This result is useful but below the 90% promotion threshold. The generated
runtime table is therefore explicitly `promotable: false`, the
`--enable-parameterized-kick` flag still defaults off, and unsupported target,
speed, mode, ball-pose or visibility conditions fall back in the same cycle.
The C++ executor loads 15 nodes, selects only within the measured 2 m forward
pass envelope, reproduces the six smooth keyframes and per-joint residual
clips, adds them to the stable walk target, and returns to zero at 1.2 s.
CTest passes 7/7 including the cross-language knee clipping value and fallback
cases.

Locked evidence:

- teacher manifest SHA-256
  `4605fc6b18f360c4079a9f8cf8c5bf463523c5d35f834f4bec351f4d44136108`;
- teacher dataset SHA-256
  `46dea0173f6b2b9dcdac07905de2fcac0330eb3e324c63216cd83d9fc7e21bf1`;
- 300-trial evaluation SHA-256
  `e3784288c8e45c50a9e4cb9cec909ae5dd255e5821aed48a0f013dde618b6a99`;
- exported runtime table SHA-256
  `791bcfcbddbc24b9b1b1e2f0b9b9d650f5100a6a226a59b717a2a08bed3d953d`;
- Apollo walk baseline SHA-256
  `6df65fa7d36fd4989fcb022e385de797d51f35c8375532841034716e4bc0d850`.

R1 is not complete. The next action milestone is an adaptive setup/contact
state machine and robust 2/3.5/5 m pass grid, followed by shot/clear envelopes,
moving-ball transitions, three-seed exact evaluation and RCSSServerMJ server
calibration. More epochs on the current BC actor or more fixed-phase CEM are
not justified by these results.

## Phase-aware transition corpus and exact-CPU teacher pivot — 2026-09-01

The kick transition is now represented by the versioned 98-value
`kick_policy_v3` contract. Seed 7003 generated 122 valid exact-CPU walk-to-kick
states from 128 randomized setup rollouts. The 97/25 whole-rollout split covers
all eight locomotion-phase buckets on both sides, and the source rollouts
record 121 contacts with zero falls. The corpus NPZ SHA-256 is
`416ca5f6447c750ca018ff83c80a8b56fa2cf87f06cfa5888da48ae373471bba`.

Backend verification rejected JAX and accepted Warp for a 60-step
identical-control diagnostic. A guarded 16,384-step Warp PPO seed 7203 was then
run with contact, full-gate and fall events scaled as episode impulses. It
retained zero gate successes and worsened deterministic validation fall rate
from 62.5% to 87.5%. ONNX parity passed at `1.40e-9`, but exact CPU validation
reported 0/25 successes, 25 contacts and 17 falls. The checkpoint is rejected.

A 150-step audit exposed a stricter problem than raw physics parity. The first
14 independently generated control targets agree, after which tiny Warp/CPU
state differences are amplified by the closed-loop walking policy. The target
gap is `0.071 rad` at cycle 15 and peaks at `1.49 rad`. The earlier diagnostic
applied the accelerated target to both engines, so it could not detect this
closed-loop divergence. Formal accelerated learning is now gated on an
independent-controller corpus comparison rather than one open-loop trace.

Exact CPU black-box optimization remains feasible. A seed-7301 single-state
proof passed the complete gate after 24 candidates by 10 generations, reaching
1.952 m progress, 0.048 m range error, 0.105 m lateral error and 0.240 m/s
launch-speed error without falling. The resumable seed-7302 eight-phase table
then took 236 seconds. It improved training results from 5/97 to 19/97 and
held-out results from 0/25 to 3/25; held-out falls remained zero. Phase buckets
0 and 5 retained mean lateral errors of 1.161 and 1.094 m, while buckets 1 and
2 were dominated by distance/speed error. Consequently, phase-only lookup is
rejected as a deployment selector. The next dataset must label individual
states and use the complete transition observation, especially ball-local
position, root velocity and torso state, to predict bounded trajectory
parameters.

Immutable local evidence:

- Warp run manifest SHA-256:
  `c0909a549a3c353d8588180120524611cbbcfff32a65ff266c0747a221d424b9`;
- rejected Warp ONNX SHA-256:
  `0c82759235f6f1ed9fad00553931fbe977339bffc23fbf25bf05a7e86db5e507`;
- exact CPU ONNX evaluation SHA-256:
  `81f11c4886823bd31532bceb51e84c76ee8224da6a418a739a2e5d26bb1e6482`;
- single-state proof SHA-256:
  `a85411c46b8c3d8d289c577715cf0eb67ce53a46578688b9f6667cdf635716d1`;
- phase-table manifest SHA-256:
  `1c8747abf87032a35766b598628af16c88294204abda66abb40f1d34f7eaee1c`.

## Per-state labels and sequential switch windows — 2026-09-01

The exact-CPU per-transition optimizer labels each training state independently
and writes atomic resumable JSON plus a compact NPZ. Seed 8001 followed by the
seed-8101 repair pass solved 361/368 (`98.1%`) enlarged-corpus training states,
with zero falls. The label manifest and NPZ SHA-256 values are respectively
`ce692307b73ac25eb2c8f7d0122b1a74b9135166eee87033598a55f1b7ed1b4c`
and `b1761811a84f75a5c69cb646ccff476d9544d1f313fd508f0633e39a814f0418`.

State-to-parameter selection did not inherit that oracle performance. Global
and phase-aware nearest-neighbour selectors reached only 36--37/92 untouched
states. A fixed phase-6 action plus a development-selected physical gate was
frozen and evaluated on independent seed 8901: it released five of 41 phase-6
states and succeeded on three (`60%`), with no falls. The report SHA-256 is
`63e288445367c8afea75533fd07778e8e0fd7530cd75724584fe059c693f09e0`.
The gate is rejected and was not retuned.

The next experiment changes the unit of supervision from an arbitrary state
to a complete approach sequence. Increasing alignment-confirmation counts
produce deterministic candidate switch frames on the same approach; the
frozen action is replayed from every exact state, and all adjacent frames stay
in the same split. Seed 9301 generated 3,579 frames from 128 approaches. The
single action has 839 successful frames and six fallen frames, but only 87/128
approaches contain any successful window (`67.97%`); untouched validation is
15/26. Its manifest/NPZ SHA-256 values are
`0acebc40b56285075a35e02ea4407dc907e085f8421b513e614e450d6227f282`
and `1e0e4fdf190b61735bf88d768961d5b45d797fa70da12884a4a2276f11cf1994`.
This retains the timing route but rejects a single-prototype trigger. A small
predeclared action bank is the next oracle-coverage gate; no selector or C++
runtime integration is allowed before it passes held-out rollouts.

The predeclared bank passes only as an oracle. Four prototypes selected on the
102 training approaches cover 98 of them and 25/26 untouched approaches; all
ten cover 26/26. The bank NPZ SHA-256 is
`afb2bcfe40dd6d8f331bd41ef5b2256736ce92117c26b4254e13d4c4fea88988`.
A grouped kNN policy realizes only 10/26, while the strongest MLP realizes
15/26 overall (15/16 released, no falls). The selectors are rejected because
oracle coverage does not imply online identifiability.

Full teacher trajectories were therefore replayed from all 361 successful
states. The resulting 54,511-sample v3 dataset uses 289/72 whole-episode
train/validation groups and all eight phase buckets; its NPZ SHA-256 is
`b49387323a1fe7982dfc10b405fbca8283bc88532b0397fa4488c764723b6dea`.
Plain BC achieves validation MSE `5.53e-4` and ONNX parity `4.32e-7`, but exact
physics passes 0/92 states, with 83 contacts and zero falls. DAgger round one
passes 10/92. Round two passes 27/92, contacts in 92/92 and falls in 1/92. The
round-two ONNX and report SHA-256 values are
`b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1`
and `978e8c5682dbc0001cf88b39dbc711cd4250d959672df9e07a866b0baf6e65c4`.
This is meaningful growth but still a rejection at `29.35%` versus the 90%
gate. A subsequent iteration must explicitly penalize learner falls; supervised
loss and contact count are diagnostic only.

## Safety DAgger, physical residuals and dense causal switching — 2026-09-01

Failure-weighted DAgger round three contained 185,365 frames. Failed learner
rollouts received weight 2 and fallen rollouts weight 6. Exact untouched replay
passed 16/92, contacted in 92/92 and had zero falls. Safety recovered, but the
round-two 27/92 peak did not. The round-three ONNX SHA-256 is
`543e02671e0cc39034b08e4910156a64abf52668313834e0caa3e205dd8a3260`;
the exact report SHA-256 is
`fb69f70405cf0c0c897a2c964b0b464adce56358f2536c1a67b22425a58ba808`.

The v3 clone can now run as a frozen JAX base policy under a bounded PPO
correction. JAX inference uses highest-precision matrix multiplication and has
a source/export parity test. The 16,384-step aggressive run passed 10/92 and
fell three times. A 131,072-step conservative run evaluated every 16,384-step
checkpoint in exact CPU physics. Safe results ranged from 12/92 to 15/92; the
65,536 and 114,688 checkpoints each fell once. No checkpoint exceeded the
safe 16/92 base, so physical residual PPO is rejected at this task scale.

Single-pass sequence capture is byte-identical to the old repeated setup path.
Seed 10601 produced 13,752 candidates from 512 approaches: 3,853 successful
frames, 17 fallen frames and successful windows on 78/102 validation
approaches for the original prototype. Its corpus manifest/NPZ SHA-256 values
are `7aa8d4cc64e6cbab5750448d399a55b7c275bd91a200ae3e646197f7dd5ae56d`
and `e6b537deb86d201f54683d28f618a0714a238a730b52c58d7d501b73ad8665d8`.

Ten actions were then evaluated over every candidate. The four-action subset
chosen only from training approaches is `[65, 107, 79, 4]`; it covers 383/408
training and 101/102 untouched validation approaches. The bank manifest/NPZ
SHA-256 values are `c5fce79d4dfa0d80ca46e8ab581df178eded55a5591485d05155b4308c1a2297`
and `a81d5d80f633248079b0b066733d81fc815554c1a3b66301dbdbe5f7060b9cc6`.

A deployable multi-label selector now supports whole-rollout fit/calibration/
validation separation, group-balanced batches, causal consecutive-frame
release, optional calibrated fallback and ONNX export. The safest all-bank
model passes 66/102 blind approaches with zero falls and `95.65%` release
precision. Current-state and anchor-history MLPs, grouped kNN, trajectory-level
nearest-neighbour planning and a safe fixed fallback all remain below the
90% total-success gate. This closes open-loop prototype selection as the main
R1 route despite its `99.02%` oracle.

An open-source refresh identified an exact robot match: the official ICRA 2026
`Daffan/humanoid-soccer` Booster T1 source, Apache-2.0, commit
`378a12ac7446cd175f973c04e32912eb9acbee10`. It trains a privileged 20-second
approach-and-kick teacher in 4,096 environments, then a 50-frame-history DAgger
student and constrained adaptation. The next experiment ports that task
structure and curriculum to this repository's MuJoCo/Warp backend. No external
checkpoint exists in the source, and Isaac Gym code is not copied into the
competition runtime.

## Long-horizon striker task and settled contact gate — 2026-09-01

`striker_policy_v1` now defines a 102-value student observation, 138-value
privileged teacher/value state and 23 residual actions. The task lasts 1,000
50 Hz cycles and composes Apollo's accepted walk target, a frozen 60-frame
kick prior and a distance-gated 0.10-scale learned correction. Reset
curricula cover near-ball, closed-loop and robust approach distributions.
The training script records hashes/configuration/checkpoints and refuses a
formal accelerated run without a passing parity report.

The first implementation exposed three experiment-invalidating effects.
Missing `time_out` metadata broke Brax bootstrap-on-timeout. Online Welford
normalization made constant pre-contact features acquire near-zero variance;
after contact, a seed-11102 checkpoint saturated almost every action near
`+/-1` despite small in-distribution error. It is rejected. With
normalization disabled, prior-free seed 11103 remained upright but produced no
contact through the retained continuation, confirming that a 5% Gaussian
residual cannot discover the coordinated 23-joint kick from scratch.

Adding the physical prior initially still produced no contact. Exact replay
located the mismatch: the original trajectory was optimized while Apollo
continued `[0.50, -0.04, 0]` for 33 frames. Restoring this versioned ownership
made the nominal pose reach 1.93 m/s directional ball speed. Removing
activation-based attenuation from the approach command then eliminated a gait
deadband. Under a strict 7 cm trigger, frozen seed 11203 reached 55/64,
contacted 55/64, succeeded 19/64 and fell zero times. All nine missed robots
settled safely between 8.2 and 10.2 cm.

The accepted trigger is now strict immediate release plus a 25-frame
confirmation inside an 11 cm/0.10 rad envelope. On the same 64 Warp rollouts
it triggers and contacts in 64/64, succeeds in 29/64 (`45.31%`) and has zero
falls. An independent exact CPU MuJoCo implementation evaluated seed 12203:
64/64 triggers, 64/64 contacts, 23/64 successes (`35.94%`) and zero falls. A
32,768-step prior-assisted teacher checkpoint reaches only 28/64 under the
accepted Warp controller, so it regresses against the 29/64 prior and is
rejected; later continuation also regresses.

Immutable local evidence:

- striker contract SHA-256:
  `5cbd1d899336416078bd377ce5c713f3c09f96f65c9f3204f3e88b8ad37e411d`;
- frozen kick-prior manifest SHA-256:
  `4605fc6b18f360c4079a9f8cf8c5bf463523c5d35f834f4bec351f4d44136108`;
- accepted prior-only Warp report SHA-256:
  `9d61cf3c1b56b496ad0b10f93d328c0e855e71a2223646e7ab606fef8b399b29`;
- rejected early-teacher Warp report SHA-256:
  `404cc55f444eec7361abedcb22a689d97717af95ce126e586173c03b1d2aea25`;
- accepted exact-CPU diagnostic report SHA-256:
  `b601785d2dcb0eea58fb2cd58b6a8c35da844d80f4d2b39840af5d7b394ef3f0`;
- rejected early/continued teacher run-manifest SHA-256 values:
  `1269930628ba571df16fb442987e30a00241f6fb77bc1caed6b1b995dd5cf4f6`
  and `619f6e171b3d27561ef70951a026362b0deeafe4273787357447411f8d16cef6`.

This is a safe closed-loop baseline, not a promoted kick. The next experiment
must train target-range-conditioned physical outcome corrections and exceed
the deterministic 23/64 exact-CPU baseline before history distillation. More
steps on the rejected PPO objective are not authorized by current evidence.

## Frozen selector, outcome and fixed-5 m audit — 2026-09-01

The exact-CPU striker was extended with a distance-indexed kick bank, exact
102-value trigger observations, walk-state and 138-value privileged capture,
50-frame causal history, continuous action-outcome labels and an independent
Warp/CPU parity driver. Success is now strict: contact, target distance at most
0.5 m and directional arrival-speed error at most 0.5 m/s. Position-only
results earlier in this log are not comparable to this gate.

| Frozen experiment | Exact result | Decision |
|---|---:|---|
| five-action oracle, all corpus rollouts | 928/1023 (`90.71%`) | action bank is expressive but not deployable by oracle |
| current-state selector | 153/205 (`74.63%`), 0 falls | reject |
| privileged trigger selector | 146/205 (`71.22%`), 0 falls | reject; privileged values do not resolve sensitivity |
| 50-frame history selector | 147/205 (`71.71%`), 0 falls | reject; history alone is not the missing motion skill |
| continuous outcome-regression selector | 128/205 (`62.44%`), 0 falls | reject despite about 90% fit-set rollout success |
| fixed 5 m prior, zero residual | 172/256 (`67.19%`), 0 falls | frozen baseline only |
| residual PPO scale 0.1, 65,536 steps | 171/256 (`66.80%`), 0 falls | reject |
| residual PPO scale 0.5, 262,144 steps | 171/256 (`66.80%`), 0 falls | reject |

Online 64/128-rollout evaluations had appeared as high as 75--77%, but the
predeclared frozen 256 set shows those changes were evaluation variance rather
than learning. No model was promoted.

The formal sine-action parity run covers 100 identical cycles and passes every
declared Warp/exact-CPU threshold. Maximum deviations are 0.0135 rad joint
position, 0.8528 rad/s joint velocity, 0.00777 m root position, 0.0143 rad
torso orientation, 0.00414 m ball position and 0.2319 m/s ball velocity. This
preserves the simulator gate while ruling out backend divergence as the primary
cause of the frozen policy failures.

The PAiD and `wbc_fsm` audit changes the next authorized experiment. The
current fixed-prior reward is also corrected from monotonically rewarding ball
speed to commanded speed tracking, explicit miss/timeout cost and controlled
arrival-speed penalty; this change must pass tests but is not itself a reason to
resume formal training. The new order is PAiD-informed T1 motion retargeting,
motion-skill tracking, perception-action adaptation, then physics robustness.
See [`paid-wbc-fsm-audit-2026-09-01.md`](paid-wbc-fsm-audit-2026-09-01.md).

Checkpoint verification: Python compilation passed for all changed/new striker
modules and tools; 11/11 targeted striker tests, 111/111 training tests and
50/50 root regression tests passed in WSL using the `my3d-rl` conda
environment. `git diff --check` also passed. The repository environment does
not currently include Black, so no formatting claim beyond the existing test
and diff gates is made.

## PAiD-to-T1 K0 corpus gate — 2026-09-01

The new local-only loader verifies the pinned unnamed PAiD array order, safe
NPZ loading, 50 Hz schema, quaternion norms, left/right filename labels,
repository revision, licence and per-file hashes. The audited corpus contains
13/13 valid motions: ten standard, three stylized, four left-foot and nine
right-foot.

Two same-source robot-to-robot conversions were then evaluated. The A baseline
projects same-semantic G1 joints into the existing T1 contract. The B candidate
uses GMR whole-body IK but calibrates the G1-to-T1 body-frame offsets from A's
first pose; the uncalibrated SMPLX offsets failed and are rejected. Both final
corpora pass 13/13 exact RCSS kinematic gates and have zero non-foot pitch
contact frames.

| K0 aggregate | Semantic A | Calibrated body-IK B |
|---|---:|---:|
| joint-limit clipped values | 1,987 | 157 |
| maximum correction | 0.303 rad | 0.10 rad |
| maximum joint velocity | 16.37 rad/s | 15.18 rad/s |
| maximum root tilt | 0.461 rad | 0.375 rad |
| mean labeled-foot peak speed | 4.991 m/s | 4.998 m/s |
| minimum labeled/other peak-speed ratio | 1.650 | 1.448 |

B is the K1 primary motion reference because it preserves full-corpus passage
with 1,830 fewer clipped values. A remains the required ablation and fallback.
This is not a kick promotion: dynamic tracking, ball contact, direction and
arrival-speed gates are still open. Evidence hashes are locked in
`training/locks/paid_k0_2026_09_01.yaml`; all CC BY-NC inputs and derivatives
remain outside Git.

K0 checkpoint verification: all changed/new Python files compile, the four
PAiD/soccer-reference tests pass, all 115 training tests pass, all 50 root
regression tests pass, the three changed YAML files parse, and
`git diff --check` passes. The full corpus was regenerated after the final
label-dominance and provenance changes before evidence hashes were locked.

## PAiD-to-T1 K1 finite motion tracking — 2026-09-01

K1 first measures the K0 corpora under the actual 50 Hz position-plus-velocity
actuator protocol. With 25/0.6 gains, semantic and calibrated body IK have zero
full-clip completions; semantic phase completion is 26.9% versus 25.0%. The
50/1.2 locomotion ablation and Apollo's deployed per-joint gains reduce tracking
RMSE from about 0.09 to 0.06 rad and raise phase completion to about 29.8%, but
still complete zero of thirteen full clips. Semantic preserves about 81% contact
agreement under Apollo gains versus 73% for body IK. This dynamic evidence makes
semantic the first training candidate without deleting the kinematically cleaner
body-IK corpus.

The implemented training boundary is finite rather than circular. It pads the
13 clips only for static JAX shapes, carries the true length, forbids endpoint
wrap, samples motion and failure-focused phases, advances exactly one frame at
50 Hz, uses Apollo per-joint gains and clamps residual targets against exact T1
limits. The actor has 110 values and the asymmetric critic 118. A 1,024-step
Warp smoke run completed optimization, evaluation and checkpoint data. An
attempt at 256 simultaneous Warp evaluation episodes later caused a process
segmentation fault on the 8 GB GPU; 128 is the verified local evaluation batch
limit and the failed attempt produced no evidence file.

Three measured policy iterations followed. Conservative v1 and higher-exploration
v2 at 0.15 rad both remained exactly 36/128 on a fixed seed set. Contract v2
raises residual authority to 0.35 rad, consistent with the existing Apollo kick
scale, without changing zero-action behavior. Moving resets five frames before
a 25-frame pre-failure window and continuing the v3 checkpoint reached 39/128
on the fixed Warp set.

The decisive exact CPU grid has eight deterministic start phases for every
motion. Semantic zero residual completes 31/104; the retained checkpoint
completes 35/104. Paired transitions are four improvements, zero regressions,
giving one-sided exact McNemar `p=0.0625`. Mean joint RMSE changes from 0.0725
to 0.0735 rad and contact agreement stays essentially flat. Applied to body-IK
references without retraining, the same actor changes 31/104 to 34/104, with
four improvements, one regression and `p=0.1875`. Neither comparison passes the
predeclared `p<=0.05` promotion rule. The checkpoint remains experimental and
no competition runtime asset changes.

The reward-only PPO branch is stopped. The next K1 experiment must optimize
phase-level residual teachers before measured failures, verify those teachers
on the same exact CPU grid, behavior-clone the successful corrections, and only
then resume PPO. All report and manifest hashes are in
`training/locks/paid_k1_2026_09_01.yaml`; PAiD-derived arrays and checkpoints
remain outside Git.

K1 checkpoint verification: every changed/new Python file compiles; all 122
training tests and all 50 root regression tests pass in WSL; both policy
contracts and the K1 evidence lock parse as YAML; `git diff --check` passes.

## Phase-v2 full high-speed-walk Apollo integration — 2026-09-01

The selected phase-v2 locomotion model was integrated into the Apollo C++
runtime as an opt-in `FastWalkV2` backend. Although its training contract and
external directory retain the word `run`, locked CPU evaluation reports zero
flight phase; the capability is therefore classified as high-speed walking.
The model owns all 21 body-joint position targets at its exact 0.5 residual
scale and 25/0.6 gains while Apollo retains head tracking. This is full policy
execution, not the earlier low-weight posture blend.

The first two 700-cycle gates produced zero fast-walk samples because the
runtime initially assumed absolute position commands. Apollo's far-target
planner actually emits body-local velocity commands; the corrected router
enters the fast backend only for long-forward demand and retains stable walking
for braking, reverse/lateral travel, sharp turns and goalkeeper behavior. A
second defect switched backends whenever normal gait oscillation crossed the
entry tilt/gyro thresholds. An entry/continuation hysteresis now preserves the
same closed-loop policy until a functional route change or genuine loss of
posture.

| Strict 7v7 run | Fast-walk samples | Get-up samples | Other evidence |
|---|---:|---:|---|
| pre-hysteresis, 700 cycles, status/5 | 216 | 168 | 14/14 clean; one pass contact |
| hysteresis, 700 cycles, status/5 | 503 | 125 | 14/14 clean; no fatal/illegal-defense errors |
| all abilities, 900 cycles, status/2 | 1,374 | 499 | 25 parameterized kicks, 210 pass plans, 176 ready samples, one pass contact, 14/14 clean |

The wiring and coexistence gate passes, but the get-up rate proves a remaining
Apollo-server domain gap. `FastWalkV2` remains local-only and opt-in. The next
locomotion experiment should train or adapt against Apollo/RCSS transition and
disturbance states, with fall rate as a primary promotion metric; increasing
nominal speed is not the next objective.

## PAiD K1-A exact-CPU phase teacher and live visualization — 2026-09-01

The K1 continuation now uses a shared deterministic actor boundary and an
exact-CPU low-dimensional phase teacher instead of adding more reward-only PPO
steps. Formal tools refuse a dirty Git tree, validate the PPO/profile contract,
keep reports and demonstrations outside Git, and record source, contract,
motion and dataset hashes. TensorBoard-compatible events expose every CEM
generation and every future finite-motion PPO evaluation; captured exact CPU
`qpos` trajectories can be replayed in the MuJoCo viewer independently from
accelerated training.

The deliberately short 4-by-1 platform smoke completed in 3.94 seconds and
proved event and 512-frame replay generation. It happened to improve the four
held-out phases, but was not used as evidence. The formal 64-population,
8-generation search ran from clean revision `f5a4837` in 238.61 seconds on the
worst retained motion, `football_stylized-001`. Its objective improved through
all eight generations from 392.67 to 468.26.

| Fixed exact-CPU phase split | Baseline completion | Teacher completion | Baseline survival | Teacher survival | Delta |
|---|---:|---:|---:|---:|---:|
| training, 4 phases | 0/4 | 1/4 | 0.4274 | 0.5409 | +0.1135 |
| untouched validation, 4 phases | 1/4 | 1/4 | 0.5028 | 0.5533 | +0.0505 |

All four predeclared K1-A checks pass: training and validation score improve,
held-out survival gains at least 0.02, and no validation completion is lost.
This supports expanding the teacher to all thirteen locked motions and then
training BC/DAgger. It does not promote a model, widen a runtime envelope, or
justify PPO resume by itself. The report, 585-frame dataset and eight-generation
TensorBoard event are bound by `training/locks/paid_k1a_2026_09_01.yaml`.

## PAiD K1-B full-corpus BC and corrected DAgger — 2026-09-01

Thirteen independent accepted teachers were assembled without copying local
motions into Git. The selected corpus contains 6,834 frames and is bound to
selection-manifest SHA-256 `9ec33391a6ebe7a849d1bd2b6c4184d3257c2b3c6394b76ba7a2e359b1381b4e`.
The compatible 512-256-128 PPO actor was fine-tuned for 5,000 steps while the
normalizer and critic were retained. Teacher-validation MSE reached
`0.00047172`.

The first model-selection grid excluded every teacher train/validation start.
Across 388 paired exact-CPU episodes, completion changed from 111 to 117,
survival from 0.5410 to 0.5638, and transitions were seven candidate-only
versus one baseline-only completion (`p=0.03515625`). Tracking tolerances pass,
so this BC checkpoint replaces the prior K1 experimental actor, not the Apollo
competition motion.

The initial DAgger implementation was then rejected for a causal defect. It
queried `student + phase correction`; because the student already distilled
`teacher base + phase correction`, this doubled the correction and caused mean
blind action magnitude to grow from 0.0666 to 0.2548. Dataset SHA-256
`d6350150...` and every candidate derived from it are explicitly invalid.

The corrected collector executes the student with beta zero and labels the
same state using the frozen original teacher-base actor plus the selected
motion correction. It aggregates 20,682 new frames. Output-head-only retraining
keeps mean blind action at 0.0685 and improves mean survival from 0.5517 to
0.5704 on 666 starts excluded from both teacher and DAgger data. Completion is
only 184 versus 183, however, with three improvements, two regressions and
`p=0.5`; the DAgger candidate is not promoted. This demonstrates that the
remaining limitation is the open-loop phase teacher, not a lack of supervised
optimizer steps. A state-feedback short-horizon exact-CPU teacher is required
before PPO resume. Full paths and hashes are in
`training/locks/paid_k1b_2026_09_01.yaml`.

## PAiD K1-B state-feedback curriculum gate — 2026-09-01

The remaining causal defect was addressed with an exact-CPU, short-horizon
teacher that evaluates bounded corrections from the state actually visited by
the student. Its two-frame search accepts a label only when local cost improves
by at least `0.001`, and clips the label to `0.2` action units from the student.
The formal beta-zero collection covers 388 student episodes and selects 12,717
useful state-feedback labels; the aggregate has 19,551 frames. An output-head-
only 1,000-step clone preserves the retained feature extractor and critic.

The 679-trial tuning grid improved mean survival from 0.56025 to 0.57919. The
final grid and its thresholds were committed before evaluation; it excludes
the state-feedback training starts and the complete tuning grid. On 777 paired
trials the original BC completes 256 and the candidate 266. Transitions are
13 candidate-only versus three baseline-only (`p=0.0106354`). Mean survival
improves by 0.021339 with 367 improvements, 70 regressions and 340 ties
(`p=5.73e-50`); tracking tolerances pass. Both the ordinary exact-grid gate and
the narrower PPO-curriculum gate pass.

The state-feedback checkpoint therefore replaces the original BC as the K1
training actor and authorizes PPO initialization only. It is not an Apollo
runtime model. Formal PPO now requires the passing comparison file, verifies
that it names the exact restored checkpoint, hashes the checkpoint tree and
refuses dirty/in-repository runs. TensorBoard stays live at
`http://localhost:6006`; the new exact-CPU checkpoint viewer supplies visual
closed-loop replays at evaluation boundaries. All evidence hashes and the
remaining three-seed, ball-outcome and server-replay gates are locked in
`training/locks/paid_k1b_2026_09_01.yaml`.

The first visual PPO resume from clean revision `8e08412` is an integration
diagnostic, not a retained model. Its deterministic 64-environment Warp
completion starts at 25/64, falls to 23/64 after one optimizer step and returns
to 25/64 after the second, while average episode length declines from 55.64 to
53.61. The request was 262,144 environment steps, but Brax correctly executed
two complete 196,608-step optimizer/evaluation intervals (393,216 total); the
initial manifest predicted only the request. Formal evidence use of this run is
therefore rejected. The trainer now mirrors Brax interval rounding in advance
and records observed final steps plus an equality check. The diagnostic still
proved that live TensorBoard, hash-bound restoration, checkpoint saving and
exact-CPU policy replay operate end to end.

A separate 634-trial exact-CPU grid excludes the state-feedback dataset, its
679-trial tuning grid and its 777-trial final grid. The retained clone completes
174. The first PPO checkpoint completes 180 with eight improvements and two
regressions (`p=0.0546875`), but mean survival falls by 0.00128 and survival
regressions exceed improvements 179 to 131. The second completes 177 with five
improvements and two regressions (`p=0.2265625`); survival rises only 0.00238
with 173 regressions versus 159 improvements. Both candidates are rejected.

The next optimizer ablation is predeclared rather than extended reactively:
one PPO pass, `1e-5` learning rate, `1e-4` entropy cost and `0.002` desired KL.
Because repeated start-frame exclusion leaves a finite and shrinking grid, the
exact-CPU evaluator now supports deterministic per-motion/start joint,
root-velocity and yaw perturbations. The perturbation seed is part of every
paired record key. Two identical-seed 26-case smoke reports are byte-identical;
their comparison pairs all 26 cases exactly. Different seeds therefore provide
disjoint reproducible selection and final grids without relaxing phase
coverage. The v3 evidence and v4 protocol are locked in
`training/locks/paid_k1c_2026_09_01.yaml`.

The predeclared v4 seed `20260966` completes one audited 196,608-step update.
Its same-run Warp evaluation rises from 16/64 to 27/64, but selection uses the
locked perturbation grid instead. Across 518 exact-CPU pairs completion rises
from 177 to 186 with 12 improvements and three regressions (`p=0.0175781`).
Mean survival rises by 0.010314 with 161 improvements and 97 regressions
(`p=4.06e-5`); tracking tolerances pass. The ordinary promotion gate passes,
so v4 advances as a training protocol, not as a runtime checkpoint.

Before starting more seeds, the family gate is fixed. Seeds `20260966`,
`20260969` and `20260970` start from the same state-feedback checkpoint and
run the same single update. All are evaluated once with perturbation seed
`20260968`. Acceptance requires correct timestep accounting and tracking for
all, no seed with a net completion loss, median completion delta at least 0.01,
and at least two exact-grid promotion passes. The median seed is then confirmed
once on disjoint perturbation seed `20260971`. This prevents selecting the best
of three noisy runs and reporting it as an untouched result.

The family gate fails as written. On perturbation seed `20260968`, completion
deltas for the three runs are +0.01544, +0.00965 and +0.00579. All preserve
tracking and avoid a net completion loss, but the median is `0.00965 < 0.01`
and only seed `20260966` passes the exact-grid promotion test. Seed `20260969`
also reduces mean survival by 0.00545. The reserved representative confirmation
seed `20260971` is not opened, and the state-feedback clone remains retained.

The next candidate is an equal parameter average of the three aligned one-step
updates. This is well-defined because all runs share one initialization and
architecture; averaging their small deltas estimates the mean update rather
than mixing independently permuted networks. Observation normalization is
disabled, but Brax still updates the statistic containers, so the formal tool
copies the retained-base normalizer instead of averaging or ignoring this
difference. It checks the remaining invariants and hashes every input. The
average is predeclared for a new
selection perturbation seed `20260972` and, only if it passes, confirmation seed
`20260973`. It does not inherit the failed family's release status.
