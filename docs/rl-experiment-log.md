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
- a 90-value deployable actor observation, 100-value privileged critic state,
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
- exact competition asset assembly and 90/23 interface contract;
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
