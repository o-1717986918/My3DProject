# My3DProject motion training

This directory contains reproducible training inputs and deployment contracts.
Generated runs belong under `/home/win98/rl_runs` and are not committed.

The preserved first task is `kick_policy_v1`: a 50 Hz direction-only residual
joint-position contract for Booster T1. Active R1 development uses
`kick_policy_v2`, which adds requested range, launch speed, arrival speed and
pass/shot/clear mode to the deployable observation. It remains intentionally
separate from the deterministic team decision maker until the documented
physics, ONNX and server gates pass. See
[`../docs/rl-training-plan.md`](../docs/rl-training-plan.md) for the acceptance
gates.

Fast locomotion keeps immutable `run_policy_v1`/`v2` compatibility contracts.
The experimental v3/v4 contracts add reference-centred residual decoding;
v4 binds the pinned GMR reference and `Kp=50`, `Kd=1.2`. All retain 23 actions,
and none becomes the competition default until every release gate passes.

## Environments

- `my3d-team`: competition runtime; never install training packages here.
- `my3d-rl`: Python 3.12 training and evaluation environment.
- `my3d-motion`: Python 3.11, CPU-only isolated Holosoma retargeting tools.

Create the base environment with:

```bash
conda env create -f training/environment.yml
```

Create the isolated CPU retargeting environment with:

```bash
conda env create -f training/environment-motion.yml
```

GMR is also installed only in `my3d-motion`. Keep its checkout outside the
repository at the source-lock commit; generated LAFAN derivatives remain
local-only under the dataset licence.

Clone Holosoma at the source-lock commit, apply the project patch, then install
`src/holosoma_retargeting` editable into `my3d-motion`; the environment file
deliberately does not install an unpatched Holosoma wheel.

The MuJoCo Playground source revision and Python packages are recorded in
`locks/` after the local CUDA smoke test. Until that pin exists, a run is an
experiment and not a reproducible training release.

## Contract checks

From the repository root:

```bash
PYTHONPATH=training python -m pytest -q training/tests
python training/tools/hash_rcss_assets.py --output /tmp/rcss-assets.json
```

The asset tool is read-only: it records source paths, licences, sizes, and
SHA-256 hashes without copying RCSSServerMJ files into this repository.

Compile and step the exact-physics kick environment on either MJX backend:

```bash
PYTHONPATH=training python training/tools/smoke_kick_env.py --impl warp
```

Compile the phase-aware running environment:

```bash
PYTHONPATH=training python training/tools/smoke_run_env.py \
  --impl warp --contract-version v2 --num-envs 8 --steps 4
```

Compile the v3 moving-reference residual environment with the local-only R1
artifact:

```bash
PYTHONPATH=training python training/tools/smoke_run_env.py \
  --impl warp --contract-version v3 --num-envs 8 --steps 4 \
  --motion-reference \
  /home/win98/rl_datasets/motion_refs/t1_run2_subject4_periodic_v3.npz
```

For the GMR v4 reference, pass `--contract-version v4` and the artifact whose
SHA-256 is locked in `locks/gmr_reference_baseline_2026_08_31.yaml`.

Before long training, replay an identical short action trace in CPU MuJoCo and
MJX-Warp. The JSON includes every decoded target, reference phase, root/torso
state, foot height, contact proxy, first threshold crossing and maximum error:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=training \
  python training/tools/compare_cpu_mjwarp.py \
  --steps 20 --action-pattern sine --action-amplitude 0.15 --strict \
  --output /tmp/my3d-cpu-mjwarp-parity.json
```

Pass `--motion-reference /path/to/reference.npz` to initialise both backends
from the same full motion state. The pinned 2026-08-31 baseline is in
`locks/sim_parity_baseline_2026_08_31.yaml`; full traces remain generated
artifacts rather than source files.

Bootstrap a phase-aware locomotion experiment from the verified walk teacher:

```bash
PYTHONPATH=training python training/tools/train_run.py \
  --stage phase_run --impl warp --num-envs 128 --num-timesteps 5000000 \
  --seed 71 --num-evals 6 --num-eval-envs 64 \
  --network-profile legacy_phase_warmstart_v2 \
  --bootstrap-onnx mujococodebase/skills/walk/walk.onnx \
  --run-dir /home/win98/rl_runs/run-phase-v2-<name>
```

Continue the accepted phase-v2 checkpoint on the football command surface
(braking, reverse, lateral motion, turning, short command segments, pushes and
one-step action delay):

```bash
PYTHONPATH=training XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python training/tools/train_run.py \
  --stage soccer_omni --impl warp --num-envs 128 \
  --num-timesteps 5000000 --seed 20260951 --num-evals 6 \
  --num-eval-envs 32 --network-profile legacy_phase_warmstart_v2 \
  --restore-checkpoint \
  /home/win98/rl_runs/run-phase-v2-formal-s71-20260831-01/checkpoints/000005898240 \
  --run-dir /home/win98/rl_runs/run-soccer-omni-s20260951-v1
```

Do not select a football locomotion policy from one straight-line evaluation.
Export a candidate and run the fixed eight-command CPU suite:

```bash
PYTHONPATH=training python training/tools/evaluate_run_command_suite.py \
  --model /home/win98/rl_runs/run-soccer-omni-s20260951-v1/policy.onnx \
  --contract training/contracts/run_policy_v2.yaml --episodes 16 \
  --output-dir /home/win98/rl_runs/run-soccer-omni-s20260951-v1/cpu-suite
```

The suite covers stand, precision/fast forward, reverse, left/right strafe and
left/right turn. It reports the worst upright completion, planar velocity RMSE
and yaw-rate RMSE; every command must pass before the candidate can replace
the retained runtime model.

The first broad `soccer_omni` run is a rejected baseline: random-command fall
rate improved, but the frozen CPU suite stayed at 5/8. The follow-up must start
again from the retained phase-v2 checkpoint and use axis-aligned sampling plus
the bounded adaptive-KL profile:

```bash
PYTHONPATH=training XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python training/tools/train_run.py \
  --stage soccer_omni_axis --impl warp --num-envs 128 \
  --num-timesteps 5000000 --seed 20260991 --num-evals 6 \
  --num-eval-envs 32 --network-profile legacy_phase_soccer_v3 \
  --restore-checkpoint \
  /home/win98/rl_runs/run-phase-v2-formal-s71-20260831-01/checkpoints/000005898240 \
  --run-dir /home/win98/rl_runs/run-soccer-omni-axis-s20260991-v1
```

Export and verify a v2 checkpoint:

```bash
PYTHONPATH=training python training/tools/export_run_onnx.py <checkpoint> \
  --network-profile legacy_phase_warmstart_v2 --output /tmp/run-v2.onnx
PYTHONPATH=training python training/tools/evaluate_onnx_run.py \
  --model /tmp/run-v2.onnx --episodes 64 --vx 1.5 --action-scale 0.5
```

Standard normal-policy Brax and legacy checkpoints share the exporter.
Reference-centred ONNX acceptance must supply both the exact contract and
external reference:

```bash
PYTHONPATH=training python training/tools/export_run_onnx.py <checkpoint> \
  --network-profile reference_residual_v4 --output /tmp/run-v4.onnx \
  --parity-output /tmp/run-v4-parity.json
PYTHONPATH=training python training/tools/evaluate_onnx_run.py \
  --model /tmp/run-v4.onnx --contract training/contracts/run_policy_v4.yaml \
  --motion-reference /home/win98/rl_datasets/motion_refs/t1_run2_subject4_gmr_periodic_v1.npz \
  --episodes 64 --vx 1.8 --output /tmp/run-v4-cpu-acceptance.json
```

The exporter embeds a learned observation normalizer when the profile uses
one and verifies CPU-to-CPU numerical parity. The evaluator reports survival
quantiles and never treats post-fall states as valid tracking data.

Before a retargeted running clip can enter motion-prior training, validate its
50 Hz T1 arrays, provenance and aerial phase:

```bash
PYTHONPATH=training python training/tools/validate_motion_reference.py \
  /path/to/t1-run-reference.npz --output /tmp/t1-run-reference-report.json
```

Import a GMR result with a pinned source checkout and explicit warm-up/source
frame ranges:

```bash
PYTHONPATH=training python training/tools/import_gmr_motion.py \
  /home/win98/rl_datasets/lafan1/raw_run/run2_subject4.bvh \
  /home/win98/rl_datasets/motion_refs/t1-gmr-parent.npz \
  --intermediate /home/win98/rl_datasets/motion_refs/t1-gmr-intermediate.npz \
  --gmr-root /home/win98/reference_sources/GMR \
  --gmr-revision bb1bbe40774794fceb2a7c579a3464a28e68c844 \
  --retarget-start 1900 --frame-start 1940 --frame-end-inclusive 2030 \
  --source-url https://github.com/ubisoft/ubisoft-laforge-animation-dataset \
  --source-version LAFAN1-release --source-license CC-BY-NC-ND-4.0
```

Project an accepted even-length clip onto the versioned half-cycle-symmetric,
periodic and contact-aware R1 reference before residual-policy training:

```bash
PYTHONPATH=training python training/tools/project_periodic_reference.py \
  /home/win98/rl_datasets/motion_refs/t1_run2_subject4_straight76_109_v1.npz \
  /home/win98/rl_datasets/motion_refs/t1_run2_subject4_periodic_v3.npz \
  --source-half-weight 0.8 --smoothing-passes 4 \
  --stance-correction-iterations 1 \
  --report /tmp/t1-periodic-reference.json
```

The output must stay outside the repository. The projector checks body-local
forward semantics, time-reverses a backward-labelled source when required,
canonicalizes mean root heading to world +X, and recomputes cyclic
velocities, replays exact RCSS contacts, and rejects endpoint, bilateral,
joint-limit, collision, flight, yaw, lateral or stance-slip gate failures.
Exact contact runs shorter than four frames remain in the replay/contact
counts but are excluded from support-foot anchoring and slip statistics. The
accepted local artifact SHA-256 and metrics are pinned in
`locks/periodic_reference_baseline_2026_08_31.yaml`.

Train only the bounded v3 residual around that reference; do not bootstrap
the fixed-nominal legacy actor into this decoder:

```bash
PYTHONPATH=training XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python training/tools/train_run.py \
  --stage reference_residual --impl warp --num-envs 64 \
  --num-timesteps 196608 --seed 107 --num-evals 2 --num-eval-envs 8 \
  --network-profile reference_residual_v2 \
  --motion-reference \
  /home/win98/rl_datasets/motion_refs/t1_run2_subject4_periodic_v3.npz \
  --run-dir /home/win98/rl_runs/run-reference-residual-v3-<name>
```

The 196608-step form is one minimum optimizer epoch for this profile and is an
optimizer/checkpoint integration test. The manifest records both requested
and effective step counts. It is not a running result and cannot be selected
for deployment.

`--fixed-vx` is available only for the `reference_residual` stage and records
the effective command in the manifest. The v5 low-speed curriculum is a
rejected, reproducible ablation; do not continue it without a new reference or
observation hypothesis.

Measure the zero-residual reference before attributing a failure to PPO:

```bash
PYTHONPATH=training python training/tools/evaluate_reference_open_loop.py \
  /home/win98/rl_datasets/motion_refs/t1_run2_subject4_periodic_v3.npz \
  --impl warp --episodes 32 --steps 500 --vx 1.8 \
  --output /tmp/t1-reference-open-loop.json
```

This is an MJWarp trackability diagnostic, not the exact-CPU flight gate.

Holosoma is pinned at `fb835ec8...` and requires the audited patch in
`patches/holosoma-t1-retargeting.patch`. Verify a patched external checkout
with:

```bash
cd /home/win98/rl_sources/holosoma
git apply --unidiff-zero --check --reverse \
  /path/to/My3DProject/training/patches/holosoma-t1-retargeting.patch
```

The LAFAN1 source archive and every derived NPZ stay outside Git because the
dataset is CC-BY-NC-ND-4.0. Import and slice only into a local dataset path:

```bash
PYTHONPATH=training python training/tools/import_holosoma_motion.py \
  /path/to/holosoma-output.npz /home/win98/rl_datasets/motion_refs/run.npz \
  --source-url https://github.com/ubisoft/ubisoft-laforge-animation-dataset \
  --source-version 94084601bacdf9cc3764b5c73daaeccae6035fac \
  --source-license CC-BY-NC-ND-4.0 --source-sha256 <archive-sha256>
PYTHONPATH=training python training/tools/slice_motion_reference.py \
  /home/win98/rl_datasets/motion_refs/run.npz --start 76 --end 109 \
  --output /home/win98/rl_datasets/motion_refs/straight-cycle.npz
```

Conservative motion transfer uses the versioned `legacy_motion_track_v3`
profile. Its adaptive-KL range is explicitly bounded to `2.5e-7..2e-6`; the
Brax default lower bound (`1e-5`) must not be used for this profile.

Build a deterministic kick teacher before PPO exploration:

```bash
PYTHONPATH=training python training/tools/optimize_kick_teacher.py \
  --target-distance 2.0 --target-angle 0 --requested-speed 1.43 \
  --population 32 --generations 6 --robust-samples 1 --seed 1701 \
  --output-prefix /home/win98/rl_runs/kick-teacher/kick-v2-2m-center-s1701
```

The optimizer replays Apollo's accepted walk policy against exact RCSS assets,
adds a bounded 14-value contact trajectory, and writes a hashed NPZ plus JSON
manifest outside Git. A teacher remains non-promotable until it passes held-out
ball-placement evaluation:

```bash
PYTHONPATH=training python training/tools/evaluate_kick_teacher.py \
  /home/win98/rl_runs/kick-teacher/kick-v2-2m-center-s1701.json \
  --trials 20 --seed 2201 \
  --output /home/win98/rl_runs/kick-teacher/kick-v2-2m-center-s1701-eval.json
```

The optimizer can evaluate every candidate over an explicit release-pose
envelope and prints one progress line per CEM generation. Keep those bounds in
the generated manifest so runtime tolerances can be traced to a predeclared
experiment instead of widened after a server run:

```bash
PYTHONPATH=training python training/tools/optimize_kick_teacher.py \
  --target-distance 4.0 --target-angle 0 --requested-speed 2.5 \
  --desired-arrival-speed 0.6 --mode shot --motion-base stand \
  --stand-base-pose neutral --population 256 --generations 30 \
  --robust-samples 17 --robust-ball-x-min -0.008 \
  --robust-ball-x-max 0.020 --robust-ball-y-min -0.012 \
  --robust-ball-y-max 0.012 --seed 20261042 \
  --output-prefix /home/win98/rl_runs/player-action-foundation/shot-4m
```

The currently mounted 4 m procedural shot uses the frozen seed-20261033
teacher. Its independently seeded release-slot evaluation passed 100/100 and
the 1200-cycle 7v7 calibration observed a physical contact. This is a narrow
right-foot, near-zero-angle fallback and teacher, not a general learned shot.
The separate 6 m `clear` teacher also passed 100/100 under its safety-clearance
profile and produced a physical contact in a 1200-cycle 7v7 run. It guarantees
at least 4.5 m forward progress inside a 1.5 m half-corridor while upright; it
does not claim exact six-metre placement. Do not rename or extrapolate either
anchor.

Do not use repeated fixed-trajectory tuning to hide placement failures. Generate
conditioned demonstrations and train the v2 range/direction/speed policy when a
single teacher cannot cover the declared ball-pose envelope.

Generate a multi-condition teacher dataset, train the supervised initializer,
collect closed-loop DAgger labels, and evaluate the exported ONNX in exact CPU
MuJoCo with:

```bash
PYTHONPATH=training python training/tools/generate_kick_teacher_dataset.py \
  --distances 2 --angles -15 0 15 --ball-x 0 \
  --ball-y -0.04 0 0.04 --requested-speed 1.43 \
  --output-prefix /home/win98/rl_runs/kick-teacher/kick-v2-grid
PYTHONPATH=training python training/tools/train_kick_bc.py \
  /home/win98/rl_runs/kick-teacher/kick-v2-grid.npz \
  --output-prefix /home/win98/rl_runs/kick-bc/kick-v2-grid
PYTHONPATH=training python training/tools/collect_kick_dagger.py \
  /home/win98/rl_runs/kick-bc/kick-v2-grid.onnx \
  /home/win98/rl_runs/kick-teacher/kick-v2-grid.json \
  /home/win98/rl_runs/kick-teacher/kick-v2-grid.npz
PYTHONPATH=training python training/tools/evaluate_kick_onnx.py \
  /home/win98/rl_runs/kick-bc/kick-v2-grid.onnx --trials 20
```

For a complete accepted teacher grid, export a provenance-carrying runtime
table only after held-out evaluation. The exporter preserves a false
promotion flag when the evidence is below the release gate:

```bash
PYTHONPATH=training python training/tools/evaluate_kick_teacher_table.py \
  /home/win98/rl_runs/kick-teacher/kick-v2-grid.json --trials 300 \
  --output /home/win98/rl_runs/kick-teacher/kick-v2-grid-eval.json
PYTHONPATH=training python training/tools/export_kick_residual_table.py \
  /home/win98/rl_runs/kick-teacher/kick-v2-grid.json \
  --evaluation /home/win98/rl_runs/kick-teacher/kick-v2-grid-eval.json \
  --output runtime/apollo/assets/keyframes/kick_residual_table.yaml
```

The checked-in dense residual table remains the 2 m forward-pass executor. In
addition, the source-tree runtime now has discrete 3.5 m and 5 m procedural
teacher anchors selected by target distance and requested speed. The two longer
anchors have nominal exact-physics evidence but no held-out/server promotion,
so they remain experimental and fail closed outside their declared envelopes.
The unpromoted ONNX actor is loaded in shadow mode for developed-versus-base
comparisons; set `APOLLO_LEARNED_KICK_MODE=active` only for a controlled model
A/B experiment.

Exercise one real PPO training/checkpoint path (integration test only):

```bash
PYTHONPATH=training python training/tools/train_kick_smoke.py \
  --impl warp --num-envs 64 --num-timesteps 4096 \
  --run-dir /home/win98/rl_runs/kick-smoke-$(date +%Y%m%d-%H%M%S)
```

The generated smoke checkpoint is deliberately marked non-releasable.  A
successful optimizer update proves pipeline integrity, not soccer performance.

## Live training visualization

Finite soccer-motion PPO runs now write TensorBoard events below each run's
`tensorboard/` directory while preserving the authoritative JSONL metrics and
run manifest. Start the local dashboard in WSL and open the printed localhost
address from Windows:

```bash
conda run -n my3d-rl tensorboard \
  --logdir /home/win98/rl_runs --port 6006 --bind_all
```

The active K1-A phase teacher accepts `--tensorboard-log-dir`; it publishes
best score, generation score, elite mean and search variance as each CEM
generation completes. When `--dataset-output` is supplied, the dataset also
contains the exact CPU `qpos` trace for visual inspection:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/view_soccer_motion_teacher.py /path/to/teacher-dataset.npz \
  --split validation --hold
```

Any retained actor checkpoint can also be replayed under the same exact-CPU
closed-loop dynamics used by the fixed-grid evaluator. This is the visual
checkpoint surface for PPO/BC/DAgger rather than a kinematic teacher trace:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/view_soccer_motion_policy.py \
  /home/win98/rl_runs/paid-k0/corpus-semantic-v4 \
  --checkpoint /path/to/checkpoint --profile soccer_motion_residual_v3 \
  --motion-index 0 --loops 2 --hold
```

The viewer reports the selected motion, terminal reason, joint RMSE and foot
contact agreement. It remains separate from the vectorized accelerator loop;
during formal PPO, TensorBoard is continuous and a checkpoint replay is
launched only at declared evaluation boundaries.

TensorBoard is a measurement surface, not a promotion gate. Candidate choice
still uses held-out exact-CPU, Warp/ONNX parity, three seeds and RCSSServer
replay. The MuJoCo viewer is deliberately separate from accelerated training
so rendering cannot reduce rollout throughput or alter timings.

Exact-CPU grids can add deterministic reset perturbations without sacrificing
paired comparison. `--perturbation-seed` derives a distinct case seed for each
motion/start pair; joint, root-velocity and yaw envelopes are explicit. The
paired comparator includes that seed in its key, so reports from different
perturbation grids cannot be compared accidentally:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/evaluate_soccer_motion_cpu.py /path/to/corpus \
  --checkpoint /path/to/checkpoint --phase-samples 256 \
  --perturbation-seed 20260967 --reset-joint-noise 0.002 \
  --reset-root-velocity-noise 0.005 --reset-yaw-range 0.01 \
  --output /path/to/evaluation.json
```

Changing the perturbation seed creates a disjoint deterministic state grid
while retaining the same phase coverage. This prevents repeated model
selection from exhausting the finite set of motion start frames.

When several PPO runs start from the same checkpoint and execute the same small
update, their parameter coordinates remain aligned. The formal averaging tool
requires identical network signatures, records all source tree hashes, and
averages actor and critic leaves equally. Because this profile disables
observation normalization, it copies the explicitly supplied retained-base
normalizer instead of averaging Brax's still-updated statistic containers:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/average_soccer_motion_checkpoints.py \
  /path/to/seed-a/checkpoint /path/to/seed-b/checkpoint \
  /path/to/seed-c/checkpoint --base-checkpoint /path/to/retained/base \
  --step 196608 \
  --output-dir /home/win98/rl_runs/paid-k1/ppo-average-<name>
```

Parameter averaging is a new candidate, not a way to override a failed
multi-seed gate. It receives its own disjoint selection and confirmation grids.

After a single-motion K1-A gate passes, generate independent teachers for the
entire locked corpus with the resumable batch orchestrator:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/optimize_soccer_motion_teacher_corpus.py \
  /home/win98/rl_runs/paid-k0/corpus-semantic-v4 \
  --checkpoint /path/to/checkpoint --population 64 --generations 8 \
  --max-workers 2 --run-dir /home/win98/rl_runs/paid-k1/k1a-corpus-<name>
```

Each motion retains its own report, dataset, console log and TensorBoard run.
Compatible completed motions are reused after interruption; an incomplete
motion directory is preserved with an `incomplete-*` suffix before retry. The
combined `teacher-corpus.npz` is created only when all 13 per-motion teacher
gates pass, so behavior cloning cannot silently mix rejected demonstrations.

Train the compatible actor from the selected corpus, then collect beta-zero
DAgger labels on student states using the frozen teacher-base policy:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/train_soccer_motion_bc.py /path/to/teacher-corpus.npz \
  --selection-manifest /path/to/selection-manifest.json \
  --base-checkpoint /path/to/base/checkpoint \
  --output-dir /home/win98/rl_runs/paid-k1/k1b-bc-<name>
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/collect_soccer_motion_dagger.py /path/to/corpus \
  --student-checkpoint /path/to/bc/checkpoint \
  --selection-manifest /path/to/selection-manifest.json \
  --source-dataset /path/to/teacher-corpus.npz \
  --output-dir /home/win98/rl_runs/paid-k1/k1b-dagger-<name>
```

The collector deliberately separates the executed student from the label
policy. The label is the frozen teacher-base actor plus its selected
motion-specific phase correction. Adding that correction to the already
distilled student double-counts the teacher and is invalid. DAgger retraining
may freeze the feature extractor with `--trainable-layers output`; it still
requires a new exact-CPU grid that excludes all teacher and DAgger start frames.

State-feedback DAgger additionally enables a short exact-CPU candidate search
with `--state-feedback-horizon`, `--minimum-state-feedback-improvement` and
`--maximum-state-feedback-action-delta`. Formal PPO restoration is deliberately
fail-closed: `train_soccer_motion.py` requires both the restored checkpoint and
the paired exact-CPU comparison that has `curriculum_advance_passed=true`.
The comparison, candidate report and checkpoint-tree hashes are copied into the
run manifest:

```bash
PYTHONPATH=training conda run -n my3d-rl python \
  training/tools/train_soccer_motion.py \
  /home/win98/rl_runs/paid-k0/corpus-semantic-v4 \
  --failure-report /home/win98/rl_runs/paid-k1/semantic-apollo-runtime-prefailure-v2.json \
  --contract training/contracts/soccer_motion_policy_v2.yaml \
  --profile soccer_motion_residual_v3 --stage reference_track \
  --restore-checkpoint /path/to/state-feedback/checkpoint \
  --curriculum-comparison /path/to/paired-comparison.json \
  --run-dir /home/win98/rl_runs/paid-k1/ppo-reference-<name>
```

Brax rounds work to complete optimizer steps and evaluation intervals. The
trainer mirrors that calculation before launch and records requested, effective
and observed final timesteps; `timestep_accounting_passed` must be true before
an experiment can be treated as reproducible evidence.

## Release rule

A checkpoint cannot be integrated until its model manifest, source revisions,
asset hashes, three-seed held-out evaluation, ONNX parity report, and
RCSSServerMJ acceptance result are present. Runtime inference must retain the
existing kick as a safe fallback.

As of 2026-09-05 the runtime keeps the original stable `walk.onnx` for precise
movement and fallback. Source-tree matches use the transition-recovered FastWalkV2 actor
for long, low-turn forward demands and the rapid-turn actor for pure yaw; the
negative-yaw branch is an exact observation/action reflection of the stronger
positive-yaw policy. Their external ONNX hashes, CPU evidence and server traces
are locked in `training/locks/competition_motion_stack_2026_09_05.yaml`. The
parent composition produced four fall events in 1,800 cycles. The promoted
transition-recovered actor reduces the same-duration composition to one fall
event/five GetUp status samples while preserving active forward and turn use.
The composition remains training-active, but a non-zero fall count alone does
not remove a useful ability.
The 0.55 m procedural dribble touch, narrow 4 m procedural shot, and safety-focused
6 m procedural clear are mounted as model-independent fallbacks. The source-tree
WSL launchers also enable the transition-kick ONNX by default, but its frozen
exact-CPU score is only 27/92, so active ownership is restricted to its measured
fixed-2 m slice and every mismatch or inference failure falls back in the same
cycle. Use `APOLLO_LEARNED_KICK_MODE=shadow` when joint ownership is undesired.
