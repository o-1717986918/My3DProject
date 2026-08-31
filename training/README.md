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

The checked-in 15-node table is experimental and defaults off. It may be
exercised with `--enable-parameterized-kick`; the C++ runner accepts only its
measured 2 m forward-pass envelope and otherwise uses the stable fallback.

Exercise one real PPO training/checkpoint path (integration test only):

```bash
PYTHONPATH=training python training/tools/train_kick_smoke.py \
  --impl warp --num-envs 64 --num-timesteps 4096 \
  --run-dir /home/win98/rl_runs/kick-smoke-$(date +%Y%m%d-%H%M%S)
```

The generated smoke checkpoint is deliberately marked non-releasable.  A
successful optimizer update proves pipeline integrity, not soccer performance.

## Release rule

A checkpoint cannot be integrated until its model manifest, source revisions,
asset hashes, three-seed held-out evaluation, ONNX parity report, and
RCSSServerMJ acceptance result are present. Runtime inference must retain the
existing kick as a safe fallback.

As of 2026-08-31 the Holosoma and independent GMR motion pipelines, true-flight
evaluator, periodic projector, standard/legacy ONNX exporter and exact CPU
acceptance loop work. No running checkpoint passes the CPU completion gate, so
the competition runtime continues to use the original stable `walk.onnx`.
