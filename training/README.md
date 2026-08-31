# My3DProject motion training

This directory contains reproducible training inputs and deployment contracts.
Generated runs belong under `/home/win98/rl_runs` and are not committed.

The first task is `kick_policy_v1`: a 50 Hz, ball-aware residual joint-position
policy for Booster T1. It is intentionally separate from the deterministic team
decision maker. See [`../docs/rl-training-plan.md`](../docs/rl-training-plan.md)
for the acceptance gates.

Fast locomotion has two immutable contracts: `run_policy_v1` preserves the
deployed 78-value actor; `run_policy_v2` appends cosine/sine gait phase for an
80-value actor while keeping the same 23 actions. Neither experimental model
is the competition default until every release gate passes.

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

Before a retargeted running clip can enter motion-prior training, validate its
50 Hz T1 arrays, provenance and aerial phase:

```bash
PYTHONPATH=training python training/tools/validate_motion_reference.py \
  /path/to/t1-run-reference.npz --output /tmp/t1-run-reference-report.json
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

As of 2026-08-31 the motion pipeline, true-flight evaluator and periodic R1
reference work, but no running checkpoint passes the CPU gate. The competition
runtime therefore continues to use the original stable `walk.onnx`.
