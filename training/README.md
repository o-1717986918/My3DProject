# My3DProject motion training

This directory contains reproducible training inputs and deployment contracts.
Generated runs belong under `/home/win98/rl_runs` and are not committed.

The first task is `kick_policy_v1`: a 50 Hz, ball-aware residual joint-position
policy for Booster T1. It is intentionally separate from the deterministic team
decision maker. See [`../docs/rl-training-plan.md`](../docs/rl-training-plan.md)
for the acceptance gates.

## Environments

- `my3d-team`: competition runtime; never install training packages here.
- `my3d-rl`: Python 3.12 training and evaluation environment.

Create the base environment with:

```bash
conda env create -f training/environment.yml
```

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
