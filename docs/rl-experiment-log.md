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

Final automated checks: four training-side tests and 29 competition-runtime
tests pass.  The custom environment also resets and steps with finite rewards
on both JAX and Warp under the final dependency lock.

Not passed yet:

- policy learning curve or actual kick performance;
- three-seed, 200-held-out-episode acceptance;
- high-speed MuJoCo/MJX versus RCSSServerMJ trajectory bounds;
- ONNX export and source-versus-ONNX parity;
- competition client feature-flag integration and full match regression.
