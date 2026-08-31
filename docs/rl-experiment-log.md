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
