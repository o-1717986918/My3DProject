# Robot-soccer action research and adoption record

Last source refresh: 2026-08-31

Rule: refresh repository HEADs and primary papers before every training
milestone; do not turn remembered hyperparameters into project facts.

## Decision

The actionable stack for this project is:

1. exact RCSSServerMJ physics and deployment contract as the acceptance truth;
2. verified locomotion teacher warm-start instead of random-policy PPO;
3. phase-aware gait observation and conservative foot-edge contact geometry;
4. a retargeted T1 running-motion tracking prior;
5. low-noise velocity/football task fine-tuning;
6. teacher/student adaptation only after the privileged policy passes motion
   and server gates.

Steps 1–3 are implemented. Step 4 is now the blocking research item because
reward-only runs remain stable high-speed walking and show no sustained aerial
phase.

## Primary-source comparison

| Source | Current evidence | Adopt | Boundary |
|---|---|---|---|
| Daffan et al., ICRA 2026 humanoid striker | T1 teacher PPO, explicit cosine/sine gait phase, geometric foot-edge contact, later DAgger/P3O adaptation | phase observation, 1–2 Hz gait scheduling, swing reward, teacher-first curriculum | Isaac Gym code is not the competition physics; evaluate independently in RCSS |
| Booster Gym | T1 foot-edge height contact and sim-to-sim deployment pipeline | oriented foot-box lowest-point proxy and CPU contact calibration | reference settings only; do not transplant an unverified policy |
| Holosoma / Fast humanoid locomotion | Apache-2.0, T1 support, MJWarp, velocity and whole-body tracking, LAFAN retargeting pipeline | preferred source for a T1 running reference and tracking prior | isolate its environment; current upstream issue reports mean data/retarget output must be hashed and audited |
| RoboNaldo | motion tracking prior followed by staged task adaptation; recommends small noise on resume | prior → task training boundary, immutable stage checkpoints, small exploration | released motion is a G1 kick, not a T1 run; reuse method, not motion |
| PAiD HumanoidSoccer | progressive motion tracking then soccer action learning | corroborates the progressive design | non-commercial restrictions: method reference only |
| ApolloCodebase | GPL C++ behavior-tree and strategy-network runtime with the familiar 78→23 walk boundary | isolated architecture and inference reference | GPL separation; no code copied into the permissive trainer |

Current pinned commits and reuse restrictions are machine-readable in
`training/locks/sources.yaml`.

## What the experiments established

The verified teacher reaches 1.6 m/s and survives ten seconds, but turns in a
large arc. Increasing flight reward generated a misleading training metric and
reduced variable-command stability. After replacing the proxy and adding a
phase-aware 80-value actor, the selected policy reaches 1.50 m/s with full
held-out survival and materially lower drift, but exact CPU contacts still show
no sustained flight. This falsifies the assumption that reward shaping alone
will convert the existing walking teacher into a run on the current budget.

## Next-stage implementation contract

The motion-prior stage must produce a versioned T1 trajectory at 50 Hz with:

- root position and quaternion;
- all 23 joints in `run_policy_v2` order and radians;
- root linear/angular velocity and joint velocity, derived reproducibly if not
  supplied;
- left/right contact labels from the exact RCSS foot geoms;
- at least one ≥10 ms two-foot aerial interval per cycle;
- source URL, source commit/dataset version, license, conversion command and
  SHA-256;
- RCSS replay report: finite state, joint limits, no non-foot collision, foot
  slip and tracking error.

Training sequence:

1. retarget a human run clip to the RCSS T1 with Holosoma in an isolated Conda
   environment;
2. reject the clip if exact RCSS replay cannot satisfy morphology and safety
   checks;
3. train pose/root/contact tracking with no football task reward;
4. require deterministic motion tracking and ONNX parity;
5. fine-tune local velocity tracking at `1.0..1.8 m/s` with small policy noise;
6. add lateral/yaw commands, perturbations and action delay;
7. test three seeds, CPU MuJoCo, RCSS single-player, then 7v7.

## Search-quality policy

- Academic claims require an original paper or official project page.
- Implementation claims require the official repository at a recorded commit.
- Community/Bilibili videos may reveal a useful implementation lead, but a
  video is not evidence of observation layout, physics parity or reproducible
  performance. Promote a lead only after locating source/config/checkpoint and
  license information.
- A model is called "running" only when exact contact evaluation demonstrates
  the aerial-phase gate; visual appearance or speed alone is insufficient.
