# RL motion reference research

Status: first systematic review and source-code audit complete
Mode: `deep-research` literature review
Search date: 2026-08-30

## Research question

Which published methods, official frameworks, robot assets, and open-source
implementations provide the most reliable and legally reusable path to train a
Booster T1 soccer kick and robust locomotion policy for RCSSServerMJ, given an
8 GB NVIDIA laptop GPU and a 50 Hz ONNX deployment interface?

Subquestions:

1. Which simulator/trainer offers the closest physics and lowest compatibility
   risk on this host?
2. Which observation, action, reward, curriculum, and randomization designs are
   supported for humanoid locomotion and dynamic kicking?
3. Which projects expose training code rather than inference assets only?
4. Which licences permit independent training and redistribution?
5. What sim-to-sim checks best predict RCSSServerMJ deployment behavior?

## Search protocol

Sources searched:

- official GitHub organizations and framework documentation;
- Crossref, Semantic Scholar, arXiv, and publisher/DOI pages;
- RoboCup Soccer Simulation official resources and team-description papers;
- robot-vendor documentation and released assets.

Search concepts:

```text
(Booster T1 OR humanoid) AND
(reinforcement learning OR PPO OR imitation OR residual policy) AND
(kick OR soccer OR locomotion OR get-up OR push recovery) AND
(MuJoCo OR MJX OR Isaac Gym OR Isaac Lab OR sim-to-sim)
```

Date range: 2022-2026 for fast-moving training frameworks, with older seminal
work included only when it defines a method still used by current systems.

Inclusion criteria:

- direct relevance to T1/humanoid low-level control, dynamic ball contact,
  simulator training, or policy deployment;
- accessible original paper, official documentation, or official source tree;
- enough implementation detail to affect this project's design;
- identifiable licence for code/assets when reuse is proposed.

Exclusion criteria:

- vision/strategy-only soccer work with no low-level motion contribution;
- repositories containing inference weights but no relevant training method,
  except when assessing deployment contracts;
- unverifiable papers, derivative summaries, marketing-only claims, or projects
  without source/asset provenance;
- hardware-only results that cannot be mapped to simulator observations.

## Evidence rules

- Technical framework facts must come from primary official documentation.
- Paper metadata is checked through DOI/publisher or Semantic Scholar records.
- Vendor code is treated as potentially conflicted evidence and validated by
  cross-simulator or independent research where possible.
- Training success on another robot/simulator is not treated as proof of
  RCSSServerMJ transfer.
- Unknown compatibility or licence terms remain explicitly unknown.

## Search accounting

Sixty queries were executed. Search endpoints did not expose stable total-hit
counts, so a false PRISMA identified-record count is not reported. DOI, title,
and publication/preprint mappings produced 42 unique primary candidates; 38
were retained for the engineering corpus and four were rejected after full
review. Semantic Scholar rate-limited part of the metadata check with HTTP 429,
so DOI and publisher metadata were used instead of Semantic Scholar IDs.

## Thematic synthesis

### Focused source update: PAiD and `wbc_fsm` (2026-09-01)

The PAiD evidence level has changed since the initial survey. Its
[official repository](https://github.com/TeleHuman/HumanoidSoccer) now releases
the three-stage training implementation, thirteen labeled G1 kick motions and
a recurrent G1 checkpoint under CC BY-NC 4.0. The
[preprint](https://arxiv.org/abs/2602.05310) and
[project page](https://soccer-humanoid.github.io/) support this sequence:
unified motion tracking with adaptive motion/phase failure sampling; resume
while retaining motion objectives and adding lightweight ball/target rewards;
then physics-aware identification/randomization. Its author-reported 91.3%
real-world result is not treated as T1/RCSS evidence.

This is now the strongest directly implementable method reference for the
current striker bottleneck, but not a deployable artifact. PAiD is G1 29-DoF,
its ONNX consumes a 160-value observation plus recurrent state, and its licence
is non-commercial. The immediate adoption is a clean-room T1 motion-skill
stage, followed by target/arrival-speed adaptation; source, motions and weights
are not copied into this repository.

[`wbc_fsm`](https://github.com/ccrpRepo/wbc_fsm) is a G1/Unitree SDK2 C++
deployment example with Passive/FixedStand/Loco/AMP/MJAmp/WBC transitions,
ONNX inference, state history and projected-gravity safety termination. No
top-level licence was found, and it supplies neither ball/goal perception nor
football tactics. It is useful only for independently implementing stronger
Apollo skill lifecycle, shape checks and fallbacks. It is not a replacement
team base. The pinned evidence and corrected R1 route are detailed in
[`paid-wbc-fsm-audit-2026-09-01.md`](paid-wbc-fsm-audit-2026-09-01.md).

### The direct algorithm reference

The closest published method is Xu et al., [Learning Agile Striker Skills for
Humanoid Soccer Robots from Noisy Sensory Input](https://arxiv.org/abs/2512.06571),
and its [official T1 code](https://github.com/Daffan/humanoid-soccer). It trains
a privileged chase teacher, fine-tunes a directional kick, distils a student
with DAgger, then adapts it under perception degradation with constrained RL.
The deployed policy runs at 50 Hz. This structure should guide the project, but
its Isaac Gym stack and checkpoint interface should not be copied.

[Learning Agile Soccer Skills for a Bipedal Robot](https://doi.org/10.1126/scirobotics.adi8022)
independently supports high-frequency control, targeted dynamics randomization,
training perturbations, and teacher/student decomposition. Its robot and
optimizer differ from T1/RCSSServerMJ, so it is method evidence rather than a
deployable-policy source.

### The engineering base

Use pinned [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
with its 23-DoF T1 environment. It provides JAX PPO, asymmetric observations,
MJX-JAX, and MJX-Warp under Apache-2.0. The local Python 3.12 environment has
already loaded JAX on `CudaDevice(0)`, MuJoCo 3.12, and Warp 1.16.

Choose JAX or Warp from local 256/512-environment throughput, memory, contact,
and NaN tests. [MJX's official documentation](https://mujoco.readthedocs.io/en/latest/mjx.html)
warns that accelerator implementations favor large batches and that collision
algorithms can differ from CPU MuJoCo.

[Booster Gym](https://github.com/BoosterRobotics/booster_gym) is valuable for
T1 rewards, observation design, domain randomization, and MuJoCo cross-sim. Its
public stack uses Python 3.8, PyTorch 2.0, CUDA 11.8, and Isaac Gym Preview 4;
the released T1 task has a 12-dimensional leg action rather than this project's
23-dimensional action contract. It is therefore a design source, not the local
training runtime.

### Three training layers and three validation layers

Training should proceed as:

1. freeze or reuse stable locomotion/standing and train a bounded 23-joint
   residual directional-kick teacher;
2. widen the curriculum from contact to direction, strength, approach, and
   perturbation;
3. only if partial observation is limiting, distil a student using measured
   competition delay, field of view, dropout, and noise.

Every release must pass:

1. deterministic CPU MuJoCo physics and actuator checks;
2. fixed-seed MJX-JAX/MJX-Warp training and numerical checks;
3. MuJoCo 3.5 RCSSServerMJ communication, perception, FSM, and 7v7 acceptance.

RCSSServerMJ 0.2.1 uses MuJoCo 3.5, `dt=0.005`, and four substeps per
50 Hz control step. Playground T1 uses MuJoCo >=3.6, `dt=0.002`, and ten
substeps. Equal policy frequency does not imply equal contact trajectories.
The official [soccer world XML](https://gitlab.com/robocup-sim/rcssservermj/-/raw/master/src/rcsssmj/resources/environments/soccer/world.xml)
defines ball radius 0.11 m, mass 0.41 kg, and friction `0.4 0.01 0.01`.

## Audited 2026 T1 striker code

The official repository was audited read-only at commit
`378a12ac7446cd175f973c04e32912eb9acbee10`; it declares Apache-2.0 and contains
one commit, no tests, tags, pretrained weights, ONNX export, or MuJoCo runner.
Its strongest reusable concepts are:

- a four-stage chase/kick/DAgger/constrained-RL curriculum;
- 84 deployable features and a separate 23-value privileged critic state;
- 23 actions in the same T1 joint order, at 50 Hz;
- action delay of 0--18 ms, visual update gaps of roughly 60--140 ms, field of
  view masking, and physics randomization;
- a 9x9 ball-position evaluation grid and explicit accuracy/speed/energy
  metrics;
- separate positive-reward and penalty critics during the final stage.

Do not run or transplant the release unchanged. The audit found:

- the first three configs declare 21 privileged values while the environment
  always produces 23, causing an input-width mismatch;
- DAgger changes from all-teacher to all-student after one iteration, discards
  old data in whole chunks, and does not hold validation samples out;
- the README describes delayed/occluded ball and goal, but the goal is not
  actually delayed or masked in code;
- missing ball data is encoded as `[0, 0]` with no validity or age signal;
- a ball-velocity computation mixes body- and world-frame values;
- the constrained-RL cost includes a positive survival reward and uses a
  violation sign that can label ordinary negative penalties as violations;
- a uniform 1-radian joint residual is too aggressive for this deployment.

The local 90-feature contract improves the missing-observation ambiguity with
ball age and validity. The first policy stays `90 -> 23`; a history encoder and
corrected DAgger are added only after the single-frame baseline is measured.

## Decision matrix

| Candidate | T1/soccer fit | Host fit | Physics fit | Licence | Decision |
|---|---:|---:|---:|---|---|
| Playground + MJX JAX/Warp | high / custom soccer needed | high, GPU import passed | high, version gap remains | Apache-2.0 | primary stack |
| 2026 T1 striker | highest method fit | low, old Isaac Gym | medium | Apache-2.0 | independently port audited concepts |
| Booster Gym | high locomotion fit | low on current WSL/Blackwell | cross-sim available | Apache-2.0 | reward/randomization reference |
| Isaac Lab / booster_train | no ready T1 task | below recommended host resources | cross-engine | permissive, heavy stack | reconsider on >=16 GB VRAM host |
| Direct RCSSServerMJ PPO | exact target | low rollout throughput | exact | MIT + T1 Apache-2.0 | validation/short fine-tuning only |
| ApolloCodebase | strong runtime architecture | already integrated | competition path | GPL, training absent | inference/architecture reference only |
| End-to-end team RL | broad scope | highest cost/risk | unknown | project-controlled | reject for current milestone |

Fallback triggers:

- Warp contact/NaN instability on `sm_120`: use MJX-JAX;
- 512 environments exceed memory: use 256/128, shorter rollouts, and no
  training-time rendering;
- 23-DoF kick destabilizes locomotion: freeze locomotion and reduce residual
  dimensions/scales;
- Playground succeeds but RCSS fails: calibrate actuators, contacts, latency,
  and randomization rather than merely increasing training steps;
- full chase and continual shooting become necessary: add the complete
  teacher/student stages after isolated kick acceptance.

## Licence and reuse boundary

- Playground, MJX-Warp, and Menagerie T1 are Apache-2.0; preserve licence,
  copyright, and NOTICE attribution.
- RCSSServerMJ is MIT and its T1 resource is separately Apache-2.0. The soccer
  PNG textures lack per-file provenance; physics parameters are used without
  redistributing those textures.
- Booster official assets are BSD-3-Clause.
- Daffan/humanoid-soccer is Apache-2.0 at the audited commit, but dependencies
  named in its licence must retain their own attribution.
- ApolloCodebase and FCPCodebase are GPL; BahiaRT Gym is AGPL. Keep them as
  isolated reference/runtime components and do not copy their code or weights
  into the permissive training package.
- Apollo ONNX files have no separate model licence or model card and are not
  used as distillation teachers.

This is a conservative engineering policy, not legal advice.

## Selected 38-source corpus

### T1 and soccer control

1. Xu et al. (2026), [T1 agile striker paper](https://arxiv.org/abs/2512.06571)
   and [official code](https://github.com/Daffan/humanoid-soccer).
2. Haarnoja et al. (2024), [agile bipedal soccer](https://doi.org/10.1126/scirobotics.adi8022).
3. Abreu et al. (2023), [RoboCup skill-set primitives](https://arxiv.org/abs/2312.14360)
   and [FCPCodebase](https://github.com/m-abr/FCPCodebase).
4. Spitznagel et al. (2021), [multi-directional kick learning](https://doi.org/10.1109/ICARSC52212.2021.9429811).
5. Beukman et al. (2024), [RobocupGym](https://arxiv.org/abs/2407.14516).
6. Simões et al. (2022), [BahiaRT Gym](https://doi.org/10.1016/j.simpa.2022.100401).
7. [magmaOffenburg 2022 TDP](https://tdp.robocup.org/wp-content/uploads/tdp/robocup/2022/robocupsoccer-simulation-3d/magmaoffenburg-387/robocup-2022-robocupsoccer-simulation-3d-magmaoffenburg18PnbW4KNL.pdf).
8. [BahiaRT 2023 TDP](https://tdp.robocup.org/wp-content/uploads/tdp/robocup/2023/robocupsoccer-simulation-3d/bahiart-435/robocup-2023-robocupsoccer-simulation-3d-bahiartFo9ESau8mk.pdf).
9. [RoboCup 3D Tools](https://ssim.robocup.org/3d-simulation/3d-tools/).
10. Marew et al. (2024), [biomechanics-inspired kick](https://arxiv.org/abs/2407.14612).
11. Ficht and Behnke (2024), [maximum-impulse kick](https://arxiv.org/abs/2412.01480).
12. [TeleHuman progressive humanoid soccer](https://github.com/TeleHuman/HumanoidSoccer).
13. [RoboNaldo motion-guided shooting](https://github.com/OpenDriveLab/RoboNaldo).

### Humanoid locomotion

14. Wang et al. (2025), [Booster Gym paper](https://arxiv.org/abs/2506.15132)
    and [official code](https://github.com/BoosterRobotics/booster_gym).
15. [MuJoCo Menagerie Booster T1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/booster_t1).
16. Gu et al. (2024), [Humanoid-Gym](https://arxiv.org/abs/2404.05695).
17. Radosavovic et al. (2024), [real-world humanoid locomotion](https://doi.org/10.1126/scirobotics.adi9579).
18. Rudin et al. (2022), [Learning to Walk in Minutes](https://proceedings.mlr.press/v164/rudin22a.html).
19. Margolis and Agrawal (2023), [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html).
20. Miki et al. (2023), [DreamWaQ](https://doi.org/10.1109/ICRA48891.2023.10161144).

### Residual, imitation, and interaction

21. Johannink et al. (2019), [Residual RL](https://doi.org/10.1109/ICRA.2019.8794127).
22. Peng et al. (2018), [DeepMimic](https://doi.org/10.1145/3197517.3201311).
23. Peng et al. (2021), [Adversarial Motion Priors](https://doi.org/10.1145/3450626.3459670).
24. He et al. (2025), [ASAP](https://arxiv.org/abs/2502.01143).
25. Wang et al. (2025), [SkillMimic](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_SkillMimic_Learning_Basketball_Interaction_Skills_from_Demonstrations_CVPR_2025_paper.html).
26. Ji et al. (2024), [ExBody2](https://arxiv.org/abs/2412.13196).
27. Miller et al. (2023), [model-based motion imitation](https://arxiv.org/abs/2305.10989).
28. Kasaei et al. (2021), [Proximal Symmetry Loss](https://arxiv.org/abs/2103.00928).

### Framework, transfer, and deployment

29. Zakka et al. (2025), [MuJoCo Playground](https://arxiv.org/abs/2502.08844).
30. [MuJoCo MJX official documentation](https://mujoco.readthedocs.io/en/latest/mjx.html).
31. Todorov et al. (2012), [MuJoCo](https://doi.org/10.1109/IROS.2012.6386109).
32. Makoviychuk et al. (2021), [Isaac Gym](https://arxiv.org/abs/2108.10470).
33. Mittal et al. (2023), [Orbit](https://doi.org/10.1109/LRA.2023.3270034)
    and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).
34. Freeman et al. (2021), [Brax](https://arxiv.org/abs/2106.13281).
35. Peng et al. (2018), [dynamics randomization](https://doi.org/10.1109/ICRA.2018.8460528).
36. Tan et al. (2018), [sim-to-real agile locomotion](https://doi.org/10.15607/RSS.2018.XIV.010).
37. [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html) and
    [ONNX Runtime C++](https://onnxruntime.ai/docs/get-started/with-cpp.html).
38. Schulman et al. (2017), [PPO](https://arxiv.org/abs/1707.06347).

## Known, unknown, and inferred

Known:

- JAX GPU, MuJoCo, and Warp load locally; the 90/23 contract and direct RCSS
  MjSpec scene tests pass.
- A 64-environment, 4096-step Warp PPO integration run completes a real
  optimizer update and writes a restorable checkpoint without contact-buffer
  overflow after sizing `naconmax` from measured rollout demand.
- Playground contains T1 locomotion but no soccer task.
- Apollo contains 78->23 walk and 75->23 get-up inference assets, not training
  rewards, data, optimizers, or code.

Unknown:

- multi-hour training SPS, complete Warp allocator peak VRAM, and numerical
  stability under the full curriculum;
- high-speed foot-ball contact error from MuJoCo 3.12/MJX to RCSS MuJoCo 3.5;
- the performance loss when the 2026 T1 method is independently ported to
  RCSSServerMJ observations.

Inferred project decisions:

- Playground plus an empirically selected JAX/Warp backend is the best current
  host fit; external papers do not prove this conclusion.
- Training the isolated kick before retraining locomotion should produce faster
  match value because the current competition loop is already stable.
- Apollo weights should not be used as permissively licensed distillation data;
  this is a conservative compliance decision rather than a legal conclusion.
