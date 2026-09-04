# My3DProject RoboCup 3D team

My3DProject is a Linux C++17 client for the RCSSServerMJ RoboCup Soccer
Simulation 3D 7v7 league. The competition runtime is based on a fresh online
import of [ApolloCodebase](https://github.com/XiangruiJiang/ApolloCodebase) at
commit `71018c968969d6e55130b0e1987cd5b4f5c3b4df`, extended with this project's
validated match and action work.

The project-level objective is an evidence-backed complete and excellent robot
football team, not merely a client that can finish a match. The capability
route focuses on excellent locomotion and ball skills, goalkeeper and
role-specific completeness, coordinated attack/defense and set plays, and
opponent-diverse paired evaluation. Competition infrastructure is maintained
as a non-regression baseline rather than treated as a team capability. The
quantitative C0--C4 route is in `docs/team-excellence-roadmap.md`.

The default path now reuses Apollo's behavior tree, dynamic role assignment,
team communication, obstacle-aware walk planner, learned 78-to-23 walking
policy, and learned get-up policy. My3D additions currently include bounded
test operation, machine-readable match telemetry, a migrated stable
approach/kick/recover action, legal defensive-kickoff placement, strict 7v7
acceptance, source-complete deployment packaging, and a first coordinated
direct/leading pass loop with typed intent, deterministic evaluation, receiver
readiness, safe fallback, and physical ball-progress telemetry.

The former Python client and the reinforcement-learning tools under
`training/` remain in the repository as reference implementations and the
motion-development workbench. They are not the default competition runtime.

## Validated platform

- Windows WSL2, Ubuntu 22.04, Linux x86-64
- GCC with C++17 and CMake 3.20+
- RCSSServerMJ 0.2.1, `fifa7vs7`, `ssim26`
- ONNX Runtime 1.22.0 (downloaded and SHA-256 verified by the bootstrap script)
- `libyaml-cpp-dev`

Install native build requirements once:

```bash
sudo apt update
sudo apt install -y build-essential cmake libyaml-cpp-dev curl
```

The existing `my3d-team` Conda environment remains useful for the server and
diagnostic scripts; `my3d-rl` remains the isolated training environment.

## Build and test

From the repository directory inside WSL:

```bash
scripts/bootstrap_apollo_runtime.sh
scripts/build_apollo_runtime.sh
ctest --test-dir runtime/apollo/build --output-on-failure
```

The bootstrap script downloads only the official ONNX Runtime 1.22.0 Linux x64
archive and verifies the pinned digest before extraction. Upstream source
provenance is recorded in `runtime/apollo/UPSTREAM.md`.

## Run

Start the server in terminal A:

```bash
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_server.sh realtime
```

Start the seven-player team in terminal B:

```bash
MY3D_TEAM_NAME=My3DTeam ./start.sh 127.0.0.1 60000
```

Stop the team with `./kill.sh`. For a strict, bounded two-team acceptance
match, including a required real kick:

```bash
MATCH_REQUIRE_KICK=1 APOLLO_STATUS_INTERVAL=10 \
  scripts/run_apollo_acceptance_match.sh 1200
```

The gate requires all 14 clients to connect, join, reach `PlayOn`, exit
cleanly, produce no client/server fatal errors, and commit no illegal-defense
foul. Set `KEEP_MATCH_LOGS=1` to preserve evidence under `/tmp`.

To exercise the complete strategy-to-contact pass contract in a deterministic
full 7v7 process:

```bash
APOLLO_ENABLE_PARAMETERIZED_KICK=1 MATCH_PASS_SCENARIO=1 \
  MATCH_REQUIRE_PASS=1 MATCH_REQUIRE_KICK=1 \
  APOLLO_STATUS_INTERVAL=5 KEEP_MATCH_LOGS=1 \
  scripts/run_apollo_acceptance_match.sh 600
```

Analyze the preserved status logs with:

```bash
conda run -n my3d-team python scripts/analyze_apollo_pass.py \
  /tmp/my3d-apollo-match.*/My3D-*.log
```

This gate requires a targeted command and measurable target-direction ball
progress. It is not yet a claim of useful pass distance or receiver control.

For the preferred visualization path, keep the simulator in WSL and open the
local console in a normal Windows Edge or Chrome window. This path uses EGL
off-screen rendering and does not depend on WSLg:

```bash
/home/win98/.local/pipx/venvs/rcsssmj/bin/python -m pip install \
  -r requirements-web-match.txt
scripts/run_web_match.sh 30000
```

Then open `http://127.0.0.1:8765/` in Windows. The browser reproduces the
native MuJoCo camera controls (left-drag rotate, right-drag pan, wheel zoom,
Tab camera switch, K/J kickoff, and B drop-ball) and adds pause, single-step,
bounded 0.5--4x pacing, fullscreen, and a live scoreboard. Logs remain on the
Linux filesystem below `$HOME/rl_runs/apollo-web-match-*`, avoiding C-drive
growth. See `docs/web-match-console.md` for architecture and controls.

`scripts/run_visual_match.sh` remains available as a WSLg compatibility path.
It is not recommended when the window title contains `[WARN:COPY MODE]` or
`/mnt/shared_memory` is unavailable.

### Experimental full high-speed walk

The retained phase-v2 asset has the historical contract name
`run_policy_v2`, but its locked evaluations contain no flight phase. The
competition runtime therefore exposes it as **high-speed walking** and reports
`motion=FastWalkV2`; it is not described or scored as running.

The backend applies the policy's complete joint-position output to all 21 body
joints (head tracking remains owned by Apollo). It is not a low-weight posture
blend. Long forward travel uses the phase policy continuously; goalkeeper,
near-target braking, lateral/reverse travel, sharp turns and unsafe posture use
the stable Apollo walk/get-up fallback. The local model is SHA-256 locked to
`c8a2f80b08a82a41cebaadc53c09467722a821edfc521e4a0d6921e1d481415b`:

```bash
export APOLLO_ENABLE_FAST_WALK=1
export APOLLO_FAST_WALK_MODEL="$HOME/rl_runs/run-phase-v2-formal-s71-20260831-01/policy-best.onnx"
export MATCH_REQUIRE_FAST_WALK=1
scripts/run_web_match.sh 30000
```

This remains an opt-in experimental capability, not the packaged default. A
900-cycle combined 7v7 gate produced 1,374 `FastWalkV2` status samples, 25
parameterized-kick samples, 210 pass-plan samples and one physical pass contact
with 14/14 clean exits, but also 499 get-up samples at a two-cycle status
interval. The deployment is complete enough for visual/domain-gap collection;
its fall rate is not yet competition-release quality.

### Mounted learned-kick path and all-capabilities profile

The runtime accepts a `kick_policy_v3` `[1,98] -> [1,23]` actor in shadow or
explicit active mode. The current best transition actor is only 27/92 on its
frozen exact-CPU set with one fall, so the recommended profile evaluates it in
shadow while the stronger residual/procedural path owns the joints:

```bash
export APOLLO_ENABLE_PARAMETERIZED_KICK=1
export APOLLO_LEARNED_KICK_MODE=shadow
export APOLLO_LEARNED_KICK_MODEL="$HOME/rl_runs/kick-transition-dagger-r2-bc-s10002/policy.onnx"
export APOLLO_ENABLE_FAST_WALK=1
export APOLLO_FAST_WALK_MODEL="$HOME/rl_runs/run-phase-v2-formal-s71-20260831-01/policy-best.onnx"
scripts/run_web_match.sh 30000
```

This mounts every currently useful path: stable walk/get-up remain available,
FastWalkV2 owns supported long forward travel, residual and procedural contact
remain executable fallbacks, and the learned kick actor produces deployment
evidence without silently replacing them. Set `APOLLO_LEARNED_KICK_MODE=active`
only for controlled experiments; it is not a promotion claim.

## Package

Build a self-contained competition archive:

```bash
./build_binary.sh My3DTeam
```

The result is `build/My3DTeam.tar.gz`. It contains the executable, ONNX assets,
required non-system libraries, launch/kill scripts, GPL license, provenance,
and corresponding C++ source/build definitions. The packaged default team name
can still be overridden with `TEAM_NAME=...` at launch.

## Development boundaries

- `runtime/apollo/` is the authoritative competition runtime.
- `mujococodebase/` is the retained Python reference, useful for comparing
  behavior and for rapid experiments.
- `training/` owns RL environments, policy contracts, research locks, and
  export/evaluation tools. Large or license-restricted artifacts stay outside
  Git.
- A policy reaches the C++ runtime only after interface, finite-output,
  uprightness, simulator, and multi-seed 7v7 gates pass.

The existing experimental v4 running candidate is an 80-to-23 residual policy
that also needs a local-only motion-reference artifact. It is not wire-compatible
with Apollo's self-contained 78-to-23 walk policy and remains rejected for
release. The migration contract and efficient next training route are in
`docs/apollo-policy-migration.md`.

Operational details and the current evidence are in
`docs/competition-runbook.md` and `docs/validation.md`. Research and reference
comparisons remain under `docs/`.

Coordinated strategy development and evidence-based RoboCup2D migration are
active. The first one-step pass loop, exact local-source audit, validation
limits, and required directional-kick training are documented in
`docs/strategy-migration-implementation.md`; the complete architecture and
strategy gates remain in `docs/strategy-development-plan.md`. Both are
workstreams under the final `docs/team-excellence-roadmap.md` objective.

## License and attribution

The combined project is distributed under GPL-3.0-or-later; see `LICENSE`.
Apollo upstream notices are retained in `runtime/apollo/LICENSE.md` and
`runtime/apollo/UPSTREAM.md`. The project also retains the history and
acknowledgments of the earlier BahiaRT/Magma/FCPortugal-derived Python work.
