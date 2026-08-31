# My3DProject RoboCup 3D team

My3DProject is a Linux C++17 client for the RCSSServerMJ RoboCup Soccer
Simulation 3D 7v7 league. The competition runtime is based on a fresh online
import of [ApolloCodebase](https://github.com/XiangruiJiang/ApolloCodebase) at
commit `71018c968969d6e55130b0e1987cd5b4f5c3b4df`, extended with this project's
validated match and action work.

The default path now reuses Apollo's behavior tree, dynamic role assignment,
team communication, obstacle-aware walk planner, learned 78-to-23 walking
policy, and learned get-up policy. My3D additions currently include bounded
test operation, machine-readable match telemetry, a migrated stable
approach/kick/recover action, legal defensive-kickoff placement, strict 7v7
acceptance, and source-complete deployment packaging.

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

Launch the same Apollo runtime with WSLg visualization:

```bash
scripts/run_visual_match.sh 3000
```

Logs are written below `artifacts/apollo-visual-match-<timestamp>/` and are
ignored by Git.

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

The accepted next-stage direction is coordinated strategy development and
evidence-based RoboCup2D algorithm migration. Its architecture, implementation
order, promotion gates, telemetry contract, and release definition are in
`docs/strategy-development-plan.md`.

## License and attribution

The combined project is distributed under GPL-3.0-or-later; see `LICENSE`.
Apollo upstream notices are retained in `runtime/apollo/LICENSE.md` and
`runtime/apollo/UPSTREAM.md`. The project also retains the history and
acknowledgments of the earlier BahiaRT/Magma/FCPortugal-derived Python work.
