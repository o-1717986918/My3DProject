# Apollo C++ competition runbook

This is the authoritative path from a clean WSL checkout to a running or
packaged seven-player team. Commands assume Ubuntu 22.04 and the repository as
the current directory.

## 1. One-time host setup

```bash
sudo apt update
sudo apt install -y build-essential cmake libyaml-cpp-dev curl
pipx install rcsssmj==0.2.1
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
```

`my3d-team` may be used for the Python monitor helper and legacy regression
suite. The C++ agents themselves do not depend on Conda.

## 2. Reproducible build

```bash
scripts/bootstrap_apollo_runtime.sh
scripts/build_apollo_runtime.sh
ctest --test-dir runtime/apollo/build --output-on-failure
```

Bootstrap pins ONNX Runtime 1.22.0 and rejects a mismatched archive digest.
The runtime source comes from the online Apollo repository at the commit in
`runtime/apollo/UPSTREAM.md`; no locally modified submodule is used as the
import source.

## 3. Pre-match gates

```bash
ctest --test-dir runtime/apollo/build --output-on-failure
bash -n start.sh start7v7.sh kill.sh build_binary.sh scripts/*.sh runtime/apollo/*.sh
git diff --check
MATCH_REQUIRE_KICK=1 APOLLO_STATUS_INTERVAL=10 \
  scripts/run_apollo_acceptance_match.sh 1200

APOLLO_ENABLE_PARAMETERIZED_KICK=1 MATCH_PASS_SCENARIO=1 \
  MATCH_REQUIRE_PASS=1 MATCH_REQUIRE_KICK=1 \
  APOLLO_STATUS_INTERVAL=5 scripts/run_apollo_acceptance_match.sh 600
```

The 7v7 gate owns a temporary server, starts two complete teams, transitions
through kickoff into `PlayOn`, and checks:

1. 14 connections, joins, `PlayOn` observations, and clean exits;
2. no crash, fatal client error, server exception, or illegal-defense foul;
3. at least one `KickForward`, `KickStabilize`, or `KickHold` sample when
   `MATCH_REQUIRE_KICK=1`;
4. bounded process shutdown and preserved logs when requested.

The second command is the deterministic strategy-to-contact gate. It retains
all 14 players and uses monitor placement only to create an open 2v0 lane
inside the full match. It additionally requires candidate selection,
`Proposed/Ready` coordination, an exact `TargetedPass` kick mode, and at least
0.10 m of ball progress projected toward the declared target. Set
`KEEP_MATCH_LOGS=1`, then use `scripts/analyze_apollo_pass.py` for the detailed
outcome record. This threshold detects physical wiring; promotion to a useful
pass still requires the 2--5 m, 16/20 corridor gate.

`RCSSServerMJ 0.2.1` currently prints one MuJoCo `CTRL` warning during each
player activation while recompiling the full model. The gate reports this
separately; a warning repeated during normal control requires investigation.

## 4. Start a match

Terminal A:

```bash
scripts/run_server.sh realtime 60000 60001
```

Terminal B:

```bash
MY3D_TEAM_NAME=My3DTeam ./start.sh 127.0.0.1 60000
```

The launcher starts exactly players 1-7 with a small stagger to avoid the
server's simultaneous ONNX-client connection race. It forwards interruption
and waits for every player. Use `./kill.sh` from another terminal for explicit
team shutdown.

## 5. Visual 7v7 in a Windows browser

Install the one stream-encoding dependency into the existing RCSSServerMJ
pipx environment:

```bash
/home/win98/.local/pipx/venvs/rcsssmj/bin/python -m pip install \
  -r requirements-web-match.txt
```

Launch the complete Apollo self-play match in WSL:

```bash
scripts/run_web_match.sh 30000
```

Open `http://127.0.0.1:8765/` in Windows Edge or Chrome. MuJoCo renders through
EGL without creating an X window; the browser receives an MJPEG stream and
transmits only validated local controls. It supports mouse camera operations,
the native Tab/K/J/B keys, pause, single-step, 0.5--4x pacing, HUD hiding, and
fullscreen. Match logs are written to `$HOME/rl_runs`, not the C drive.
At `GameOver`, the launcher holds the final frame for 15 seconds and then
reaps all fourteen players and the server; it does not wait for the remaining
client cycle budget.

The browser endpoint binds to `127.0.0.1` by default. Do not change
`--web-host` to a non-loopback address without adding authentication and a
network access policy. Full details are in `web-match-console.md`.

### WSLg compatibility path

With WSLg available:

```bash
scripts/run_visual_match.sh 3000
```

This is the same strict Apollo self-play path with rendering enabled. Evidence
is retained under `artifacts/apollo-visual-match-<timestamp>/`.
When `rcssservermj` works from an interactive WSL terminal, this native path
remains fully supported and retains GLFW's direct window controls. A process
launched by a background Windows host may enter a different WSLg/RDP window
session even on the same machine; that does not imply the interactive WSLg
session is broken.

If the taskbar entry exists but no window is visible and the title contains
`[WARN:COPY MODE]`, use the browser path. That symptom is below the project at
the WSLg RDP/RAIL shared-memory layer; restarting the match does not repair it.

To require telemetry proving that the source-tree launch reached both active
locomotion specialists:

```bash
APOLLO_ENABLE_FAST_WALK=1 \
APOLLO_ENABLE_RAPID_TURN=1 \
MATCH_REQUIRE_FAST_WALK=1 \
MATCH_REQUIRE_RAPID_TURN=1 scripts/run_web_match.sh 30000
```

`run_policy_v2` is a historical artifact/contract identifier. Its locked
rollouts have zero flight phase, so runtime telemetry and project documentation
call the capability `FastWalkV2`, not running. The launcher rejects a missing
model and any model whose SHA-256 differs from
`6214b656c28f0b95300287e5e3a26508078a6a8d036dbeda0ec5130051a190d6`.

The source-tree WSL launchers mount parameterized contact, the bounded active
learned kick, transition-recovered FastWalkV2 forward specialist and RapidTurnV1 by
default:

```bash
scripts/run_web_match.sh 30000
```

Set `APOLLO_ENABLE_FAST_WALK=0` or `APOLLO_ENABLE_RAPID_TURN=0` for independent
stable-walk ablations. The learned kick actor controls joints only inside its measured fixed-2 m envelope;
residual-table or procedural contact remains the same-cycle fallback. The
launcher checks the selected r2 actor SHA-256
`b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1`.
Use `APOLLO_LEARNED_KICK_MODE=shadow` for inference-only evidence. Stable walk
is already the default; set `APOLLO_ENABLE_PARAMETERIZED_KICK=0` to disable
experimental contact as well, in which case learned kick defaults to off.

## 6. Package and inspect

```bash
./build_binary.sh My3DTeam
tar -tzf build/My3DTeam.tar.gz | sort
```

After extraction:

```bash
./My3DTeam/start.sh 127.0.0.1 60000
./My3DTeam/kill.sh
```

The archive includes corresponding source, build definitions, runtime assets,
license, and notices. Before delivery, test the extracted archive against the
same server rather than relying only on the source-tree executable.

## 7. Motion promotion policy

Apollo's existing self-contained learned walk and get-up networks are the
release baseline. New motion models are never swapped in only because their
training reward is higher. Promotion requires:

1. an explicit observation/action/decoder contract and asset hashes;
2. deterministic ONNX parity and finite, clamped joint targets;
3. forward/turn/stop/disturbance single-agent simulator tests;
4. multiple seeds with uprightness, fall-rate, speed, and energy criteria;
5. strict 7v7, then visual 7v7, with same-cycle fallback retained.

The source-tree FastWalkV2/RapidTurnV1 composition is usable but still under
training. It uses
the exact 80-to-23 observation/decoder contract and full 21-body-joint policy
targets, exact T1 joint-limit clamping, and Apollo head tracking. The recovered
parent composition produced 47 FastWalk, 163 rapid-turn samples and four
independent falls in 1,800 cycles. After transition-recovery training, the same
duration produced 39 FastWalk and 161 rapid-turn samples with one independent
fall (five GetUp status samples), followed by successful recovery. Stable-walk
control produced no falls. The composition remains enabled because zero falls
are not a prerequisite for useful play; continued promotion compares
time-to-position benefit against recovery loss. Portable binary defaults remain
off because the ONNX files are external-local assets.

The current v4/GMR result stays in the training workbench because it needs an
external restricted reference and its 80-value input is incompatible with the
Apollo 78-value runtime observation. See `apollo-policy-migration.md`.

## 8. Troubleshooting

- ONNX dependency missing or corrupt: rerun
  `scripts/bootstrap_apollo_runtime.sh`; it verifies the official archive.
- `yaml-cpp` missing: install `libyaml-cpp-dev`.
- Connection reset during team startup: use the project launcher, which
  staggers 14 local clients; do not launch all processes in one scheduler slice.
- Illegal defense: rerun `field_geometry_test`; field players must stay outside
  the inclusive 4.0 m goalkeeper-area boundary.
- No kick observed in a short gate: use 1200 or more cycles and status interval
  10 before treating it as a regression.
- High-speed-walk model rejected at launch: verify the exact phase-v2 path and
  SHA-256; do not bypass the hash gate.
- Excess get-up activity with `FastWalkV2`: preserve the stable fallback and
  collect per-player gate/status timelines for transition-replay training;
  compare time-to-target benefit with recovery loss rather than applying a
  zero-fall rule.
