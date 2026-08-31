# Competition runbook

This runbook is the operational path from a clean WSL checkout to a running
seven-player match. Commands assume Ubuntu 22.04 and the repository as the
current directory.

## 1. Environment

Create or refresh the project environment:

```bash
conda env update -n my3d-team -f environment.yml --prune
conda activate my3d-team
python --version
pytest -q
```

The validated interpreter is Python 3.13. The client is installed editable so
source changes take effect without rebuilding.

Install RCSSServerMJ separately from the team environment. The validated
server is 0.2.1 and is exposed through pipx:

```bash
pipx install rcsssmj==0.2.1
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
```

The server package uses Python 3.10 internally; this is intentionally isolated
from the Python 3.13 team environment.

## 2. Pre-match checks

Run all checks before entering the competition launcher:

```bash
pytest -q
python -m compileall -q mujococodebase run_player.py scripts
bash -n scripts/*.sh start7v7.sh
git diff --check
```

The motor-safety tests reject non-finite values, excessive gains, out-of-range
joint targets, and nondeterministic motor ordering. Behaviour tests cover side
normalisation, beam confirmation, stale-ball search, attack state transitions,
and role selection.

## 3. Launch a match

Terminal A — server:

```bash
conda activate my3d-team
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_server.sh realtime
```

Terminal B — our team:

```bash
conda activate my3d-team
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
scripts/run_team.sh My3DTeam 127.0.0.1 60000
```

The launcher starts players 1–7, forwards termination signals, and cleans up
all child processes. Player roles are goalkeeper (1), defenders (2–3),
midfielders (4–6), and primary attacker (7). All tactical targets are expressed
in a canonical frame where our goal is always at negative x.

## 4. Local 7v7 acceptance match

For the automated acceptance gate, do not start a separate server. Execute:

```bash
conda activate my3d-team
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_acceptance_match.sh 600
```

This owns a temporary server, starts `My3D-A` and `My3D-B`, waits for all 14
players, gives the left team a deterministic kickoff, drops the ball for
`PLAY_ON`, and stops each player at the cycle limit. Acceptance criteria are:

1. all 14 clients join without traceback;
2. left and right teams both transition through kickoff into `PLAY_ON`;
3. the ball-facing player cycles through approach, alignment, kick, and
   recovery without sending invalid motor values;
4. all processes exit cleanly when the cycle limit or interrupt is reached.

Set `KEEP_MATCH_LOGS=1` to preserve the server and team logs under `/tmp`.
Set `MATCH_REQUIRE_ATTACK_LOOP=0` only for connection-only diagnostics; the
default gate requires observed `ALIGN`, `KICK`, and `RECOVER` transitions.

### Guarded reference-posture integration

The formal `Walk` entry point can consume the current GMR/v4 posture policy,
but it does not promote that rejected candidate to the main locomotion
controller. The stable policy is computed on every cycle and owns at least 90%
of the target. The reference backend is allowed only for non-goalkeepers in
`PLAY_ON`, for an absolute target at least 3.5 m ahead, with at most 6 degrees
heading error, an upright torso, low entry angular velocity, and valid height.
A burst lasts 16 control cycles (0.32 s), ramps in and out, rate-limits its
joint target, cools down for two seconds, and returns to stable walk in the
same cycle if the posture guard or inference boundary fails.

The backend is disabled by default. On the validated development host, enable
the exact local-only assets with:

```bash
export MY3D_RUN_BACKEND=reference_v4_burst
export MY3D_RUN_MODEL=/home/win98/rl_runs/run-reference-residual-v4-gmr-formal-s139/exports/000001179648.onnx
export MY3D_RUN_REFERENCE=/home/win98/rl_datasets/motion_refs/t1_run2_subject4_gmr_periodic_v1.npz
export MATCH_REQUIRE_RUN_BURST=1
scripts/run_acceptance_match.sh 800
```

Runtime loading requires model SHA-256 `a107ffe6...` and reference SHA-256
`02cd6409...`, plus the exact 80-to-23 ONNX interface and 34-by-23 reference
shape. `MATCH_REQUIRE_RUN_BURST=1` requires at least one activation, zero
posture/inference aborts, and at least 80% complete bursts. The LAFAN-derived
reference is local-only under CC-BY-NC-ND-4.0 and must not be added to the
repository or a redistribution package. Leave `MY3D_RUN_BACKEND` unset for the
release-safe stable path.

For a full match, omit the final cycle limit or run `./start7v7.sh`.

For a WSLg visual demonstration with preserved server, match, and team logs:

```bash
conda activate my3d-team
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_visual_match.sh 3000
```

The viewer runs in real time and closes when the bounded match finishes. Logs
are retained under `artifacts/visual-match-<timestamp>/`, which is ignored by
Git.

## 5. Match behaviour and fallback boundaries

The attacker uses a deterministic finite-state controller:

```text
SEARCH -> APPROACH -> ALIGN -> KICK -> RECOVER -> APPROACH
```

Ball observations are time-limited and low-pass filtered. When the ball is
stale, the robot stops translation and scans instead of walking toward an old
coordinate. The kick is a short learned-walk-policy burst, selected because it
moves the ball repeatedly while keeping the T1 upright. A hand-authored high
energy keyframe was rejected after simulation showed repeatable falls.

One field player owns the ball at a time through deterministic pitch zones:
defenders own the defensive third, midfield lanes own the middle third, and the
forward owns the attacking third. The other players hold support positions,
which avoids a seven-player swarm without depending on radio communication.
All roles react to own and opponent set plays. Falling first routes into the
Apollo learned recovery policy when its submodule model is available, verifies
an upright stable torso, and then holds a zero-velocity learned stance before
normal play resumes. Front/back keyframes remain the no-model fallback. Use
`scripts/run_recovery_scenario.sh front|back|left|right` for deterministic
single-player recovery checks.

## 6. Troubleshooting

- `RCSSServerMJ executable not found`: set `RCSSSERVERMJ_BIN` to the absolute
  pipx executable path.
- Connection refused: start the server first and verify ports 60000/60001.
- Player remains outside the field: do not start walking before beam position
  confirmation; the current decision maker already retries beam placement.
- Ball disappears near the feet: retain head tracking; removing it causes close
  ball observations to expire.
- Apollo model unavailable: initialize git submodules, or set
  `MY3D_APOLLO_GETUP_MODEL=/absolute/path/policy.onnx`. Force the independent
  fallback with `MY3D_GETUP_BACKEND=keyframe`.
- Reference-run asset rejected: compare the model/reference SHA-256 with the
  values above, verify the 80-to-23 ONNX boundary, and keep the backend at
  `stable`; do not bypass the integrity checks.
- A MuJoCo `CTRL` warning appears once when a player joins: RCSSServerMJ 0.2.1
  recompiles the entire MuJoCo model on every `add_players` activation. This
  warning was reproduced at activation while the client output checks remained
  finite. Treat it as an upstream activation warning unless it repeats during
  normal control; a repeated warning requires immediate motor-log inspection.

## 7. Development strategy

Use a staged evidence loop for future motion work:

1. unit-test motor bounds and state transitions;
2. validate a single player in a deterministic ball pose;
3. require upright recovery after every action;
4. run a bounded 7v7 smoke test;
5. only then tune speed, kick energy, or role pressure.

Keep ApolloCodebase pinned under `external/`. Independent concepts should be
implemented behind the project's own interfaces with regression tests. The
only active reuse is the optional get-up model adapter; follow
`apollo-integration.md` and resolve GPL distribution obligations before making
a combined competition archive.
