# WSL simulation and Windows browser match console

## Purpose

The web match console is the primary interactive visualization for local 7v7
development. RCSSServerMJ, MuJoCo, and all fourteen Apollo players stay in
Ubuntu 22.04 under WSL. A normal Windows browser receives rendered frames over
loopback and sends explicit operator controls back to the simulator.

This separation avoids WSLg's window-remoting path. It also makes display
behavior independent from `/mnt/shared_memory`, RDP/RAIL copy mode, the active
Windows virtual desktop, and X window placement.

This is an automation-safe alternative, not a removal of native rendering.
Launching `rcssservermj` or `scripts/run_visual_match.sh` from an interactive
WSL terminal may still produce a normal integrated Windows window and remains
useful for direct GLFW operation.

## Data path

```text
Apollo A/B players (14 C++ processes)
            │ agent actions/perceptions
            ▼
RCSSServerMJ + MuJoCo in WSL
            │ EGL off-screen RGB frames
            ▼
bounded JPEG encoder → local MJPEG endpoint
            │ http://127.0.0.1:8765
            ▼
Windows Edge / Chrome
            │ validated JSON controls
            └──────────────► camera / pacing / monitor command queues
```

The browser never executes team decisions. Apollo's action contracts, motion
fallbacks, strategy coordination, and match telemetry remain authoritative.

## Setup and launch

Install Pillow into the same pipx environment that owns RCSSServerMJ 0.2.1:

```bash
/home/win98/.local/pipx/venvs/rcsssmj/bin/python -m pip install \
  -r requirements-web-match.txt
```

Start a ten-minute-equivalent, 30,000-cycle self-play session:

```bash
scripts/run_web_match.sh 30000
```

The launcher treats the rules engine's `GameOver` as a normal terminal state.
It keeps the final browser frame visible for 15 seconds, then closes all player
and server processes instead of waiting for unused client cycles. Override the
hold with `MATCH_GAME_OVER_HOLD`, or set `MATCH_STOP_ON_GAME_OVER=0` only for a
deliberate post-match simulator investigation.

`MATCH_INITIAL_PLAY_TIME=299` is available for a bounded launcher-lifecycle
test. It is a diagnostic setting and must not be used as match evidence.

Open `http://127.0.0.1:8765/` in a Windows browser. The default run directory
is `$HOME/rl_runs/apollo-web-match-<timestamp>`, keeping large evidence away
from the C drive.

Useful launch settings:

```bash
MATCH_WEB_PORT=8876 MATCH_RENDER_INTERVAL=5 \
MATCH_RENDER_WIDTH=1280 MATCH_RENDER_HEIGHT=720 \
scripts/run_web_match.sh 30000
```

Lower render intervals produce more browser frames but leave less CPU/GPU time
for the simulation. The default interval of four targets 12.5 frames per
simulated second on a 50 Hz server. Simulation pacing remains authoritative;
render frames may be dropped rather than slowing the physics clock.

The existing experimental action switches are forwarded unchanged. For
example, a validated local high-speed-walk asset can be opted in with the same
environment variables used by the acceptance launcher. Its hash and fallback
checks are still enforced.

## Native interaction map

| Input | Operation |
| --- | --- |
| Left mouse drag | Rotate the MuJoCo free camera |
| Right mouse drag | Pan the camera |
| Mouse wheel | Zoom |
| `Tab` | Switch between static field and ball-follow cameras |
| `K` / `J` | Grant kickoff to the left / right team |
| `B` | Drop the ball and continue |
| `Space` | Pause or resume simulation |
| `1`, `2`, `4` | Set bounded simulation speed |
| `F` | Enter or leave browser fullscreen |
| `H` | Hide or restore the console HUD |

Camera motion remains available while physics is paused: the server performs a
control-only monitor refresh without advancing the simulation frame. Single
step consumes exactly one pending physics cycle and stays paused.

## Safety boundary

- The HTTP server binds to loopback only.
- `/api/control` accepts a small JSON body and a closed action vocabulary.
- Arbitrary RCSSServerMJ symbolic expressions, shell commands, filesystem
  paths, and process identifiers are rejected.
- Speed is restricted to `0.25`, `0.5`, `1`, `2`, or `4`.
- Camera deltas and zoom values are finite and bounded.
- Kickoff, drop-ball, and center-ball use typed RCSSServerMJ commands.

Do not bind the console to a LAN address without authentication, request
origin checks, and an explicit operator authorization design.

## Troubleshooting

- No page: check `server.log` in the printed run directory and verify the web
  port is unused.
- Page but no first frame: verify `MUJOCO_GL=egl` and run the server-environment
  import check printed by `run_web_match.sh`.
- Browser frame rate is low but match time is correct: increase
  `MATCH_RENDER_INTERVAL` to 5 or 6; do not enable synchronous server mode for
  a visual match.
- Match time is intentionally paused: use Space or the Continue button.
- WSLg shows an invisible native window: close that compatibility run and use
  this console; the web path does not consume WSLg.
- Port 8765 is already used: set `MATCH_WEB_PORT` to another loopback port.
