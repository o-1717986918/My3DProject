# My3DProject competition client

This repository is a Python 3.13 team client for the RCSSServerMJ RoboCup 3D
Soccer Simulation League. It started from the BahiaRT MuJoCo base and now adds
a seven-player match controller, canonical team coordinates, safe motor output,
ball tracking, role formations, set-play handling, and a repeatable
approach-align-kick-recover loop.

Development and competition execution are supported in WSL2 Ubuntu 22.04 with
the `my3d-team` Conda environment. See `docs/competition-runbook.md` for the
full operating and troubleshooting procedure, and `docs/validation.md` for the
latest evidence-backed validation status.

Formal running-policy development is tracked in
`docs/run-policy-training.md`. Training uses the separate `my3d-rl` Conda
environment and writes large runs only below `/home/win98/rl_runs`. The current
measured result is stable high-speed walking plus a rejected motion-prior
running candidate, not accepted running; source research, licensing boundaries
and the next periodic-reference stage are recorded in
`docs/open-strategy-search-2026-08-31.md`,
`docs/robot-soccer-action-research.md`, and `docs/rl-experiment-log.md`.

## WSL quick start

From the repository directory inside WSL:

```bash
conda env create -f environment.yml
conda activate my3d-team
git submodule update --init --recursive
pytest -q
```

If the environment already exists, update it instead:

```bash
conda env update -n my3d-team -f environment.yml --prune
conda activate my3d-team
```

Start the official server in terminal A:

```bash
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_server.sh realtime
```

Start one seven-player team in terminal B:

```bash
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
scripts/run_team.sh My3DTeam 127.0.0.1 60000
```

For deterministic local 7v7 self-play, start the server first and then run:

```bash
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
scripts/run_selfplay.sh 127.0.0.1 60000 60001
```

`run_selfplay.sh` starts two seven-player teams, sends kickoff after all clients
have had time to join, and then drops the ball to enter deterministic `PLAY_ON`.
The deterministic kickoff is assigned to the left team so the active/passive
beam formations match the referee placement checks.

Run the complete competition acceptance gate, including a real
`APPROACH -> ALIGN -> KICK -> RECOVER` transition:

```bash
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_acceptance_match.sh 600
```

Launch a real-time WSLg viewer and a bounded 7v7 demonstration:

```bash
scripts/run_visual_match.sh 3000
```

The Apollo3D learned get-up network is enabled automatically when its pinned
submodule asset is present. Set `MY3D_GETUP_BACKEND=keyframe` to force the
built-in fallback. See `docs/apollo-integration.md` before distributing a team
package that contains the GPL-3.0-or-later Apollo asset.

## Installation

### Make sure the following are installed on your system:

- Python ≥ 3.13
 > ⚠️ The project has been tested only with Python 3.13, but it will likely work with other versions as well.


- Any Python dependency manager can be used, but **either Hatch or Poetry are recommended**.

- **Poetry ≥ 2.0.0** ([Installation Guide](https://python-poetry.org/docs/#installing-with-pipx))  
  **or**  
- **Hatch ≥ 1.9.0** ([Installation Guide](https://hatch.pypa.io/latest/install/))

### Install Dependencies
The project dependencies are listed inside pyproject.toml

Using **Hatch**:
```bash
hatch build
```

Using **Poetry**:
```bash
poetry install
```

## Instructions

### Run an agent
After installing the dependencies and setting up the environment, you can launch a player instance:

```bash
python3 run_player.py -n <player-number> -t <team-name>
```

Using **Hatch**:
```bash
hatch run python run_player.py -n <player-number> -t <team-name>
```

Using **Poetry**:
```bash
poetry run python run_player.py -n <player-number> -t <team-name>
```

CLI parameter (a usage help is also available):

- `--host <ip>` to specify the host IP (default: 'localhost')
- `--port <port>` to specify the agent port (default: 60000)
- `-n <number>` Player number (1–11) (default: 1)
- `-t <team_name>` Team name (default: 'Default')


### Run a team
You can also use a shell script to start the entire team, optionally specifying host and port:

```bash
scripts/run_team.sh [team-name] [host] [agent-port]
```

Using **Hatch**:
```bash
hatch run scripts/run_team.sh [team-name] [host] [agent-port]
```

Using **Poetry**:
```bash
poetry run scripts/run_team.sh [team-name] [host] [agent-port]
```

CLI parameter:

- `[team-name]` Team name (default: `My3DTeam`)
- `[host]` Server IP address (default: `127.0.0.1`)
- `[agent-port]` Server port for agents (default: `60000`)

An optional fourth argument limits cycles for smoke tests, for example:

```bash
scripts/run_team.sh My3DTeam 127.0.0.1 60000 800
```

### Binary building
To compete, a binary is needed. It provides a compact, portable version and protects the source code. To create a binary, just run the script ```build_binary.sh```

```bash
./build_binary.sh <team-name>
```

Using **Hatch**:
```bash
hatch run ./build_binary.sh <team-name>
```

Using **Poetry**:
```bash
poetry run ./build_binary.sh <team-name>
```

Once binary generation is finished, the result will be inside the build folder, as ```<team-name>.tar.gz```

### Brazil Open Mujoco Demo
In the Brazil Open Mujoco Demo, the adult humanoid field will be used, with 3 players in each team. The ```start3v3.sh``` script can be used for that purpose.

### Authors and acknowledgment

The match client remains influenced by the early MagmaOffenburg
RCSSServerMJ demonstrations and the FCPortugal SimSpark base. Architecture
comparisons with ApolloCodebase are documented in `docs/reference-projects.md`;
Apollo remains an isolated GPL-3.0-or-later Git submodule.

This project was developed and contributed by:
- **Alan Nascimento**
- **Luís Magalhães**
- **Pedro Rabelo**
- **Melissa Damasceno**

Contributions, bug reports, and feature requests are welcome via pull requests.
