# BahiaRT Mujoco base code

This is a Python-based base code developed for the RCSSServerMJ. It was created to simplify the onboarding process for new teams joining the RoboCup 3D Soccer Simulation League using the Mujoco Simulator.

This code was influenced by the early demonstrations from MagmaOffenburg team of a client for the RCSSServerMJ, and the FCPortugal base code for the SimSpark simulator.

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
./start.sh [host] [port]
```

Using **Hatch**:
```bash
hatch run ./start.sh [host] [port]
```

Using **Poetry**:
```bash
poetry run ./start.sh [host] [port]
```

CLI parameter:

- `[host]` Server IP address (default: 'localhost')
- `[port]` Server port for agents (default: 60000)

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
This project was developed and contributed by:
- **Alan Nascimento**
- **Luís Magalhães**
- **Pedro Rabelo**
- **Melissa Damasceno**

Contributions, bug reports, and feature requests are welcome via pull requests.