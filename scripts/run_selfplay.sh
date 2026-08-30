#!/usr/bin/env bash
set -euo pipefail

host=${1:-127.0.0.1}
agent_port=${2:-60000}
monitor_port=${3:-60001}
max_cycles=${4:-}
python_bin=${MY3D_PYTHON:-python3}

team_pids=()
cleanup() {
    for team_pid in "${team_pids[@]:-}"; do
        kill -- -"$team_pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

setsid scripts/run_team.sh My3D-A "$host" "$agent_port" "$max_cycles" &
team_pids+=("$!")
setsid scripts/run_team.sh My3D-B "$host" "$agent_port" "$max_cycles" &
team_pids+=("$!")

sleep 4
"$python_bin" scripts/send_monitor_command.py \
    --host "$host" \
    --port "$monitor_port" \
    --delay 0.5 \
    "(kickOff Left)" \
    "(dropBall)"

wait
