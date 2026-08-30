#!/usr/bin/env bash
set -euo pipefail

team=${1:-My3DTeam}
host=${2:-127.0.0.1}
port=${3:-60000}
max_cycles=${4:-}
python_bin=${MY3D_PYTHON:-python3}
status_interval=${MY3D_STATUS_INTERVAL:-0}

cycle_args=()
if [[ -n "$max_cycles" ]]; then
    cycle_args=(--max-cycles "$max_cycles" --status-interval "$status_interval")
fi

agent_pids=()
cleanup() {
    for agent_pid in "${agent_pids[@]:-}"; do
        kill "$agent_pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for number in {1..7}; do
    "$python_bin" run_player.py \
        --host "$host" \
        --port "$port" \
        --number "$number" \
        --team "$team" \
        --field my_field \
        "${cycle_args[@]}" &
    agent_pids+=("$!")
done

wait
