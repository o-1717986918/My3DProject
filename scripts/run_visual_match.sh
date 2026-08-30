#!/usr/bin/env bash
set -euo pipefail

max_cycles=${1:-3000}
agent_port=${MATCH_AGENT_PORT:-60000}
monitor_port=${MATCH_MONITOR_PORT:-60001}
python_bin=${MY3D_PYTHON:-python3}
server_bin=${RCSSSERVERMJ_BIN:-rcssservermj}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
timestamp=$(date +%Y%m%d-%H%M%S)
run_dir="$repo_dir/artifacts/visual-match-$timestamp"
server_pid=

cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if ! [[ "$max_cycles" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-cycle-count]" >&2
    exit 2
fi

mkdir -p "$run_dir"

(
    cd "$run_dir"
    RCSSSERVERMJ_BIN="$server_bin" \
    RCSSSERVERMJ_LOGFILE="$run_dir/match.jsonl" \
        "$repo_dir/scripts/run_server.sh" \
        realtime "$agent_port" "$monitor_port" render
) >"$run_dir/server.log" 2>&1 &
server_pid=$!

sleep 3
set +e
(
    cd "$repo_dir"
    MY3D_PYTHON="$python_bin" scripts/run_selfplay.sh \
        127.0.0.1 "$agent_port" "$monitor_port" "$max_cycles"
) 2>&1 | tee "$run_dir/teams.log"
team_status=${PIPESTATUS[0]}
set -e

connections=$(grep -c "Server connection established" "$run_dir/teams.log" || true)
play_on=$(grep -c "playmode=PLAY_ON" "$run_dir/teams.log" || true)
failures=$(grep -Eci "Traceback|\[ERROR\]|segmentation fault|core dumped" \
    "$run_dir/teams.log" || true)

echo "Visual 7v7 finished: status=$team_status connected=$connections " \
    "play_on=$play_on failures=$failures logs=$run_dir"

if [[ $team_status -ne 0 || $connections -lt 14 || $play_on -lt 14 \
    || $failures -ne 0 ]]; then
    exit 1
fi
