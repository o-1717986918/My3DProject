#!/usr/bin/env bash
set -euo pipefail

pose=${1:-front}
duration=${2:-9}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
python_bin=${MY3D_PYTHON:-python3}
server_bin=${RCSSSERVERMJ_BIN:-rcssservermj}
apollo_bin=${APOLLO_BIN:-$repo_dir/external/ApolloCodebase/build/ApolloCodeBase}
agent_port=${APOLLO_PROBE_AGENT_PORT:-$((62000 + $$ % 1000))}
monitor_port=${APOLLO_PROBE_MONITOR_PORT:-$((agent_port + 1))}
run_dir=$(mktemp -d -t apollo-recovery.XXXXXX)
server_pid=
agent_pid=

cleanup() {
    if [[ -n "$agent_pid" ]]; then
        kill "$agent_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

case "$pose" in
    front) quaternion="0.70710678 0 0.70710678 0" ;;
    back) quaternion="0.70710678 0 -0.70710678 0" ;;
    left) quaternion="0.70710678 -0.70710678 0 0" ;;
    right) quaternion="0.70710678 0.70710678 0 0" ;;
    *)
        echo "usage: $0 [front|back|left|right] [duration-seconds]" >&2
        exit 2
        ;;
esac

(
    cd "$run_dir"
    exec "$server_bin" \
        --host 127.0.0.1 \
        --aport "$agent_port" \
        --mport "$monitor_port" \
        --sync \
        --realtime \
        --no-render \
        --field fifa7vs7 \
        --rules ssim26 \
        --logfile "$run_dir/scene.log"
) >"$run_dir/server.log" 2>&1 &
server_pid=$!

sleep 2
(
    cd "$repo_dir"
    exec "$apollo_bin" \
        --team ApolloRecovery \
        --player-number 7 \
        --host 127.0.0.1 \
        --port "$agent_port" \
        --asset-root "$repo_dir/external/ApolloCodebase/assets"
) >"$run_dir/agent.log" 2>&1 &
agent_pid=$!

sleep 3
"$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
    --port "$monitor_port" "(dropBall)"
sleep 0.5
"$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
    --port "$monitor_port" \
    "(agent (unum 7) (team ApolloRecovery) (move3d -3 0 0.35 $quaternion))"

sleep "$duration"
kill "$agent_pid" 2>/dev/null || true
agent_pid=
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=

echo "Apollo recovery probe artifacts: $run_dir"
wc -c "$run_dir"/*
