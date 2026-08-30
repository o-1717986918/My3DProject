#!/usr/bin/env bash
set -euo pipefail

max_cycles=${1:-500}
python_bin=${MY3D_PYTHON:-python3}
agent_port=${MATCH_AGENT_PORT:-$((62000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
run_dir=$(mktemp -d -t my3d-match.XXXXXX)
server_pid=

cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ "${KEEP_MATCH_LOGS:-0}" != 1 && "$run_dir" == /tmp/my3d-match.* ]]; then
        rm -rf -- "$run_dir"
    fi
}
trap cleanup EXIT INT TERM

if ! [[ "$max_cycles" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-cycle-count]" >&2
    exit 2
fi

(
    cd "$run_dir"
    exec "$repo_dir/scripts/run_server.sh" realtime "$agent_port" "$monitor_port"
) >"$run_dir/server.log" 2>&1 &
server_pid=$!

sleep 2
set +e
(
    cd "$repo_dir"
    MY3D_PYTHON="$python_bin" \
    MY3D_STATUS_INTERVAL="${MY3D_STATUS_INTERVAL:-100}" \
    scripts/run_selfplay.sh \
        127.0.0.1 "$agent_port" "$monitor_port" "$max_cycles"
) >"$run_dir/teams.log" 2>&1
team_status=$?
set -e

connections=$(grep -c "Server connection established" "$run_dir/teams.log" || true)
play_on=$(grep -c "playmode=PLAY_ON" "$run_dir/teams.log" || true)
shutdowns=$(grep -c "Shutting down" "$run_dir/teams.log" || true)
failures=$(grep -Eci "Traceback|\[ERROR\]|segmentation fault|core dumped" \
    "$run_dir/teams.log" || true)
kicks=$(grep -c "attack=KICK" "$run_dir/teams.log" || true)
recoveries=$(grep -ci "get-up verified" "$run_dir/teams.log" || true)
alignments=$(grep -c "attack=ALIGN" "$run_dir/teams.log" || true)
attack_recoveries=$(grep -c "attack=RECOVER" "$run_dir/teams.log" || true)
require_attack_loop=${MATCH_REQUIRE_ATTACK_LOOP:-1}

if [[ $team_status -ne 0 || $connections -lt 14 || $play_on -lt 14 \
    || $shutdowns -lt 14 || $failures -ne 0 \
    || ( $require_attack_loop == 1 \
        && ( $alignments -lt 1 || $kicks -lt 1 || $attack_recoveries -lt 1 ) ) ]]; then
    echo "7v7 acceptance failed: status=$team_status connected=$connections " \
        "play_on=$play_on shutdowns=$shutdowns failures=$failures " \
        "alignments=$alignments kicks=$kicks attack_recoveries=$attack_recoveries" >&2
    tail -100 "$run_dir/teams.log" >&2
    if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
        echo "Logs preserved at $run_dir" >&2
    fi
    exit 1
fi

echo "7v7 acceptance passed: cycles=$max_cycles connected=$connections " \
    "play_on=$play_on shutdowns=$shutdowns failures=$failures " \
    "alignments=$alignments kicks=$kicks attack_recoveries=$attack_recoveries " \
    "getups=$recoveries"
if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
    echo "Logs preserved at $run_dir"
fi
