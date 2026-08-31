#!/usr/bin/env bash
set -euo pipefail

max_cycles=${1:-500}
python_bin=${MY3D_PYTHON:-python3}
# Keep temporary listeners below the Linux/Windows ephemeral client ranges.
# RCSSServerMJ does not set SO_REUSEADDR, so a recently closed accepted socket
# can otherwise make a subsequent acceptance run fail during bind.
agent_port=${MATCH_AGENT_PORT:-$((26000 + $$ % 1000))}
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
run_bursts=$(grep -c "reference-run burst activated" "$run_dir/teams.log" || true)
run_completions=$(grep -c "reference-run burst completed" "$run_dir/teams.log" || true)
run_aborts=$(grep -Ec "reference-run burst (posture-guard|inference-error)" \
    "$run_dir/teams.log" || true)
alignments=$(grep -c "attack=ALIGN" "$run_dir/teams.log" || true)
attack_recoveries=$(grep -c "attack=RECOVER" "$run_dir/teams.log" || true)
require_attack_loop=${MATCH_REQUIRE_ATTACK_LOOP:-1}
require_run_burst=${MATCH_REQUIRE_RUN_BURST:-0}

if [[ $team_status -ne 0 || $connections -lt 14 || $play_on -lt 14 \
    || $shutdowns -lt 14 || $failures -ne 0 \
    || ( $require_attack_loop == 1 \
        && ( $alignments -lt 1 || $kicks -lt 1 || $attack_recoveries -lt 1 ) ) \
    || ( $require_run_burst == 1 \
        && ( $run_bursts -lt 1 || $run_aborts -ne 0 \
            || $((run_completions * 100)) -lt $((run_bursts * 80)) ) ) ]]; then
    echo "7v7 acceptance failed: status=$team_status connected=$connections " \
        "play_on=$play_on shutdowns=$shutdowns failures=$failures " \
        "alignments=$alignments kicks=$kicks attack_recoveries=$attack_recoveries " \
        "run_bursts=$run_bursts run_completions=$run_completions " \
        "run_aborts=$run_aborts" >&2
    tail -100 "$run_dir/teams.log" >&2
    if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
        echo "Logs preserved at $run_dir" >&2
    fi
    exit 1
fi

echo "7v7 acceptance passed: cycles=$max_cycles connected=$connections " \
    "play_on=$play_on shutdowns=$shutdowns failures=$failures " \
    "alignments=$alignments kicks=$kicks attack_recoveries=$attack_recoveries " \
    "getups=$recoveries run_bursts=$run_bursts " \
    "run_completions=$run_completions run_aborts=$run_aborts"
if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
    echo "Logs preserved at $run_dir"
fi
