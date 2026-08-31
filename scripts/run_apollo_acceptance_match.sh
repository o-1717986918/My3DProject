#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

max_cycles=${1:-600}
python_bin=${MY3D_PYTHON:-python3}
agent_port=${MATCH_AGENT_PORT:-$((28000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
binary="${APOLLO_BINARY:-$runtime_dir/build/ApolloCodeBase}"
asset_root="${APOLLO_ASSET_ROOT:-$runtime_dir/assets}"
if [[ -n "${MATCH_RUN_DIR:-}" ]]; then
    run_dir=$(realpath -m "$MATCH_RUN_DIR")
    mkdir -p "$run_dir"
else
    run_dir=$(mktemp -d -t my3d-apollo-match.XXXXXX)
fi
server_pid=
player_pids=()

cleanup() {
    for pid in "${player_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -z "${MATCH_RUN_DIR:-}" && "${KEEP_MATCH_LOGS:-0}" != 1 \
        && "$run_dir" == /tmp/my3d-apollo-match.* ]]; then
        rm -rf -- "$run_dir"
    fi
}
trap cleanup EXIT INT TERM

if ! [[ "$max_cycles" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-cycle-count]" >&2
    exit 2
fi

if [[ -z "${APOLLO_BINARY:-}" ]]; then
    "$repo_dir/scripts/build_apollo_runtime.sh" >/dev/null
elif [[ ! -x "$binary" ]]; then
    echo "Apollo binary is not executable: $binary" >&2
    exit 2
fi
if [[ ! -d "$asset_root" ]]; then
    echo "Apollo asset root does not exist: $asset_root" >&2
    exit 2
fi

(
    cd "$run_dir"
    render_mode=headless
    if [[ "${MATCH_RENDER:-0}" == 1 ]]; then
        render_mode=render
    fi
    exec "$repo_dir/scripts/run_server.sh" realtime "$agent_port" "$monitor_port" "$render_mode"
) >"$run_dir/server.log" 2>&1 &
server_pid=$!
sleep 2

for team in My3D-A My3D-B; do
    for number in $(seq 1 7); do
        "$binary" \
            --team "$team" \
            --player-number "$number" \
            --host 127.0.0.1 \
            --port "$agent_port" \
            --asset-root "$asset_root" \
            --max-cycles "$max_cycles" \
            --status-interval "${APOLLO_STATUS_INTERVAL:-100}" \
            >"$run_dir/${team}-${number}.log" 2>&1 &
        player_pids+=("$!")
        # RCSSServerMJ 0.2.1 can reset connections when fourteen ONNX-backed
        # clients initialize in the same scheduler slice. This still starts
        # both complete teams well inside the official three-second limit.
        sleep 0.05
    done
done

sleep 4
"$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 \
    --port "$monitor_port" \
    --delay 0.5 \
    "(kickOff Left)" \
    "(dropBall)"

clean_exits=0
for pid in "${player_pids[@]}"; do
    if wait "$pid"; then
        clean_exits=$((clean_exits + 1))
    fi
done

failures=$(
    { grep -Ehi \
        "Failed to start agent|segmentation fault|core dumped|terminate called" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
server_errors=$(grep -Eci "Traceback|ERROR|Segmentation fault|core dumped" \
    "$run_dir/server.log" 2>/dev/null || true)
connections=$(grep -c "New agent connection" "$run_dir/server.log" || true)
joins=$(grep -c "joined the game" "$run_dir/server.log" || true)
play_on=$(
    { grep -El "MY3D_STATUS.*play_on=1" "$run_dir"/My3D-*.log \
        2>/dev/null || true; } | wc -l
)
illegal_defense=$(grep -c "Illegal defense" "$run_dir/server.log" || true)
kick_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=Kick(Forward|Stabilize|Hold)" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
getup_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=GetUp" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
activation_warnings=0
if [[ -f "$run_dir/MUJOCO_LOG.TXT" ]]; then
    activation_warnings=$(grep -c "Nan, Inf or huge value in CTRL" \
        "$run_dir/MUJOCO_LOG.TXT" || true)
fi

kick_requirement_failed=0
if [[ "${MATCH_REQUIRE_KICK:-0}" == 1 && $kick_samples -eq 0 ]]; then
    kick_requirement_failed=1
fi

if [[ $clean_exits -ne 14 || $connections -ne 14 || $joins -ne 14 \
    || $play_on -ne 14 || $failures -ne 0 || $server_errors -ne 0 \
    || $illegal_defense -ne 0 || $kick_requirement_failed -ne 0 ]]; then
    echo "Apollo 7v7 acceptance failed: cycles=$max_cycles clean_exits=$clean_exits " \
        "connections=$connections joins=$joins play_on=$play_on failures=$failures " \
        "server_errors=$server_errors illegal_defense=$illegal_defense " \
        "kick_samples=$kick_samples getup_samples=$getup_samples " \
        "activation_warnings=$activation_warnings" >&2
    tail -100 "$run_dir/server.log" >&2
    if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
        echo "Logs preserved at $run_dir" >&2
    fi
    exit 1
fi

echo "Apollo 7v7 acceptance passed: cycles=$max_cycles clean_exits=$clean_exits " \
    "connections=$connections joins=$joins play_on=$play_on failures=$failures " \
    "server_errors=$server_errors illegal_defense=$illegal_defense " \
    "kick_samples=$kick_samples getup_samples=$getup_samples " \
    "activation_warnings=$activation_warnings"
if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
    echo "Logs preserved at $run_dir"
fi
