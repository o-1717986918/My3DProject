#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Minimal headless A/B match: the new Apollo rebuild versus the frozen upstream
# Apollo. Use REBUILD_SIDE=right for the side-swapped run.

set -euo pipefail

wall_seconds=${1:-20}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
workspace_dir=$(cd "$repo_dir/.." && pwd -P)
rebuild_source="$repo_dir/runtime/apollo_rebuild"
rebuild_build=${APOLLO_REBUILD_BUILD_DIR:-/home/win98/.cache/my3d/apollo-rebuild-build}
rebuild_binary="$rebuild_build/ApolloCodeBase"
base_source=${APOLLO_BASE_REPO:-$workspace_dir/ApolloCodebase-reference}
base_revision=71018c968969d6e55130b0e1987cd5b4f5c3b4df
base_build=${APOLLO_BASE_BUILD_DIR:-/home/win98/.cache/my3d/apollo-base-$base_revision}
base_binary="$base_build/ApolloCodeBase"
onnxruntime_root=${APOLLO_REBUILD_ONNXRUNTIME_ROOT:-$repo_dir/runtime/apollo/deploy/thirdparty/onnxruntime-linux-x64-1.22.0}
server_python=${RCSSSERVERMJ_PYTHON:-/home/win98/.local/pipx/venvs/rcsssmj/bin/python}
server_binary=${RCSSSERVERMJ_BIN:-/home/win98/.local/bin/rcssservermj}
agent_port=${MATCH_AGENT_PORT:-$((34000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
rebuild_side=${REBUILD_SIDE:-left}
kickoff_side=${MATCH_KICKOFF_SIDE:-left}
run_dir=${MATCH_RUN_DIR:-/home/win98/rl_runs/apollo-rebuild-vs-base-$(date +%Y%m%d-%H%M%S)-$rebuild_side}

server_pid=
player_pids=()
rebuild_args=()
cleanup() {
    for pid in "${player_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if ! [[ "$wall_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-wall-seconds]" >&2
    exit 2
fi
case "$rebuild_side" in left|right) ;; *) echo "REBUILD_SIDE must be left or right" >&2; exit 2 ;; esac
case "$kickoff_side" in left|right) ;; *) echo "MATCH_KICKOFF_SIDE must be left or right" >&2; exit 2 ;; esac
if [[ "${APOLLO_REBUILD_STATUS_INTERVAL:-0}" != 0 ]]; then
    if ! [[ "$APOLLO_REBUILD_STATUS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
        echo "APOLLO_REBUILD_STATUS_INTERVAL must be a positive integer" >&2
        exit 2
    fi
    rebuild_args=(--status-interval "$APOLLO_REBUILD_STATUS_INTERVAL")
fi

if [[ ! -x "$server_python" || ! -x "$server_binary" ]]; then
    echo "RCSSServerMJ environment is missing" >&2
    exit 2
fi
if [[ ! -x "$rebuild_binary" ]]; then
    "$repo_dir/scripts/build_apollo_rebuild.sh" >/dev/null
fi
if [[ ! -d "$base_source/.git" || "$(git -C "$base_source" rev-parse HEAD)" != "$base_revision" ]]; then
    echo "Frozen Apollo checkout is missing or at the wrong revision: $base_source" >&2
    exit 2
fi
if [[ -n "$(git -C "$base_source" status --porcelain --untracked-files=normal)" ]]; then
    echo "Frozen Apollo checkout is dirty: $base_source" >&2
    exit 2
fi
if [[ ! -x "$base_binary" ]]; then
    cmake -S "$base_source" -B "$base_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_ROOT="$onnxruntime_root" >/dev/null
    cmake --build "$base_build" --parallel "${APOLLO_BUILD_JOBS:-2}" >/dev/null
fi

mkdir -p "$run_dir"
"$server_binary" \
    --host 127.0.0.1 \
    --aport "$agent_port" \
    --mport "$monitor_port" \
    --sync --no-realtime --no-render \
    --field fifa7vs7 --rules ssim26 \
    >"$run_dir/server.log" 2>&1 &
server_pid=$!
sleep 2

if [[ "$rebuild_side" == left ]]; then
    left_name=Apollo-Rebuild
    left_binary=$rebuild_binary
    left_assets=$rebuild_source/assets
    right_name=Apollo-Base
    right_binary=$base_binary
    right_assets=$base_source/assets
else
    left_name=Apollo-Base
    left_binary=$base_binary
    left_assets=$base_source/assets
    right_name=Apollo-Rebuild
    right_binary=$rebuild_binary
    right_assets=$rebuild_source/assets
fi

for side in left right; do
    if [[ "$side" == left ]]; then
        team_name=$left_name
        binary=$left_binary
        asset_root=$left_assets
    else
        team_name=$right_name
        binary=$right_binary
        asset_root=$right_assets
    fi
    team_args=()
    if [[ "$team_name" == Apollo-Rebuild ]]; then
        team_args=("${rebuild_args[@]}")
    fi
    for number in $(seq 1 7); do
        OMP_NUM_THREADS=1 "$binary" \
            --team "$team_name" \
            --player-number "$number" \
            --host 127.0.0.1 \
            --port "$agent_port" \
            --asset-root "$asset_root" \
            "${team_args[@]}" \
            >"$run_dir/$team_name-$number.log" 2>&1 &
        player_pids+=("$!")
        sleep 0.05
    done
done

sleep 4
"$server_python" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 \
    --port "$monitor_port" \
    --delay 0.1 \
    "(kickOff ${kickoff_side^})"

sleep "$wall_seconds"

alive=0
for pid in "${player_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
    fi
done
if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited early; logs: $run_dir" >&2
    exit 1
fi
if [[ "$alive" -ne 14 ]]; then
    echo "only $alive/14 agents remain alive; logs: $run_dir" >&2
    exit 1
fi

echo "Apollo rebuild side: $rebuild_side"
echo "Agents alive: $alive/14"
echo "Logs: $run_dir"
