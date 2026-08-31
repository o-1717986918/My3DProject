#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

team_name=${1:-My3DTeam}
host=${2:-127.0.0.1}
port=${3:-60000}
max_cycles=${4:-}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
binary="$runtime_dir/build/ApolloCodeBase"

if [[ ! -x "$binary" ]]; then
    "$repo_dir/scripts/build_apollo_runtime.sh"
fi

player_pids=()
cleanup() {
    for pid in "${player_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

cycle_args=()
if [[ -n "$max_cycles" ]]; then
    cycle_args=(--max-cycles "$max_cycles")
fi

for number in $(seq 1 7); do
    "$binary" \
        --team "$team_name" \
        --player-number "$number" \
        --host "$host" \
        --port "$port" \
        --asset-root "$runtime_dir/assets" \
        "${cycle_args[@]}" &
    player_pids+=("$!")
    sleep 0.05
done

status=0
for pid in "${player_pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
