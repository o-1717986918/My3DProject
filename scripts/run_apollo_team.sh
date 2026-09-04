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
motion_args=()
parameterized_kick_mode=${APOLLO_ENABLE_PARAMETERIZED_KICK:-1}
learned_kick_mode=${APOLLO_LEARNED_KICK_MODE:-}
if [[ -z "$learned_kick_mode" ]]; then
    learned_kick_mode=$(
        [[ "$parameterized_kick_mode" == 1 ]] && echo active || echo off
    )
fi
fast_walk_mode=${APOLLO_ENABLE_FAST_WALK:-1}
if [[ -n "$max_cycles" ]]; then
    cycle_args=(--max-cycles "$max_cycles")
fi

case "$parameterized_kick_mode" in
    1) motion_args+=(--enable-parameterized-kick) ;;
    0) ;;
    *) echo "APOLLO_ENABLE_PARAMETERIZED_KICK must be 0 or 1" >&2; exit 2 ;;
esac

case "$learned_kick_mode" in
    active|shadow)
        if [[ "$parameterized_kick_mode" != 1 ]]; then
            echo "learned kick requires APOLLO_ENABLE_PARAMETERIZED_KICK=1" >&2
            exit 2
        fi
        learned_kick_model=${APOLLO_LEARNED_KICK_MODEL:-$HOME/rl_runs/kick-transition-dagger-r2-bc-s10002/policy.onnx}
        learned_kick_sha256=${APOLLO_LEARNED_KICK_SHA256:-b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1}
        if [[ -z "$learned_kick_model" || ! -f "$learned_kick_model" ]]; then
            echo "APOLLO_LEARNED_KICK_MODEL must name a kick_policy_v3 ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$learned_kick_model" | cut -d " " -f 1)" != "$learned_kick_sha256" ]]; then
            echo "APOLLO_LEARNED_KICK_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        if [[ "$learned_kick_mode" == active ]]; then
            motion_args+=(--enable-learned-kick)
        else
            motion_args+=(--shadow-learned-kick)
        fi
        motion_args+=(--learned-kick-model "$learned_kick_model")
        ;;
    off) ;;
    *) echo "APOLLO_LEARNED_KICK_MODE must be off, shadow, or active" >&2; exit 2 ;;
esac

case "$fast_walk_mode" in
    1)
        fast_walk_model=${APOLLO_FAST_WALK_MODEL:-$HOME/rl_runs/run-phase-v2-formal-s71-20260831-01/policy-best.onnx}
        fast_walk_sha256=c8a2f80b08a82a41cebaadc53c09467722a821edfc521e4a0d6921e1d481415b
        if [[ -z "$fast_walk_model" || ! -f "$fast_walk_model" ]]; then
            echo "APOLLO_FAST_WALK_MODEL must name the phase-v2 ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$fast_walk_model" | cut -d " " -f 1)" != "$fast_walk_sha256" ]]; then
            echo "APOLLO_FAST_WALK_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        motion_args+=(--enable-fast-walk --fast-walk-model "$fast_walk_model")
        ;;
    0) ;;
    *) echo "APOLLO_ENABLE_FAST_WALK must be 0 or 1" >&2; exit 2 ;;
esac

for number in $(seq 1 7); do
    "$binary" \
        --team "$team_name" \
        --player-number "$number" \
        --host "$host" \
        --port "$port" \
        --asset-root "$runtime_dir/assets" \
        "${cycle_args[@]}" \
        "${motion_args[@]}" &
    player_pids+=("$!")
    sleep 0.05
done

status=0
for pid in "${player_pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
