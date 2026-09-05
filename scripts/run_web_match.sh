#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

max_cycles=${1:-30000}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
binary="${APOLLO_BINARY:-$runtime_dir/build/ApolloCodeBase}"
asset_root="${APOLLO_ASSET_ROOT:-$runtime_dir/assets}"
server_python=${RCSSSERVERMJ_PYTHON:-/home/win98/.local/pipx/venvs/rcsssmj/bin/python}
web_port=${MATCH_WEB_PORT:-8765}
agent_port=${MATCH_AGENT_PORT:-$((30000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
launch_stagger=${APOLLO_LAUNCH_STAGGER:-0.05}
stop_on_game_over=${MATCH_STOP_ON_GAME_OVER:-1}
game_over_hold=${MATCH_GAME_OVER_HOLD:-15}
initial_play_time=${MATCH_INITIAL_PLAY_TIME:-}
timestamp=$(date +%Y%m%d-%H%M%S)
run_dir=${MATCH_RUN_DIR:-/home/win98/rl_runs/apollo-web-match-$timestamp}
server_pid=
player_pids=()
client_strategy_args=()
server_time_args=()
parameterized_kick_mode=${APOLLO_ENABLE_PARAMETERIZED_KICK:-1}
learned_kick_mode=${APOLLO_LEARNED_KICK_MODE:-}
if [[ -z "$learned_kick_mode" ]]; then
    learned_kick_mode=$(
        [[ "$parameterized_kick_mode" == 1 ]] && echo active || echo off
    )
fi
# Visualization follows the same specialist composition as source-tree team
# runs; explicit zero values provide a stable-walk-only ablation.
fast_walk_mode=${APOLLO_ENABLE_FAST_WALK:-1}
rapid_turn_mode=${APOLLO_ENABLE_RAPID_TURN:-1}

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

if ! [[ "$max_cycles" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-cycle-count]" >&2
    exit 2
fi
if ! [[ "$web_port" =~ ^[1-9][0-9]{0,4}$ ]] || (( web_port > 65535 )); then
    echo "MATCH_WEB_PORT must be a valid TCP port" >&2
    exit 2
fi
case "$stop_on_game_over" in
    0|1) ;;
    *) echo "MATCH_STOP_ON_GAME_OVER must be 0 or 1" >&2; exit 2 ;;
esac
if ! [[ "$game_over_hold" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "MATCH_GAME_OVER_HOLD must be a non-negative number of seconds" >&2
    exit 2
fi
if [[ -n "$initial_play_time" ]]; then
    if ! [[ "$initial_play_time" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "MATCH_INITIAL_PLAY_TIME must be a non-negative number" >&2
        exit 2
    fi
    server_time_args+=(--time "$initial_play_time")
fi
if [[ ! -x "$server_python" ]]; then
    echo "RCSSServerMJ Python is missing: $server_python" >&2
    exit 2
fi
if ! "$server_python" -c "import mujoco, PIL, rcsssmj" 2>/dev/null; then
    echo "Web match dependencies are missing from the rcsssmj environment." >&2
    echo "Run: $server_python -m pip install -r $repo_dir/requirements-web-match.txt" >&2
    exit 2
fi

case "${APOLLO_ENABLE_PASS_STRATEGY:-1}" in
    1) ;;
    0) client_strategy_args+=(--disable-pass-strategy) ;;
    *) echo "APOLLO_ENABLE_PASS_STRATEGY must be 0 or 1" >&2; exit 2 ;;
esac
case "$parameterized_kick_mode" in
    1) client_strategy_args+=(--enable-parameterized-kick) ;;
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
        expected_sha=${APOLLO_LEARNED_KICK_SHA256:-b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1}
        if [[ -z "$learned_kick_model" || ! -f "$learned_kick_model" ]]; then
            echo "APOLLO_LEARNED_KICK_MODEL must name a kick_policy_v3 ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$learned_kick_model" | cut -d " " -f 1)" != "$expected_sha" ]]; then
            echo "APOLLO_LEARNED_KICK_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        if [[ "$learned_kick_mode" == active ]]; then
            client_strategy_args+=(--enable-learned-kick)
        else
            client_strategy_args+=(--shadow-learned-kick)
        fi
        client_strategy_args+=(--learned-kick-model "$learned_kick_model")
        ;;
    off) ;;
    *) echo "APOLLO_LEARNED_KICK_MODE must be off, shadow, or active" >&2; exit 2 ;;
esac
case "$fast_walk_mode" in
    1)
        fast_walk_model=${APOLLO_FAST_WALK_MODEL:-$HOME/rl_runs/stable-motion/fast-walk-transition-recovery-s20261160-v1/policy.onnx}
        expected_sha=${APOLLO_FAST_WALK_SHA256:-6214b656c28f0b95300287e5e3a26508078a6a8d036dbeda0ec5130051a190d6}
        if [[ -z "$fast_walk_model" || ! -f "$fast_walk_model" ]]; then
            echo "APOLLO_FAST_WALK_MODEL must name the phase-v2 ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$fast_walk_model" | cut -d " " -f 1)" != "$expected_sha" ]]; then
            echo "APOLLO_FAST_WALK_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        client_strategy_args+=(--enable-fast-walk --fast-walk-model "$fast_walk_model")
        ;;
    0) ;;
    *) echo "APOLLO_ENABLE_FAST_WALK must be 0 or 1" >&2; exit 2 ;;
esac
case "$rapid_turn_mode" in
    1)
        rapid_turn_model=${APOLLO_RAPID_TURN_MODEL:-$HOME/rl_runs/stable-motion/rapid-turn-s20261101-v1/policy.onnx}
        expected_sha=${APOLLO_RAPID_TURN_SHA256:-c086b819d3ffa3dbb971dbcc2bb2e40c949864a4a702546f678d89414c510cca}
        if [[ -z "$rapid_turn_model" || ! -f "$rapid_turn_model" ]]; then
            echo "APOLLO_RAPID_TURN_MODEL must name the validated run-policy ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$rapid_turn_model" | cut -d " " -f 1)" != "$expected_sha" ]]; then
            echo "APOLLO_RAPID_TURN_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        client_strategy_args+=(--enable-rapid-turn --rapid-turn-model "$rapid_turn_model")
        ;;
    0) ;;
    *) echo "APOLLO_ENABLE_RAPID_TURN must be 0 or 1" >&2; exit 2 ;;
esac

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

mkdir -p "$run_dir"
(
    cd "$run_dir"
    exec env MUJOCO_GL=egl PYTHONPATH="$repo_dir" "$server_python" \
        -m mujococodebase.web_match_app \
        --host 127.0.0.1 \
        --aport "$agent_port" \
        --mport "$monitor_port" \
        --web-host 127.0.0.1 \
        --web-port "$web_port" \
        --field fifa7vs7 \
        --rules ssim26 \
        --render-interval "${MATCH_RENDER_INTERVAL:-4}" \
        --width "${MATCH_RENDER_WIDTH:-1280}" \
        --height "${MATCH_RENDER_HEIGHT:-720}" \
        --jpeg-quality "${MATCH_JPEG_QUALITY:-82}" \
        --speed "${MATCH_INITIAL_SPEED:-1}" \
        "${server_time_args[@]}"
) >"$run_dir/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
    if curl --silent --fail "http://127.0.0.1:$web_port/health" >/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        break
    fi
    sleep 0.1
done
if [[ "$ready" != 1 ]]; then
    echo "Web match server did not become ready" >&2
    tail -100 "$run_dir/server.log" >&2 || true
    exit 1
fi

for team in My3D-A My3D-B; do
    for number in $(seq 1 7); do
        "$binary" \
            --team "$team" \
            --player-number "$number" \
            --host 127.0.0.1 \
            --port "$agent_port" \
            --asset-root "$asset_root" \
            --max-cycles "$max_cycles" \
            --status-interval "${APOLLO_STATUS_INTERVAL:-50}" \
            "${client_strategy_args[@]}" \
            >"$run_dir/${team}-${number}.log" 2>&1 &
        player_pids+=("$!")
        sleep "$launch_stagger"
    done
done

sleep 4
"$server_python" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 \
    --port "$monitor_port" \
    "(kickOff Left)"

echo "Web 7v7 ready: http://127.0.0.1:$web_port/"
echo "Logs: $run_dir"
echo "Controls: mouse drag/wheel, Tab, K/J/B, Space, 1/2/4, F, H"

game_over=0
while [[ "$stop_on_game_over" == 1 ]]; do
    live_players=0
    for pid in "${player_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            live_players=$((live_players + 1))
        fi
    done
    if [[ "$live_players" == 0 ]]; then
        break
    fi
    if curl --silent --fail "http://127.0.0.1:$web_port/api/status" \
        | grep -q '"play_mode":"GameOver"'; then
        game_over=1
        echo "GameOver detected; holding the final frame for ${game_over_hold}s"
        sleep "$game_over_hold"
        break
    fi
    sleep 0.5
done

if [[ "$game_over" == 1 ]]; then
    for pid in "${player_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
fi
for pid in "${player_pids[@]}"; do
    wait "$pid" || true
done
