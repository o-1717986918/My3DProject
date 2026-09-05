#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Run the developed My3D team against the pristine upstream ApolloCodebase in
# one browser-rendered 7v7 match. This is intentionally separate from
# run_web_match.sh so the ordinary self-play launcher keeps its own contract.

set -euo pipefail

max_cycles=${1:-120000}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
workspace_dir=$(cd "$repo_dir/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
current_binary="${APOLLO_BINARY:-$runtime_dir/build/ApolloCodeBase}"
current_asset_root="${APOLLO_ASSET_ROOT:-$runtime_dir/assets}"
base_repo="${APOLLO_BASE_REPO:-$workspace_dir/ApolloCodebase-reference}"
base_revision="${APOLLO_BASE_EXPECTED_REVISION:-71018c968969d6e55130b0e1987cd5b4f5c3b4df}"
base_build_dir="${APOLLO_BASE_BUILD_DIR:-/home/win98/.cache/my3d/apollo-base-$base_revision}"
base_binary="$base_build_dir/ApolloCodeBase"
base_asset_root="$base_repo/assets"
onnxruntime_root="${APOLLO_BASE_ONNXRUNTIME_ROOT:-$runtime_dir/deploy/thirdparty/onnxruntime-linux-x64-1.22.0}"
server_python=${RCSSSERVERMJ_PYTHON:-/home/win98/.local/pipx/venvs/rcsssmj/bin/python}
web_port=${MATCH_WEB_PORT:-8765}
agent_port=${MATCH_AGENT_PORT:-$((30000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
launch_stagger=${APOLLO_LAUNCH_STAGGER:-0.05}
stop_on_game_over=${MATCH_STOP_ON_GAME_OVER:-1}
game_over_hold=${MATCH_GAME_OVER_HOLD:-15}
open_windows_browser=${MATCH_OPEN_WINDOWS_BROWSER:-1}
rebuild_current=${APOLLO_REBUILD_CURRENT:-0}
rebuild_base=${APOLLO_REBUILD_BASE:-0}
initial_play_time=${MATCH_INITIAL_PLAY_TIME:-}
current_team=${MATCH_CURRENT_TEAM_NAME:-My3D-Current}
base_team=${MATCH_BASE_TEAM_NAME:-Apollo-Base}
timestamp=$(date +%Y%m%d-%H%M%S)
run_dir=${MATCH_RUN_DIR:-/home/win98/rl_runs/apollo-vs-base-web-match-$timestamp}

server_pid=
current_pids=()
base_pids=()
current_args=()
server_time_args=()

cleanup() {
    for pid in "${current_pids[@]:-}" "${base_pids[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
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
if [[ "$current_team" == "$base_team" ]]; then
    echo "MATCH_CURRENT_TEAM_NAME and MATCH_BASE_TEAM_NAME must differ" >&2
    exit 2
fi
case "$stop_on_game_over" in
    0|1) ;;
    *) echo "MATCH_STOP_ON_GAME_OVER must be 0 or 1" >&2; exit 2 ;;
esac
case "$open_windows_browser" in
    0|1) ;;
    *) echo "MATCH_OPEN_WINDOWS_BROWSER must be 0 or 1" >&2; exit 2 ;;
esac
case "$rebuild_current:$rebuild_base" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "APOLLO_REBUILD_CURRENT and APOLLO_REBUILD_BASE must be 0 or 1" >&2; exit 2 ;;
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

if [[ ! -d "$base_repo/.git" ]]; then
    echo "Pristine ApolloCodebase checkout is missing: $base_repo" >&2
    exit 2
fi
actual_base_revision=$(git -C "$base_repo" rev-parse HEAD)
if [[ "$actual_base_revision" != "$base_revision" ]]; then
    echo "Apollo base revision mismatch: expected $base_revision, got $actual_base_revision" >&2
    exit 2
fi
if [[ -n "$(git -C "$base_repo" status --porcelain --untracked-files=normal)" ]]; then
    echo "Apollo base checkout is not pristine: $base_repo" >&2
    exit 2
fi
if [[ ! -d "$base_asset_root" ]]; then
    echo "Apollo base assets are missing: $base_asset_root" >&2
    exit 2
fi
if [[ ! -f "$onnxruntime_root/include/onnxruntime_cxx_api.h" || \
      ! -f "$onnxruntime_root/lib/libonnxruntime.so.1.22.0" ]]; then
    echo "ONNX Runtime dependency is missing: $onnxruntime_root" >&2
    exit 2
fi

# The developed side uses every currently supported source-tree capability.
case "${APOLLO_ENABLE_PASS_STRATEGY:-1}" in
    1) ;;
    0) current_args+=(--disable-pass-strategy) ;;
    *) echo "APOLLO_ENABLE_PASS_STRATEGY must be 0 or 1" >&2; exit 2 ;;
esac

case "${APOLLO_ENABLE_TEAM_TACTICS:-1}" in
    1) ;;
    0) current_args+=(--disable-team-tactics) ;;
    *) echo "APOLLO_ENABLE_TEAM_TACTICS must be 0 or 1" >&2; exit 2 ;;
esac

parameterized_kick_mode=${APOLLO_ENABLE_PARAMETERIZED_KICK:-1}
case "$parameterized_kick_mode" in
    1) current_args+=(--enable-parameterized-kick) ;;
    0) ;;
    *) echo "APOLLO_ENABLE_PARAMETERIZED_KICK must be 0 or 1" >&2; exit 2 ;;
esac

learned_kick_mode=${APOLLO_LEARNED_KICK_MODE:-}
if [[ -z "$learned_kick_mode" ]]; then
    if [[ "$parameterized_kick_mode" == 1 ]]; then
        # The retained v3 actor is useful for inference telemetry but has not
        # beaten the deterministic 2 m action on frozen execution trials.
        # Keep it loaded in shadow mode so it cannot mask the stronger action
        # bank during the developed-versus-pristine comparison.
        learned_kick_mode=shadow
    else
        learned_kick_mode=off
    fi
fi
case "$learned_kick_mode" in
    active|shadow)
        if [[ "$parameterized_kick_mode" != 1 ]]; then
            echo "learned kick requires APOLLO_ENABLE_PARAMETERIZED_KICK=1" >&2
            exit 2
        fi
        learned_kick_model=${APOLLO_LEARNED_KICK_MODEL:-$HOME/rl_runs/kick-transition-dagger-r2-bc-s10002/policy.onnx}
        learned_kick_sha=${APOLLO_LEARNED_KICK_SHA256:-b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1}
        if [[ ! -f "$learned_kick_model" ]] || \
           [[ "$(sha256sum "$learned_kick_model" | cut -d " " -f 1)" != "$learned_kick_sha" ]]; then
            echo "Learned-kick model is missing or failed its SHA-256 lock" >&2
            exit 2
        fi
        if [[ "$learned_kick_mode" == active ]]; then
            current_args+=(--enable-learned-kick)
        else
            current_args+=(--shadow-learned-kick)
        fi
        current_args+=(--learned-kick-model "$learned_kick_model")
        ;;
    off) ;;
    *) echo "APOLLO_LEARNED_KICK_MODE must be off, shadow, or active" >&2; exit 2 ;;
esac

case "${APOLLO_ENABLE_FAST_WALK:-1}" in
    1)
        fast_walk_model=${APOLLO_FAST_WALK_MODEL:-$HOME/rl_runs/stable-motion/fast-walk-transition-recovery-s20261160-v1/policy.onnx}
        fast_walk_sha=${APOLLO_FAST_WALK_SHA256:-6214b656c28f0b95300287e5e3a26508078a6a8d036dbeda0ec5130051a190d6}
        if [[ ! -f "$fast_walk_model" ]] || \
           [[ "$(sha256sum "$fast_walk_model" | cut -d " " -f 1)" != "$fast_walk_sha" ]]; then
            echo "FastWalk model is missing or failed its SHA-256 lock" >&2
            exit 2
        fi
        current_args+=(--enable-fast-walk --fast-walk-model "$fast_walk_model")
        ;;
    0) ;;
    *) echo "APOLLO_ENABLE_FAST_WALK must be 0 or 1" >&2; exit 2 ;;
esac

case "${APOLLO_ENABLE_RAPID_TURN:-1}" in
    1)
        rapid_turn_model=${APOLLO_RAPID_TURN_MODEL:-$HOME/rl_runs/stable-motion/rapid-turn-s20261101-v1/policy.onnx}
        rapid_turn_sha=${APOLLO_RAPID_TURN_SHA256:-c086b819d3ffa3dbb971dbcc2bb2e40c949864a4a702546f678d89414c510cca}
        if [[ ! -f "$rapid_turn_model" ]] || \
           [[ "$(sha256sum "$rapid_turn_model" | cut -d " " -f 1)" != "$rapid_turn_sha" ]]; then
            echo "RapidTurn model is missing or failed its SHA-256 lock" >&2
            exit 2
        fi
        current_args+=(--enable-rapid-turn --rapid-turn-model "$rapid_turn_model")
        ;;
    0) ;;
    *) echo "APOLLO_ENABLE_RAPID_TURN must be 0 or 1" >&2; exit 2 ;;
esac

if [[ "$rebuild_current" == 1 || ! -x "$current_binary" ]]; then
    "$repo_dir/scripts/build_apollo_runtime.sh" >/dev/null
fi
if [[ ! -x "$current_binary" ]]; then
    echo "Developed Apollo binary is not executable: $current_binary" >&2
    exit 2
fi
if [[ ! -d "$current_asset_root" ]]; then
    echo "Developed Apollo assets are missing: $current_asset_root" >&2
    exit 2
fi

if [[ "$rebuild_base" == 1 || ! -x "$base_binary" ]]; then
    cmake -S "$base_repo" -B "$base_build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_ROOT="$onnxruntime_root" >/dev/null
    cmake --build "$base_build_dir" \
        --parallel "${APOLLO_BUILD_JOBS:-$(nproc)}" >/dev/null
fi
if [[ ! -x "$base_binary" ]]; then
    echo "Apollo base binary was not produced: $base_binary" >&2
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
        --web-host 0.0.0.0 \
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

for number in $(seq 1 7); do
    "$current_binary" \
        --team "$current_team" \
        --player-number "$number" \
        --host 127.0.0.1 \
        --port "$agent_port" \
        --asset-root "$current_asset_root" \
        --max-cycles "$max_cycles" \
        --status-interval "${APOLLO_STATUS_INTERVAL:-50}" \
        "${current_args[@]}" \
        >"$run_dir/${current_team}-${number}.log" 2>&1 &
    current_pids+=("$!")
    sleep "$launch_stagger"
done

for number in $(seq 1 7); do
    "$base_binary" \
        --team "$base_team" \
        --player-number "$number" \
        --host 127.0.0.1 \
        --port "$agent_port" \
        --asset-root "$base_asset_root" \
        >"$run_dir/${base_team}-${number}.log" 2>&1 &
    base_pids+=("$!")
    sleep "$launch_stagger"
done

sleep 4
"$server_python" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 \
    --port "$monitor_port" \
    "(kickOff Left)"

match_url="http://127.0.0.1:$web_port/"
echo "Developed-vs-base Web 7v7 ready: $match_url"
echo "Left/current: $current_team ($(git -C "$repo_dir" rev-parse --short HEAD))"
echo "Right/pristine: $base_team (${base_revision:0:7})"
echo "Logs: $run_dir"
echo "Controls: mouse drag/wheel, Tab, K/J/B, Space, 1/2/4, F, H"

if [[ "$open_windows_browser" == 1 ]]; then
    powershell.exe -NoProfile -NonInteractive -Command \
        "Start-Process '$match_url'" >/dev/null 2>&1 || \
        echo "Could not open the Windows browser automatically; open $match_url manually" >&2
fi

game_over=0
while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Web match server exited unexpectedly; see $run_dir/server.log" >&2
        exit 1
    fi

    current_live=0
    for pid in "${current_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && current_live=$((current_live + 1))
    done
    base_live=0
    for pid in "${base_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && base_live=$((base_live + 1))
    done
    if [[ "$current_live" == 0 || "$base_live" == 0 ]]; then
        break
    fi

    if [[ "$stop_on_game_over" == 1 ]] && \
       curl --silent --fail "http://127.0.0.1:$web_port/api/status" \
           | grep -q '"play_mode":"GameOver"'; then
        game_over=1
        echo "GameOver detected; holding the final frame for ${game_over_hold}s"
        sleep "$game_over_hold"
        break
    fi
    sleep 0.5
done

if [[ "$game_over" == 0 ]]; then
    echo "One team stopped before GameOver; logs retained at $run_dir"
fi
