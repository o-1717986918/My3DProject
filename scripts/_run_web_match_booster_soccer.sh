#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Shared implementation for the two public BoosterSoccer comparison launchers.
# Keep heavyweight archives, compatibility libraries, and logs on the WSL ext4
# filesystem so these matches do not consume scarce space on C:.

set -euo pipefail

max_cycles=${1:-120000}
match_duration_seconds=${MATCH_DURATION_SECONDS:-1200}
apollo_variant=${MATCH_APOLLO_VARIANT:-current}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
workspace_dir=$(cd "$repo_dir/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
server_python=${RCSSSERVERMJ_PYTHON:-/home/win98/.local/pipx/venvs/rcsssmj/bin/python}
web_port=${MATCH_WEB_PORT:-8765}
agent_port=${MATCH_AGENT_PORT:-$((31000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
launch_stagger=${MATCH_LAUNCH_STAGGER:-0.05}
stop_on_game_over=${MATCH_STOP_ON_GAME_OVER:-1}
game_over_hold=${MATCH_GAME_OVER_HOLD:-15}
open_windows_browser=${MATCH_OPEN_WINDOWS_BROWSER:-1}
apollo_side=${MATCH_APOLLO_SIDE:-left}
kickoff_side=${MATCH_KICKOFF_SIDE:-$apollo_side}
initial_play_time=${MATCH_INITIAL_PLAY_TIME:-}
timestamp=$(date +%Y%m%d-%H%M%S)

booster_archive=${BOOSTER_ARCHIVE:-/mnt/c/Users/26532/Downloads/BoosterSoccer.tar.gz}
booster_archive_sha=${BOOSTER_ARCHIVE_SHA256:-e38e2fda3dd22999f2dac5a2e0812855cda38b380b628e0b0e0ac20ddd0777a0}
booster_cache=${BOOSTER_CACHE_DIR:-/home/win98/.cache/my3d/booster-soccer-${booster_archive_sha:0:16}}
booster_root=${BOOSTER_ROOT:-$booster_cache}
booster_team=${MATCH_BOOSTER_TEAM_NAME:-BoosterSoccer}

glibc_version=2.39-0ubuntu8.8
glibc_deb_name="libc6_${glibc_version}_amd64.deb"
glibc_deb_url=${BOOSTER_GLIBC_DEB_URL:-https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/$glibc_deb_name}
glibc_deb_sha=${BOOSTER_GLIBC_DEB_SHA256:-3b8d5391b6b484a4c81fd000b6064885ad967ec3cb966bc57603f3fb3ebf0ed5}
glibc_cache=${BOOSTER_GLIBC_CACHE_DIR:-/home/win98/.cache/my3d/booster-glibc-2.39}

server_pid=
apollo_pids=()
booster_pids=()
apollo_args=()
server_time_args=()
booster_command=()

cleanup() {
    for pid in "${apollo_pids[@]:-}" "${booster_pids[@]:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

fail() {
    echo "$*" >&2
    exit 2
}

version_at_least() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

validate_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer"
}

validate_switch() {
    case "$2" in
        0|1) ;;
        *) fail "$1 must be 0 or 1" ;;
    esac
}

prepare_booster_archive() {
    if [[ -n "${BOOSTER_ROOT:-}" ]]; then
        return
    fi
    [[ -f "$booster_archive" ]] || fail "BoosterSoccer archive is missing: $booster_archive"
    local actual_sha
    actual_sha=$(sha256sum "$booster_archive" | cut -d ' ' -f 1)
    [[ "$actual_sha" == "$booster_archive_sha" ]] || \
        fail "BoosterSoccer archive SHA-256 mismatch: expected $booster_archive_sha, got $actual_sha"
    if [[ -x "$booster_cache/BoosterSoccer" ]]; then
        return
    fi

    local entry
    while IFS= read -r entry; do
        [[ -n "$entry" ]] || continue
        [[ "$entry" != /* ]] || fail "Unsafe absolute path in BoosterSoccer archive: $entry"
        case "/$entry/" in
            */../*) fail "Unsafe parent traversal in BoosterSoccer archive: $entry" ;;
        esac
    done < <(tar -tzf "$booster_archive")

    install -d -m 0755 "$(dirname "$booster_cache")"
    local extract_dir
    extract_dir=$(mktemp -d "$(dirname "$booster_cache")/.booster-extract.XXXXXX")
    tar -xzf "$booster_archive" --strip-components=1 -C "$extract_dir"
    [[ -x "$extract_dir/BoosterSoccer" ]] || \
        fail "BoosterSoccer binary is missing from the archive"
    [[ -d "$extract_dir/assets" && -d "$extract_dir/libs" ]] || \
        fail "BoosterSoccer assets or libraries are missing from the archive"
    if [[ -e "$booster_cache" ]]; then
        find "$extract_dir" -depth -delete
    else
        mv "$extract_dir" "$booster_cache"
    fi
}

prepare_private_glibc() {
    local host_glibc
    host_glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
    if version_at_least "$host_glibc" 2.38; then
        booster_command=(env "LD_LIBRARY_PATH=$booster_root/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$booster_root/BoosterSoccer")
        return
    fi

    local glibc_root="$glibc_cache/root"
    local glibc_lib="$glibc_root/usr/lib/x86_64-linux-gnu"
    local loader="$glibc_lib/ld-linux-x86-64.so.2"
    local deb="$glibc_cache/$glibc_deb_name"
    if [[ ! -x "$loader" ]]; then
        command -v curl >/dev/null || fail "curl is required to prepare the private BoosterSoccer GLIBC runtime"
        command -v dpkg-deb >/dev/null || fail "dpkg-deb is required to prepare the private BoosterSoccer GLIBC runtime"
        install -d -m 0755 "$glibc_cache" "$glibc_root"
        if [[ ! -f "$deb" ]]; then
            local partial="$deb.partial.$$"
            curl -fL --retry 3 "$glibc_deb_url" -o "$partial"
            local downloaded_sha
            downloaded_sha=$(sha256sum "$partial" | cut -d ' ' -f 1)
            if [[ "$downloaded_sha" != "$glibc_deb_sha" ]]; then
                rm -f "$partial"
                fail "Downloaded private GLIBC failed its SHA-256 lock"
            fi
            mv "$partial" "$deb"
        fi
        [[ "$(sha256sum "$deb" | cut -d ' ' -f 1)" == "$glibc_deb_sha" ]] || \
            fail "Cached private GLIBC package failed its SHA-256 lock"
        dpkg-deb -x "$deb" "$glibc_root"
    fi
    [[ -x "$loader" && -f "$glibc_lib/libc.so.6" ]] || \
        fail "Private GLIBC runtime was not produced under $glibc_root"
    booster_command=(
        "$loader"
        --library-path "$glibc_lib:$booster_root/libs:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
        "$booster_root/BoosterSoccer"
    )
}

configure_current_apollo() {
    apollo_team=${MATCH_APOLLO_TEAM_NAME:-My3D-Current}
    apollo_label="developed"
    apollo_binary=${APOLLO_BINARY:-$runtime_dir/build/ApolloCodeBase}
    apollo_asset_root=${APOLLO_ASSET_ROOT:-$runtime_dir/assets}
    if [[ "${APOLLO_REBUILD_CURRENT:-0}" == 1 || ! -x "$apollo_binary" ]]; then
        "$repo_dir/scripts/build_apollo_runtime.sh" >/dev/null
    fi
    [[ -x "$apollo_binary" ]] || fail "Developed Apollo binary is not executable: $apollo_binary"
    [[ -d "$apollo_asset_root" ]] || fail "Developed Apollo assets are missing: $apollo_asset_root"

    case "${APOLLO_ENABLE_PASS_STRATEGY:-1}" in
        1) ;;
        0) apollo_args+=(--disable-pass-strategy) ;;
        *) fail "APOLLO_ENABLE_PASS_STRATEGY must be 0 or 1" ;;
    esac
    case "${APOLLO_ENABLE_TEAM_TACTICS:-0}" in
        1) apollo_args+=(--enable-team-tactics) ;;
        0) apollo_args+=(--disable-team-tactics) ;;
        *) fail "APOLLO_ENABLE_TEAM_TACTICS must be 0 or 1" ;;
    esac

    local parameterized=${APOLLO_ENABLE_PARAMETERIZED_KICK:-1}
    case "$parameterized" in
        1) apollo_args+=(--enable-parameterized-kick) ;;
        0) ;;
        *) fail "APOLLO_ENABLE_PARAMETERIZED_KICK must be 0 or 1" ;;
    esac

    local learned_mode=${APOLLO_LEARNED_KICK_MODE:-}
    if [[ -z "$learned_mode" ]]; then
        [[ "$parameterized" == 1 ]] && learned_mode=shadow || learned_mode=off
    fi
    case "$learned_mode" in
        active|shadow)
            [[ "$parameterized" == 1 ]] || fail "learned kick requires parameterized kick"
            local learned_model=${APOLLO_LEARNED_KICK_MODEL:-$HOME/rl_runs/kick-transition-dagger-r2-bc-s10002/policy.onnx}
            local learned_sha=${APOLLO_LEARNED_KICK_SHA256:-b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1}
            [[ -f "$learned_model" ]] || fail "Learned-kick model is missing: $learned_model"
            [[ "$(sha256sum "$learned_model" | cut -d ' ' -f 1)" == "$learned_sha" ]] || \
                fail "Learned-kick model failed its SHA-256 lock"
            [[ "$learned_mode" == active ]] && \
                apollo_args+=(--enable-learned-kick) || \
                apollo_args+=(--shadow-learned-kick)
            apollo_args+=(--learned-kick-model "$learned_model")
            ;;
        off) ;;
        *) fail "APOLLO_LEARNED_KICK_MODE must be off, shadow, or active" ;;
    esac

    case "${APOLLO_ENABLE_FAST_WALK:-1}" in
        1)
            local fast_model=${APOLLO_FAST_WALK_MODEL:-$HOME/rl_runs/stable-motion/fast-walk-transition-recovery-s20261160-v1/policy.onnx}
            local fast_sha=${APOLLO_FAST_WALK_SHA256:-6214b656c28f0b95300287e5e3a26508078a6a8d036dbeda0ec5130051a190d6}
            [[ -f "$fast_model" ]] || fail "FastWalk model is missing: $fast_model"
            [[ "$(sha256sum "$fast_model" | cut -d ' ' -f 1)" == "$fast_sha" ]] || \
                fail "FastWalk model failed its SHA-256 lock"
            apollo_args+=(--enable-fast-walk --fast-walk-model "$fast_model")
            ;;
        0) ;;
        *) fail "APOLLO_ENABLE_FAST_WALK must be 0 or 1" ;;
    esac

    case "${APOLLO_ENABLE_RAPID_TURN:-1}" in
        1)
            local turn_model=${APOLLO_RAPID_TURN_MODEL:-$HOME/rl_runs/stable-motion/rapid-turn-s20261101-v1/policy.onnx}
            local turn_sha=${APOLLO_RAPID_TURN_SHA256:-c086b819d3ffa3dbb971dbcc2bb2e40c949864a4a702546f678d89414c510cca}
            [[ -f "$turn_model" ]] || fail "RapidTurn model is missing: $turn_model"
            [[ "$(sha256sum "$turn_model" | cut -d ' ' -f 1)" == "$turn_sha" ]] || \
                fail "RapidTurn model failed its SHA-256 lock"
            apollo_args+=(--enable-rapid-turn --rapid-turn-model "$turn_model")
            ;;
        0) ;;
        *) fail "APOLLO_ENABLE_RAPID_TURN must be 0 or 1" ;;
    esac
}

configure_base_apollo() {
    local base_revision=${APOLLO_BASE_EXPECTED_REVISION:-71018c968969d6e55130b0e1987cd5b4f5c3b4df}
    local base_repo=${APOLLO_BASE_REPO:-$workspace_dir/ApolloCodebase-reference}
    local base_build=${APOLLO_BASE_BUILD_DIR:-/home/win98/.cache/my3d/apollo-base-$base_revision}
    local onnxruntime=${APOLLO_BASE_ONNXRUNTIME_ROOT:-$runtime_dir/deploy/thirdparty/onnxruntime-linux-x64-1.22.0}
    [[ -d "$base_repo/.git" ]] || fail "Pristine Apollo checkout is missing: $base_repo"
    [[ "$(git -C "$base_repo" rev-parse HEAD)" == "$base_revision" ]] || \
        fail "Pristine Apollo revision does not match $base_revision"
    [[ -z "$(git -C "$base_repo" status --porcelain --untracked-files=normal)" ]] || \
        fail "Pristine Apollo checkout is dirty: $base_repo"
    [[ -f "$onnxruntime/include/onnxruntime_cxx_api.h" ]] || \
        fail "Apollo base ONNX Runtime dependency is missing: $onnxruntime"
    if [[ "${APOLLO_REBUILD_BASE:-0}" == 1 || ! -x "$base_build/ApolloCodeBase" ]]; then
        cmake -S "$base_repo" -B "$base_build" \
            -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$onnxruntime" >/dev/null
        cmake --build "$base_build" --parallel "${APOLLO_BUILD_JOBS:-$(nproc)}" >/dev/null
    fi
    apollo_team=${MATCH_APOLLO_TEAM_NAME:-Apollo-Base}
    apollo_label="pristine Apollo ${base_revision:0:7}"
    apollo_binary="$base_build/ApolloCodeBase"
    apollo_asset_root="$base_repo/assets"
    [[ -x "$apollo_binary" ]] || fail "Apollo base binary is missing: $apollo_binary"
    [[ -d "$apollo_asset_root" ]] || fail "Apollo base assets are missing: $apollo_asset_root"
}

validate_positive_integer max_cycles "$max_cycles"
validate_positive_integer MATCH_DURATION_SECONDS "$match_duration_seconds"
validate_switch MATCH_STOP_ON_GAME_OVER "$stop_on_game_over"
validate_switch MATCH_OPEN_WINDOWS_BROWSER "$open_windows_browser"
if ! [[ "$web_port" =~ ^[1-9][0-9]{0,4}$ ]] || (( web_port > 65535 )); then
    fail "MATCH_WEB_PORT must be a valid TCP port"
fi
[[ "$game_over_hold" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "MATCH_GAME_OVER_HOLD must be a non-negative number"
case "$apollo_variant" in
    current) configure_current_apollo ;;
    base) configure_base_apollo ;;
    *) fail "MATCH_APOLLO_VARIANT must be current or base" ;;
esac
case "$apollo_side" in left|right) ;; *) fail "MATCH_APOLLO_SIDE must be left or right" ;; esac
case "$kickoff_side" in left|right) ;; *) fail "MATCH_KICKOFF_SIDE must be left or right" ;; esac
[[ "$apollo_team" != "$booster_team" ]] || fail "Apollo and Booster team names must differ"
[[ -x "$server_python" ]] || fail "RCSSServerMJ Python is missing: $server_python"
"$server_python" -c "import mujoco, PIL, rcsssmj" 2>/dev/null || \
    fail "Web-match dependencies are missing from the RCSSServerMJ environment"
if [[ -n "$initial_play_time" ]]; then
    [[ "$initial_play_time" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
        fail "MATCH_INITIAL_PLAY_TIME must be non-negative"
    server_time_args+=(--time "$initial_play_time")
fi

prepare_booster_archive
[[ -x "$booster_root/BoosterSoccer" ]] || fail "BoosterSoccer binary is missing: $booster_root/BoosterSoccer"
[[ -f "$booster_root/assets/networks/walk/policy.onnx" ]] || fail "BoosterSoccer walk model is missing"
[[ -f "$booster_root/assets/networks/getup/policy.onnx" ]] || fail "BoosterSoccer get-up model is missing"
[[ -f "$booster_root/libs/libonnxruntime.so.1" ]] || fail "BoosterSoccer ONNX Runtime library is missing"
prepare_private_glibc

run_dir=${MATCH_RUN_DIR:-/home/win98/rl_runs/${apollo_variant}-vs-booster-web-match-$timestamp}
install -d -m 0755 "$run_dir"
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
        --match-duration-seconds "$match_duration_seconds" \
        --render-interval "${MATCH_RENDER_INTERVAL:-4}" \
        --width "${MATCH_RENDER_WIDTH:-1600}" \
        --height "${MATCH_RENDER_HEIGHT:-900}" \
        --jpeg-quality "${MATCH_JPEG_QUALITY:-82}" \
        --speed "${MATCH_INITIAL_SPEED:-4}" \
        "${server_time_args[@]}"
) >"$run_dir/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
    if curl --silent --fail "http://127.0.0.1:$web_port/health" >/dev/null; then
        ready=1
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 0.1
done
if [[ "$ready" != 1 ]]; then
    echo "Web match server did not become ready" >&2
    tail -100 "$run_dir/server.log" >&2 || true
    exit 1
fi

launch_apollo_team() {
    local number
    for number in $(seq 1 7); do
        if [[ "$apollo_variant" == current ]]; then
            "$apollo_binary" \
                --team "$apollo_team" --player-number "$number" \
                --host 127.0.0.1 --port "$agent_port" \
                --asset-root "$apollo_asset_root" \
                --max-cycles "$max_cycles" \
                --status-interval "${APOLLO_STATUS_INTERVAL:-50}" \
                "${apollo_args[@]}" \
                >"$run_dir/${apollo_team}-${number}.log" 2>&1 &
        else
            "$apollo_binary" \
                --team "$apollo_team" --player-number "$number" \
                --host 127.0.0.1 --port "$agent_port" \
                --asset-root "$apollo_asset_root" \
                >"$run_dir/${apollo_team}-${number}.log" 2>&1 &
        fi
        apollo_pids+=("$!")
        sleep "$launch_stagger"
    done
}

launch_booster_team() {
    local number
    for number in $(seq 1 7); do
        "${booster_command[@]}" \
            --team "$booster_team" --player-number "$number" \
            --host 127.0.0.1 --port "$agent_port" \
            --asset-root "$booster_root/assets" \
            >"$run_dir/${booster_team}-${number}.log" 2>&1 &
        booster_pids+=("$!")
        sleep "$launch_stagger"
    done
}

if [[ "$apollo_side" == left ]]; then
    launch_apollo_team
    launch_booster_team
else
    launch_booster_team
    launch_apollo_team
fi

sleep 4
for pid in "${apollo_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "An Apollo player exited during startup" >&2
        tail -80 "$run_dir"/"$apollo_team"-*.log >&2 || true
        exit 1
    fi
done
for pid in "${booster_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "A BoosterSoccer player exited during startup" >&2
        tail -80 "$run_dir"/"$booster_team"-*.log >&2 || true
        exit 1
    fi
done

if [[ "$kickoff_side" == left ]]; then monitor_kickoff=Left; else monitor_kickoff=Right; fi
"$server_python" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 --port "$monitor_port" "(kickOff $monitor_kickoff)"

match_url="http://127.0.0.1:$web_port/"
echo "Apollo-vs-BoosterSoccer Web 7v7 ready: $match_url"
if [[ "$apollo_side" == left ]]; then
    echo "Left/$apollo_label: $apollo_team"
    echo "Right/BoosterSoccer: $booster_team"
else
    echo "Left/BoosterSoccer: $booster_team"
    echo "Right/$apollo_label: $apollo_team"
fi
echo "Booster archive SHA-256: $booster_archive_sha"
echo "Logs: $run_dir"
echo "Match duration: ${match_duration_seconds}s of simulation time"
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
    apollo_live=0
    for pid in "${apollo_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && apollo_live=$((apollo_live + 1))
    done
    booster_live=0
    for pid in "${booster_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && booster_live=$((booster_live + 1))
    done
    if [[ "$apollo_live" == 0 || "$booster_live" == 0 ]]; then
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
