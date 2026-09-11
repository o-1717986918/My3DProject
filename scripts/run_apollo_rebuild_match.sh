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
forced_goal_kick_side=${MATCH_FORCE_GOAL_KICK_SIDE:-}
force_near_ball=${MATCH_FORCE_NEAR_BALL:-0}
force_goalkeeper_shot=${MATCH_FORCE_GOALKEEPER_SHOT:-0}
near_ball_x=${MATCH_NEAR_BALL_X:-0}
near_ball_y=${MATCH_NEAR_BALL_Y:-0}
near_ball_robot_x=${MATCH_NEAR_BALL_ROBOT_X:--0.55}
near_ball_robot_y=${MATCH_NEAR_BALL_ROBOT_Y:-0}
near_ball_robot_qw=${MATCH_NEAR_BALL_ROBOT_QW:-1}
near_ball_robot_qz=${MATCH_NEAR_BALL_ROBOT_QZ:-0}
near_ball_opponent_gk_x=${MATCH_NEAR_BALL_OPPONENT_GK_X:-26}
near_ball_opponent_field_x=${MATCH_NEAR_BALL_OPPONENT_FIELD_X:-18}
goalkeeper_shot_x=${MATCH_GOALKEEPER_SHOT_X:-18}
goalkeeper_shot_y=${MATCH_GOALKEEPER_SHOT_Y:-0.7}
goalkeeper_shot_speed=${MATCH_GOALKEEPER_SHOT_SPEED:-5.0}
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
case "$forced_goal_kick_side" in ""|left|right) ;; *) echo "MATCH_FORCE_GOAL_KICK_SIDE must be left or right" >&2; exit 2 ;; esac
case "$force_near_ball" in 0|1) ;; *) echo "MATCH_FORCE_NEAR_BALL must be 0 or 1" >&2; exit 2 ;; esac
case "$force_goalkeeper_shot" in 0|1) ;; *) echo "MATCH_FORCE_GOALKEEPER_SHOT must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$force_near_ball" == 1 && "$force_goalkeeper_shot" == 1 ]]; then
    echo "MATCH_FORCE_NEAR_BALL and MATCH_FORCE_GOALKEEPER_SHOT are mutually exclusive" >&2
    exit 2
fi
if [[ "${APOLLO_REBUILD_STATUS_INTERVAL:-0}" != 0 ]]; then
    if ! [[ "$APOLLO_REBUILD_STATUS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
        echo "APOLLO_REBUILD_STATUS_INTERVAL must be a positive integer" >&2
        exit 2
    fi
    rebuild_args+=(--status-interval "$APOLLO_REBUILD_STATUS_INTERVAL")
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

all_agents_active=0
for _ in $(seq 1 100); do
    activated=$(awk '/ activated\.$/ { count += 1 } END { print count + 0 }' "$run_dir/server.log")
    if [[ "$activated" -ge 14 ]]; then
        all_agents_active=1
        break
    fi
    sleep 0.1
done
if [[ "$all_agents_active" != 1 ]]; then
    echo "agents did not all activate before kickoff; logs: $run_dir" >&2
    exit 1
fi
# Give every newly activated client one synchronized cycle to apply its beam.
sleep 1
"$server_python" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 \
    --port "$monitor_port" \
    --delay 0.1 \
    "(kickOff ${kickoff_side^})"

if [[ "$force_near_ball" == 1 ]]; then
    # kickOff enters the restart mode; it is not PlayOn. Move to PlayOn before
    # placing an attacking-third actor, otherwise the referee correctly sends
    # that player to the penalty position for crossing halfway at kickoff.
    # Stage the ball before the mode change so the first PlayOn observation is
    # at the fixed-scene location and is not rejected by Apollo's 12 m
    # anti-teleport vision guard.
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        "(ball (pos $near_ball_x $near_ball_y 0.11) (vel 0 0 0))"
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        "(dropBall)"
    # Let every client observe PlayOn before moving the actors. This also keeps
    # a final BeforeKickOff Beam from overwriting the fixed release pose.
    sleep 0.2
    # Put the left striker just behind a stationary ball and keep every other
    # player out of the lane. This is a repeatable contact-delay probe, not a
    # different decision or motion path.
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay 0.02 \
        "(agent (unum 1) (team $left_name) (move3d -26 0 0.8 1 0 0 0))" \
        "(agent (unum 2) (team $left_name) (move3d -12 -7 0.8 1 0 0 0))" \
        "(agent (unum 3) (team $left_name) (move3d -12 7 0.8 1 0 0 0))" \
        "(agent (unum 4) (team $left_name) (move3d -9 -5 0.8 1 0 0 0))" \
        "(agent (unum 5) (team $left_name) (move3d -9 5 0.8 1 0 0 0))" \
        "(agent (unum 6) (team $left_name) (move3d -6 0 0.8 1 0 0 0))" \
        "(agent (unum 7) (team $left_name) (move3d $near_ball_robot_x $near_ball_robot_y 0.8 $near_ball_robot_qw 0 0 $near_ball_robot_qz))" \
        "(agent (unum 1) (team $right_name) (move3d $near_ball_opponent_gk_x 0 0.8 0 0 0 1))" \
        "(agent (unum 2) (team $right_name) (move3d $near_ball_opponent_field_x -7 0.8 0 0 0 1))" \
        "(agent (unum 3) (team $right_name) (move3d $near_ball_opponent_field_x -5 0.8 0 0 0 1))" \
        "(agent (unum 4) (team $right_name) (move3d $near_ball_opponent_field_x -3 0.8 0 0 0 1))" \
        "(agent (unum 5) (team $right_name) (move3d $near_ball_opponent_field_x 3 0.8 0 0 0 1))" \
        "(agent (unum 6) (team $right_name) (move3d $near_ball_opponent_field_x 5 0.8 0 0 0 1))" \
        "(agent (unum 7) (team $right_name) (move3d $near_ball_opponent_field_x 7 0.8 0 0 0 1))" \
        "(ball (pos $near_ball_x $near_ball_y 0.11) (vel 0 0 0))"
fi

if [[ -n "$forced_goal_kick_side" ]]; then
    sleep 1
    if [[ "$forced_goal_kick_side" == left ]]; then
        goal_line_x=-28.0
    else
        goal_line_x=28.0
    fi
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay 0.1 \
        "(ball (pos $goal_line_x 8.0 0.11) (vel 0 0 0))"
fi

if [[ "$force_goalkeeper_shot" == 1 ]]; then
    # Mirror one canonical shot toward the rebuild team's own goal. The ball is
    # first held at the release point long enough for every agent's teleport
    # guard and velocity filter to acquire it, then receives the test velocity.
    if [[ "$rebuild_side" == left ]]; then
        shot_x=-${goalkeeper_shot_x#-}
        shot_y=${goalkeeper_shot_y#-}
        shot_vx=-${goalkeeper_shot_speed#-}
        keeper_x=-27
        keeper_qw=1
        keeper_qz=0
        left_keeper_x=$keeper_x
        left_keeper_qw=$keeper_qw
        left_keeper_qz=$keeper_qz
        right_keeper_x=20
        right_keeper_qw=0
        right_keeper_qz=1
    else
        shot_x=${goalkeeper_shot_x#-}
        shot_y=-${goalkeeper_shot_y#-}
        shot_vx=${goalkeeper_shot_speed#-}
        keeper_x=27
        keeper_qw=0
        keeper_qz=1
        left_keeper_x=-20
        left_keeper_qw=1
        left_keeper_qz=0
        right_keeper_x=$keeper_x
        right_keeper_qw=$keeper_qw
        right_keeper_qz=$keeper_qz
    fi

    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay 0.05 \
        "(ball (pos $shot_x $shot_y 0.11) (vel 0 0 0))" \
        "(dropBall)"
    sleep 0.2
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay 0.02 \
        "(agent (unum 1) (team $left_name) (move3d $left_keeper_x 0 0.8 $left_keeper_qw 0 0 $left_keeper_qz))" \
        "(agent (unum 1) (team $right_name) (move3d $right_keeper_x 0 0.8 $right_keeper_qw 0 0 $right_keeper_qz))" \
        "(agent (unum 2) (team $left_name) (move3d -5 -12 0.8 1 0 0 0))" \
        "(agent (unum 3) (team $left_name) (move3d -5 -9 0.8 1 0 0 0))" \
        "(agent (unum 4) (team $left_name) (move3d -5 -6 0.8 1 0 0 0))" \
        "(agent (unum 5) (team $left_name) (move3d -5 6 0.8 1 0 0 0))" \
        "(agent (unum 6) (team $left_name) (move3d -5 9 0.8 1 0 0 0))" \
        "(agent (unum 7) (team $left_name) (move3d -5 12 0.8 1 0 0 0))" \
        "(agent (unum 2) (team $right_name) (move3d 5 -12 0.8 0 0 0 1))" \
        "(agent (unum 3) (team $right_name) (move3d 5 -9 0.8 0 0 0 1))" \
        "(agent (unum 4) (team $right_name) (move3d 5 -6 0.8 0 0 0 1))" \
        "(agent (unum 5) (team $right_name) (move3d 5 6 0.8 0 0 0 1))" \
        "(agent (unum 6) (team $right_name) (move3d 5 9 0.8 0 0 0 1))" \
        "(agent (unum 7) (team $right_name) (move3d 5 12 0.8 0 0 0 1))" \
        "(ball (pos $shot_x $shot_y 0.11) (vel 0 0 0))"
    sleep 2.5
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        "(ball (pos $shot_x $shot_y 0.11) (vel $shot_vx 0 0))"
fi

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
