#!/usr/bin/env bash
# Collect repeated, independent RCSS Walk-to-ball entries for kick training.
# Training artifacts stay in WSL; this does not change the competition launcher.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
run_dir=${MATCH_RUN_DIR:-/home/win98/rl_runs/training-transition/kick-approaches-$(date +%Y%m%d-%H%M%S)}
agent_port=${MATCH_AGENT_PORT:-$((38000 + $$ % 1000))}
monitor_port=$((agent_port + 1))
server_python=${RCSSSERVERMJ_PYTHON:-/home/win98/.local/pipx/venvs/rcsssmj/bin/python}
approach_side=${KICK_APPROACH_SIDE:-left}
scenario_dwell_s=${KICK_APPROACH_DWELL_S:-2.2}
match_pid=

cleanup() {
    if [[ -n "$match_pid" ]] && kill -0 "$match_pid" 2>/dev/null; then
        kill "$match_pid" 2>/dev/null || true
        wait "$match_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ -e "$run_dir" ]]; then
    echo "run directory already exists: $run_dir" >&2
    exit 2
fi
if [[ ! -x "$server_python" ]]; then
    echo "RCSSServerMJ Python environment is missing" >&2
    exit 2
fi
case "${KICK_APPROACH_VARIANT:-train}" in
    train|holdout|random) ;;
    *)
        echo "KICK_APPROACH_VARIANT must be train, holdout, or random" >&2
        exit 2
        ;;
esac
if [[ "$approach_side" != left && "$approach_side" != right ]]; then
    echo "KICK_APPROACH_SIDE must be left or right" >&2
    exit 2
fi
if ! [[ "$scenario_dwell_s" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    ! awk -v value="$scenario_dwell_s" 'BEGIN { exit !(value >= 1.0 && value <= 4.0) }'; then
    echo "KICK_APPROACH_DWELL_S must be in [1.0, 4.0]" >&2
    exit 2
fi
if [[ "${KICK_APPROACH_VARIANT:-train}" == random ]] &&
    [[ ! "${KICK_APPROACH_SEED:-20260921}" =~ ^[0-9]+$ ]]; then
    echo "KICK_APPROACH_SEED must be a nonnegative integer" >&2
    exit 2
fi

near_ball_robot_x=-0.8
if [[ "$approach_side" == right ]]; then
    near_ball_robot_x=0.8
fi
MATCH_RUN_DIR="$run_dir" \
MATCH_AGENT_PORT="$agent_port" \
MATCH_MONITOR_PORT="$monitor_port" \
REBUILD_SIDE="$approach_side" \
MATCH_FORCE_NEAR_BALL=1 \
MATCH_NEAR_BALL_ROBOT_X="$near_ball_robot_x" \
APOLLO_REBUILD_ENABLE_DYNAMIC_PASS=0 \
APOLLO_REBUILD_STATUS_INTERVAL=5 \
APOLLO_REBUILD_TRAINING_TELEMETRY_INTERVAL=1 \
    "$repo_dir/scripts/run_apollo_rebuild_match.sh" 45 &
match_pid=$!

for _ in $(seq 1 150); do
    if [[ -f "$run_dir/Apollo-Rebuild-7.log" ]] &&
        rg -q 'APOLLO_REBUILD_STATUS .*mode=4 ' "$run_dir/Apollo-Rebuild-7.log"; then
        break
    fi
    if ! kill -0 "$match_pid" 2>/dev/null; then
        echo "match exited before PlayOn: $run_dir" >&2
        wait "$match_pid" || true
        exit 1
    fi
    sleep 0.1
done
if [[ ! -f "$run_dir/Apollo-Rebuild-7.log" ]] ||
    ! rg -q 'APOLLO_REBUILD_STATUS .*mode=4 ' "$run_dir/Apollo-Rebuild-7.log"; then
    echo "match did not reach PlayOn: $run_dir" >&2
    exit 1
fi
sleep 0.5

# x, y, quaternion w/z, and ball x/y velocity. Keep the ball in the same
# central region so the runtime's anti-teleport filter remains meaningful.
# The held-out set deliberately uses different release poses and ball speeds.
case "${KICK_APPROACH_VARIANT:-train}" in
    train)
        scenarios=(
            '-0.75 0.00 1.000 0.000 0.00 0.00'
            '-0.90 0.15 0.996 0.087 0.00 0.00'
            '-0.90 -0.15 0.996 -0.087 0.00 0.00'
            '-1.10 0.00 1.000 0.000 -0.30 0.00'
            '-1.10 0.20 0.996 0.087 -0.30 0.05'
            '-1.10 -0.20 0.996 -0.087 -0.30 -0.05'
            '-1.35 0.00 1.000 0.000 0.20 0.00'
            '-1.35 0.25 0.996 0.087 0.20 -0.05'
            '-1.35 -0.25 0.996 -0.087 0.20 0.05'
            '-0.95 0.00 1.000 0.000 -0.50 0.00'
        )
        ;;
    holdout)
        scenarios=(
            '-0.82 0.08 0.999 0.044 0.10 0.00'
            '-1.02 -0.08 0.999 -0.044 -0.15 0.00'
            '-1.22 0.12 0.991 0.131 -0.20 0.08'
            '-1.22 -0.12 0.991 -0.131 -0.20 -0.08'
            '-0.86 0.26 0.991 0.131 0.00 -0.08'
            '-0.86 -0.26 0.991 -0.131 0.00 0.08'
            '-1.42 0.10 0.999 0.044 0.30 0.00'
            '-1.42 -0.10 0.999 -0.044 0.30 0.00'
            '-1.05 0.00 1.000 0.000 -0.40 0.00'
            '-0.78 0.00 1.000 0.000 0.15 0.00'
        )
        ;;
    random)
        # Seeded, distinct match-level approach distributions; not natural
        # match releases. Limit to ten resets so all fit the 45 s match.
        mapfile -t scenarios < <("$server_python" - "${KICK_APPROACH_SEED:-20260921}" <<'PY'
import math
import random
import sys

rng = random.Random(int(sys.argv[1]))
for _ in range(10):
    x = rng.uniform(-1.38, -0.78)
    y = rng.uniform(-0.25, 0.25)
    yaw = rng.uniform(-0.20, 0.20)
    vx = rng.uniform(-0.35, 0.20)
    vy = rng.uniform(-0.08, 0.08)
    print(f"{x:.4f} {y:.4f} {math.cos(yaw / 2):.6f} "
          f"{math.sin(yaw / 2):.6f} {vx:.4f} {vy:.4f}")
PY
        )
        ;;
esac

if [[ "$approach_side" == left ]]; then
    placement_commands=(
        '(agent (unum 2) (team Apollo-Rebuild) (move3d -20 -9 0.8 1 0 0 0))'
        '(agent (unum 3) (team Apollo-Rebuild) (move3d -20 9 0.8 1 0 0 0))'
        '(agent (unum 4) (team Apollo-Rebuild) (move3d -19 -6 0.8 1 0 0 0))'
        '(agent (unum 5) (team Apollo-Rebuild) (move3d -19 6 0.8 1 0 0 0))'
        '(agent (unum 6) (team Apollo-Rebuild) (move3d -18 0 0.8 1 0 0 0))'
        '(agent (unum 2) (team Apollo-Base) (move3d 20 -9 0.8 0 0 0 1))'
        '(agent (unum 3) (team Apollo-Base) (move3d 20 9 0.8 0 0 0 1))'
        '(agent (unum 4) (team Apollo-Base) (move3d 19 -6 0.8 0 0 0 1))'
        '(agent (unum 5) (team Apollo-Base) (move3d 19 6 0.8 0 0 0 1))'
        '(agent (unum 6) (team Apollo-Base) (move3d 18 0 0.8 0 0 0 1))'
        '(agent (unum 7) (team Apollo-Base) (move3d 18 8 0.8 0 0 0 1))'
    )
else
    # Raw server coordinates are a 180-degree rotation of the canonical
    # right-team frame used by runtime telemetry.
    placement_commands=(
        '(agent (unum 2) (team Apollo-Rebuild) (move3d 20 9 0.8 0 0 0 1))'
        '(agent (unum 3) (team Apollo-Rebuild) (move3d 20 -9 0.8 0 0 0 1))'
        '(agent (unum 4) (team Apollo-Rebuild) (move3d 19 6 0.8 0 0 0 1))'
        '(agent (unum 5) (team Apollo-Rebuild) (move3d 19 -6 0.8 0 0 0 1))'
        '(agent (unum 6) (team Apollo-Rebuild) (move3d 18 0 0.8 0 0 0 1))'
        '(agent (unum 2) (team Apollo-Base) (move3d -20 9 0.8 1 0 0 0))'
        '(agent (unum 3) (team Apollo-Base) (move3d -20 -9 0.8 1 0 0 0))'
        '(agent (unum 4) (team Apollo-Base) (move3d -19 6 0.8 1 0 0 0))'
        '(agent (unum 5) (team Apollo-Base) (move3d -19 -6 0.8 1 0 0 0))'
        '(agent (unum 6) (team Apollo-Base) (move3d -18 0 0.8 1 0 0 0))'
        '(agent (unum 7) (team Apollo-Base) (move3d -18 -8 0.8 1 0 0 0))'
    )
fi

for scenario in "${scenarios[@]}"; do
    read -r robot_x robot_y robot_qw robot_qz ball_vx ball_vy <<<"$scenario"
    if [[ "$approach_side" == right ]]; then
        read -r robot_x robot_y robot_qw robot_qz ball_vx ball_vy < <(
            "$server_python" -c \
                'import sys; x,y,qw,qz,vx,vy=map(float,sys.argv[1:]); print(-x,-y,-qz,qw,-vx,-vy)' \
                "$robot_x" "$robot_y" "$robot_qw" "$robot_qz" "$ball_vx" "$ball_vy"
        )
    fi
    "$server_python" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 --port "$monitor_port" --delay 0.02 \
        "(agent (unum 7) (team Apollo-Rebuild) (move3d $robot_x $robot_y 0.8 $robot_qw 0 0 $robot_qz))" \
        "${placement_commands[@]}" \
        "(ball (pos 0 0 0.11) (vel $ball_vx $ball_vy 0))"
    printf 'scenario side=%s raw_x=%s raw_y=%s ball_v=(%s,%s)\n' \
        "$approach_side" "$robot_x" "$robot_y" "$ball_vx" "$ball_vy"
    sleep "$scenario_dwell_s"
done

wait "$match_pid"
match_pid=
echo "Kick approach telemetry: $run_dir"
