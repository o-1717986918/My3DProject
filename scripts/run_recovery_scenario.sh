#!/usr/bin/env bash
set -euo pipefail

pose=${1:-front}
max_cycles=${2:-700}
python_bin=${MY3D_PYTHON:-python3}
server_bin=${RCSSSERVERMJ_BIN:-rcssservermj}
agent_port=${RECOVERY_AGENT_PORT:-$((61000 + $$ % 1000))}
monitor_port=${RECOVERY_MONITOR_PORT:-$((agent_port + 1))}
drop_height=${RECOVERY_DROP_HEIGHT:-}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
run_dir=$(mktemp -d -t my3d-recovery.XXXXXX)
server_pid=
player_pid=

cleanup() {
    if [[ -n "$player_pid" ]]; then
        kill "$player_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ "$run_dir" == /tmp/my3d-recovery.* ]]; then
        rm -rf -- "$run_dir"
    fi
}
trap cleanup EXIT INT TERM

case "$pose" in
    front) quaternion="0.70710678 0 0.70710678 0" ;;
    back) quaternion="0.70710678 0 -0.70710678 0" ;;
    left) quaternion="0.70710678 -0.70710678 0 0" ;;
    right) quaternion="0.70710678 0.70710678 0 0" ;;
    *)
        echo "usage: $0 [front|back|left|right]" >&2
        exit 2
        ;;
esac

# The left-side geometry can bounce back upright when spawned at 0.35 m,
# bypassing recovery entirely. A slightly lower placement makes contact before
# gravity can self-right the robot and keeps this acceptance test deterministic.
if [[ -z "$drop_height" ]]; then
    if [[ "$pose" == left ]]; then
        drop_height=0.25
    else
        drop_height=0.35
    fi
fi

(
    cd "$run_dir"
    exec "$server_bin" \
        --host 127.0.0.1 \
        --aport "$agent_port" \
        --mport "$monitor_port" \
        --sync \
        --realtime \
        --no-render \
        --field fifa7vs7 \
        --rules ssim26
) >"$run_dir/server.log" 2>&1 &
server_pid=$!

sleep 2
(
    cd "$repo_dir"
    exec "$python_bin" run_player.py \
        --team RecoveryTest \
        --number 7 \
        --port "$agent_port" \
        --field my_field \
        --max-cycles "$max_cycles" \
        --status-interval 20
) >"$run_dir/player.log" 2>&1 &
player_pid=$!

sleep 3
"$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
    --port "$monitor_port" "(dropBall)"
sleep 0.5
"$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
    --port "$monitor_port" \
    "(agent (unum 7) (team RecoveryTest) (move3d -3 0 $drop_height $quaternion))"

set +e
wait "$player_pid"
player_status=$?
set -e
player_pid=

if [[ $player_status -ne 0 ]] || ! grep -qi "get-up verified" "$run_dir/player.log"; then
    echo "Recovery scenario failed: $pose" >&2
    tail -80 "$run_dir/player.log" >&2
    exit 1
fi

grep -Ei "starting get-up|get-up verified|cycle=.*skill=GetUp" "$run_dir/player.log"
echo "Recovery scenario passed: $pose"
