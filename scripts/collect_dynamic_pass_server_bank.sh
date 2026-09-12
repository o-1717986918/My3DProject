#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Collect balanced RCSSServerMJ outcomes for every frozen dynamic-pass
# prototype. This is an offline experiment driver; normal matches must not set
# APOLLO_REBUILD_FORCE_DYNAMIC_PASS_ROLLOUT.

set -euo pipefail

repetitions=${1:-1}
wall_seconds=${2:-7}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runner="$repo_dir/scripts/run_apollo_rebuild_match.sh"
python_bin=${DYNAMIC_PASS_COLLECTION_PYTHON:-/home/win98/miniconda3/envs/my3d-rl/bin/python}
base_port=${DYNAMIC_PASS_COLLECTION_BASE_PORT:-38200}
run_root=${DYNAMIC_PASS_COLLECTION_DIR:-/home/win98/rl_runs/dynamic-pass-forced-bank-$(date +%Y%m%d-%H%M%S)}
rollout_ids=(${DYNAMIC_PASS_ROLLOUT_IDS:-302 117 4 84 99 107 43 79 65 17})

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ && "$wall_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-repetitions] [positive-wall-seconds]" >&2
    exit 2
fi
if ! [[ "$base_port" =~ ^[1-9][0-9]*$ ]]; then
    echo "DYNAMIC_PASS_COLLECTION_BASE_PORT must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "collection Python is not executable: $python_bin" >&2
    exit 2
fi

mkdir -p "$run_root"
match_dirs=()
failed=0
run_index=0
for repetition in $(seq 1 "$repetitions"); do
    for rollout_id in "${rollout_ids[@]}"; do
        agent_port=$((base_port + run_index * 2))
        monitor_port=$((agent_port + 1))
        if ((monitor_port > 65535)); then
            echo "collection port range exceeds 65535" >&2
            exit 2
        fi
        match_dir="$run_root/r${rollout_id}-rep${repetition}"
        echo "collect rollout=$rollout_id repetition=$repetition logs=$match_dir"
        if env \
            MATCH_AGENT_PORT="$agent_port" \
            MATCH_MONITOR_PORT="$monitor_port" \
            MATCH_RUN_DIR="$match_dir" \
            MATCH_FORCE_NEAR_BALL=1 \
            MATCH_NEAR_BALL_ROBOT_X=-0.40 \
            MATCH_NEAR_BALL_ROBOT_Y=0 \
            APOLLO_REBUILD_FORCE_DYNAMIC_PASS_ROLLOUT="$rollout_id" \
            APOLLO_REBUILD_STATUS_INTERVAL=0 \
            bash "$runner" "$wall_seconds"; then
            match_dirs+=("$match_dir")
        else
            echo "collection failed for rollout=$rollout_id repetition=$repetition" >&2
            failed=$((failed + 1))
        fi
        run_index=$((run_index + 1))
    done
done

if [[ "${#match_dirs[@]}" -eq 0 ]]; then
    echo "no successful match runs were collected" >&2
    exit 1
fi

PYTHONPATH="$repo_dir/training" "$python_bin" \
    "$repo_dir/training/tools/analyze_dynamic_pass_outcomes.py" \
    --match-dir "${match_dirs[@]}" \
    --output-npz "$run_root/outcomes.npz" \
    | tee "$run_root/summary.json"

echo "Successful matches: ${#match_dirs[@]}"
echo "Failed matches: $failed"
echo "Corpus: $run_root/outcomes.npz"
if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
