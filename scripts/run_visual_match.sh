#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

max_cycles=${1:-3000}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
timestamp=$(date +%Y%m%d-%H%M%S)
run_dir="$repo_dir/artifacts/apollo-visual-match-$timestamp"

KEEP_MATCH_LOGS=1 \
MATCH_RENDER=1 \
MATCH_RUN_DIR="$run_dir" \
MATCH_REQUIRE_KICK=1 \
APOLLO_STATUS_INTERVAL="${APOLLO_STATUS_INTERVAL:-10}" \
APOLLO_LAUNCH_STAGGER="${APOLLO_LAUNCH_STAGGER:-0.25}" \
    "$repo_dir/scripts/run_apollo_acceptance_match.sh" "$max_cycles"
