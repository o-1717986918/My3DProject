#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Browser-rendered 7v7: baseline-first Apollo rebuild versus pristine Apollo.
# Kept separate from the legacy developed-team launcher so both remain usable.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

"$repo_dir/scripts/build_apollo_rebuild.sh" >/dev/null

export APOLLO_BINARY="$repo_dir/scripts/_run_apollo_rebuild_web_agent.sh"
export APOLLO_ASSET_ROOT="$repo_dir/runtime/apollo_rebuild/assets"
export APOLLO_REBUILD_BINARY="${APOLLO_REBUILD_BINARY:-/home/win98/.cache/my3d/apollo-rebuild-build/ApolloCodeBase}"
export APOLLO_ENABLE_PASS_STRATEGY=1
export APOLLO_ENABLE_TEAM_TACTICS=0
export APOLLO_ENABLE_PARAMETERIZED_KICK=0
export APOLLO_LEARNED_KICK_MODE=off
export APOLLO_ENABLE_FAST_WALK=0
export APOLLO_ENABLE_RAPID_TURN=0
export MATCH_CURRENT_TEAM_NAME="${MATCH_CURRENT_TEAM_NAME:-My3D-Rebuild}"
export MATCH_BASE_TEAM_NAME="${MATCH_BASE_TEAM_NAME:-Apollo-Base}"

exec "$repo_dir/scripts/run_web_match_vs_apollo_base.sh" "$@"
