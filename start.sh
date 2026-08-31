#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
team_name=${MY3D_TEAM_NAME:-My3DTeam}

exec "$repo_dir/scripts/run_apollo_team.sh" \
    "$team_name" "${1:-127.0.0.1}" "${2:-60000}"
