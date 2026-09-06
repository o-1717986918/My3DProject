#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Pristine upstream Apollo team versus the supplied BoosterSoccer binary team.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
export MATCH_APOLLO_VARIANT=base
exec "$repo_dir/scripts/_run_web_match_booster_soccer.sh" "$@"
