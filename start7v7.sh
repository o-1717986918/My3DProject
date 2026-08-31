#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec "$repo_dir/scripts/run_apollo_acceptance_match.sh" "${1:-1200}"
