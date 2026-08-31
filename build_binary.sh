#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "usage: $0 <team-name>" >&2
    exit 2
fi

team_name=$1
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

"$repo_dir/scripts/build_apollo_runtime.sh"
"$repo_dir/runtime/apollo/pack.sh" "$repo_dir/build" "$team_name"

echo "Competition archive: $repo_dir/build/$team_name.tar.gz"
