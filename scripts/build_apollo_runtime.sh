#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
build_dir="$runtime_dir/build"

"$repo_dir/scripts/bootstrap_apollo_runtime.sh"

cmake -S "$runtime_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE="${APOLLO_BUILD_TYPE:-RelWithDebInfo}"
cmake --build "$build_dir" -j"${APOLLO_BUILD_JOBS:-2}"

test -x "$build_dir/ApolloCodeBase"
echo "Apollo runtime built: $build_dir/ApolloCodeBase"
