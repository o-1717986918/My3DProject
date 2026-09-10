#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source_dir="$repo_dir/runtime/apollo_rebuild"
build_dir=${APOLLO_REBUILD_BUILD_DIR:-/home/win98/.cache/my3d/apollo-rebuild-build}
onnxruntime_root=${APOLLO_REBUILD_ONNXRUNTIME_ROOT:-$repo_dir/runtime/apollo/deploy/thirdparty/onnxruntime-linux-x64-1.22.0}

if [[ ! -f "$onnxruntime_root/include/onnxruntime_cxx_api.h" || \
      ! -f "$onnxruntime_root/lib/libonnxruntime.so.1.22.0" ]]; then
    echo "ONNX Runtime dependency is missing: $onnxruntime_root" >&2
    exit 2
fi

cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT="$onnxruntime_root"
cmake --build "$build_dir" --parallel "${APOLLO_BUILD_JOBS:-2}"

test -x "$build_dir/ApolloCodeBase"
echo "Apollo rebuild binary: $build_dir/ApolloCodeBase"

