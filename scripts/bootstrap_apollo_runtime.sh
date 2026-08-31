#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
version=1.22.0
archive="onnxruntime-linux-x64-${version}.tgz"
expected_sha256=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3
thirdparty_dir="$runtime_dir/deploy/thirdparty"
onnx_root="$thirdparty_dir/onnxruntime-linux-x64-${version}"

if [[ -f "$onnx_root/include/onnxruntime_cxx_api.h" \
    && -f "$onnx_root/lib/libonnxruntime.so.${version}" ]]; then
    echo "Apollo dependency ready: $onnx_root"
    exit 0
fi

if [[ ! -f /usr/include/yaml-cpp/yaml.h ]]; then
    echo "Missing yaml-cpp headers. Install libyaml-cpp-dev in WSL first." >&2
    exit 1
fi

tmp_dir=$(mktemp -d -t my3d-apollo-bootstrap.XXXXXX)
cleanup() {
    if [[ "$tmp_dir" == /tmp/my3d-apollo-bootstrap.* ]]; then
        rm -rf -- "$tmp_dir"
    fi
}
trap cleanup EXIT INT TERM

url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive}"
curl -fL --retry 3 -o "$tmp_dir/$archive" "$url"
echo "$expected_sha256  $tmp_dir/$archive" | sha256sum --check --status

mkdir -p "$thirdparty_dir"
tar -xzf "$tmp_dir/$archive" -C "$thirdparty_dir"
test -f "$onnx_root/include/onnxruntime_cxx_api.h"
test -f "$onnx_root/lib/libonnxruntime.so.${version}"
echo "Apollo dependency installed: $onnx_root"
