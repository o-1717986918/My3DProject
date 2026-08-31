#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

set -euo pipefail

# Package ApolloCodeBase for deployment
# Usage: pack.sh [destination] [package-and-default-team-name]
# Output: <destination>/<name>.tar.gz

DEST="${1:-.}"
PACK_NAME="${2:-ApolloCodeBase}"
if [[ ! "$PACK_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Error: package name may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$repo_root/build"
binary="$build_dir/ApolloCodeBase"
onnx_distribution="${ONNXRUNTIME_ROOT:-$repo_root/deploy/thirdparty/onnxruntime-linux-x64-1.22.0}"

if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  echo "Error: $build_dir is not configured. Run cmake -S . -B build first." >&2
  exit 1
fi

cmake --build "$build_dir" -j"$(nproc)"

if [[ ! -x "$binary" ]]; then
  echo "Error: $binary not found after build." >&2
  exit 1
fi

staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

pkg="$staging/$PACK_NAME"
mkdir -p "$pkg/libs"

cp "$binary" "$pkg/"

cp -r "$repo_root/assets" "$pkg/"
cp "$repo_root/LICENSE.md" "$pkg/"
cp "$repo_root/README.md" "$repo_root/UPSTREAM.md" "$pkg/"
printf '%s\n' "$PACK_NAME" > "$pkg/team_name.txt"

mkdir -p "$pkg/third_party/onnxruntime"
cp "$onnx_distribution/LICENSE" "$pkg/third_party/onnxruntime/"
cp "$onnx_distribution/ThirdPartyNotices.txt" "$pkg/third_party/onnxruntime/"

# GPL corresponding source and build definitions travel with the binary.
mkdir -p "$pkg/source"
cp -r "$repo_root/src" "$repo_root/tests" "$pkg/source/"
cp "$repo_root/CMakeLists.txt" "$repo_root/main.cc" "$repo_root/pack.sh" "$pkg/source/"
ln -s ../assets "$pkg/source/assets"

# Non-system shared libraries. yaml-cpp and ONNX Runtime must be packaged
# when they remain dynamic dependencies.
while IFS= read -r line; do
  lib=""
  if [[ "$line" == *"=>"* ]]; then
    lib="$(awk '{print $3}' <<< "$line")"
  else
    lib="$(awk '{print $1}' <<< "$line")"
  fi

  [[ "$lib" == /* && -f "$lib" ]] || continue

  base="$(basename "$lib")"
  case "$base" in
    linux-vdso*|ld-linux*|libc.so*|libm.so*|libpthread.so*|librt.so*|libdl.so*)
      ;;
    libyaml-cpp.so*|libonnxruntime.so*)
      cp -L "$lib" "$pkg/libs/$base"
      ;;
    *)
      case "$lib" in
        /lib/x86_64-linux-gnu/*|/usr/lib/x86_64-linux-gnu/*) ;;  # skip base system libs
        *) cp -L "$lib" "$pkg/libs/$base" ;;
      esac
      ;;
  esac
done < <(ldd "$binary")

ensure_packaged_soname() {
  local soname="$1"

  if [[ -e "$pkg/libs/$soname" ]]; then
    return
  fi

  local lib
  for lib in "$pkg"/libs/*; do
    [[ -e "$lib" ]] || continue
    if [[ "$(readelf -d "$lib" 2>/dev/null | awk -F'[][]' '/SONAME/ {print $2; exit}')" == "$soname" ]]; then
      ln -sf "$(basename "$lib")" "$pkg/libs/$soname"
      return
    fi
  done

  echo "Error: $binary needs $soname but it was not packaged under $pkg/libs." >&2
  exit 1
}

while IFS= read -r needed; do
  case "$needed" in
    libyaml-cpp.so*|libonnxruntime.so*) ensure_packaged_soname "$needed" ;;
  esac
done < <(readelf -d "$binary" | awk -F'[][]' '/NEEDED/ {print $2}')

missing="$(LD_LIBRARY_PATH="$pkg/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$pkg/ApolloCodeBase" | awk '/not found/')"
if [[ -n "$missing" ]]; then
  echo "Error: packaged binary still has unresolved shared libraries:" >&2
  echo "$missing" >&2
  exit 1
fi

# Start / kill scripts adapted for packaged layout
cat > "$pkg/start.sh" << 'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${dir}/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

pid_file="/tmp/apollo_code_base_mj_team.pids"
host="${1:-127.0.0.1}"
port="${2:-60000}"
team_name="${TEAM_NAME:-$(cat "$dir/team_name.txt")}"

rm -f "$pid_file"
touch "$pid_file"

for i in $(seq 1 7); do
  "$dir/ApolloCodeBase" --team "$team_name" --player-number "$i" \
    --host "$host" --port "$port" \
    --asset-root "$dir/assets" \
    2>"$dir/err_${i}.txt" >/dev/null &
  echo $! >> "$pid_file"
  sleep 0.1
done
SCRIPT
chmod +x "$pkg/start.sh"

cat > "$pkg/kill.sh" << 'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

pid_file="/tmp/apollo_code_base_mj_team.pids"

if [[ -f "$pid_file" ]]; then
  while IFS= read -r pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done < "$pid_file"
  rm -f "$pid_file"
fi

pkill -f "ApolloCodeBase.*assets" 2>/dev/null || true
SCRIPT
chmod +x "$pkg/kill.sh"

# Pack
mkdir -p "$DEST"
tar -czf "$DEST/$PACK_NAME.tar.gz" -C "$staging" "$PACK_NAME"
echo "Packaged to $DEST/$PACK_NAME.tar.gz"
