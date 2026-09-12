#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

# Adapter used by the existing browser match launcher.  That launcher adds one
# legacy strategy flag which is not part of the baseline-first rebuild CLI.
# Drop only that flag and explicitly request the two default-on capabilities so
# the launched process command line remains auditable.

set -euo pipefail

binary=${APOLLO_REBUILD_BINARY:-/home/win98/.cache/my3d/apollo-rebuild-build/ApolloCodeBase}
filtered_args=()
dynamic_pass_args=()
goalkeeper_args=()

for arg in "$@"; do
    case "$arg" in
        --disable-team-tactics) ;;
        *) filtered_args+=("$arg") ;;
    esac
done

case "${APOLLO_REBUILD_ENABLE_DYNAMIC_PASS:-1}" in
    0) dynamic_pass_args+=(--disable-dynamic-pass) ;;
    1) dynamic_pass_args+=(--enable-dynamic-pass) ;;
    *)
        echo "APOLLO_REBUILD_ENABLE_DYNAMIC_PASS must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${APOLLO_REBUILD_ENABLE_GOALKEEPER_INTERCEPT:-1}" in
    0) goalkeeper_args+=(--disable-goalkeeper-intercept) ;;
    1) goalkeeper_args+=(--enable-goalkeeper-intercept) ;;
    *)
        echo "APOLLO_REBUILD_ENABLE_GOALKEEPER_INTERCEPT must be 0 or 1" >&2
        exit 2
        ;;
esac

exec "$binary" \
    "${filtered_args[@]}" \
    "${dynamic_pass_args[@]}" \
    "${goalkeeper_args[@]}"
