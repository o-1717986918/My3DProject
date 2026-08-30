#!/usr/bin/env bash
set -euo pipefail

exec scripts/run_selfplay.sh \
    "${1:-127.0.0.1}" \
    "${2:-60000}" \
    "${3:-60001}" \
    "${4:-}"
