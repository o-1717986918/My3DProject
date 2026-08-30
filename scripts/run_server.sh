#!/usr/bin/env bash
set -euo pipefail

mode=${1:-realtime}
agent_port=${2:-60000}
monitor_port=${3:-60001}
render_mode=${4:-headless}
server_bin=${RCSSSERVERMJ_BIN:-rcssservermj}

if ! command -v "$server_bin" >/dev/null 2>&1; then
    echo "RCSSServerMJ executable not found: $server_bin" >&2
    echo "Set RCSSSERVERMJ_BIN=/absolute/path/to/rcssservermj." >&2
    exit 127
fi

speed_args=(--realtime)
if [[ "$mode" == "fast" ]]; then
    speed_args=(--no-realtime)
elif [[ "$mode" != "realtime" ]]; then
    echo "usage: $0 [realtime|fast] [agent-port] [monitor-port] [headless|render]" >&2
    exit 2
fi

render_args=(--no-render)
if [[ "$render_mode" == "render" ]]; then
    render_args=(--render)
elif [[ "$render_mode" != "headless" ]]; then
    echo "usage: $0 [realtime|fast] [agent-port] [monitor-port] [headless|render]" >&2
    exit 2
fi

log_args=()
if [[ -n "${RCSSSERVERMJ_LOGFILE:-}" ]]; then
    log_args=(--logfile "$RCSSSERVERMJ_LOGFILE")
fi

exec "$server_bin" \
    --host 127.0.0.1 \
    --aport "$agent_port" \
    --mport "$monitor_port" \
    --sync \
    "${speed_args[@]}" \
    "${render_args[@]}" \
    "${log_args[@]}" \
    --field fifa7vs7 \
    --rules ssim26
