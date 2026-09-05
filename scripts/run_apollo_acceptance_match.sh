#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

max_cycles=${1:-600}
launch_stagger=${APOLLO_LAUNCH_STAGGER:-0.05}
python_bin=${MY3D_PYTHON:-python3}
agent_port=${MATCH_AGENT_PORT:-$((28000 + $$ % 1000))}
monitor_port=${MATCH_MONITOR_PORT:-$((agent_port + 1))}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runtime_dir="$repo_dir/runtime/apollo"
binary="${APOLLO_BINARY:-$runtime_dir/build/ApolloCodeBase}"
asset_root="${APOLLO_ASSET_ROOT:-$runtime_dir/assets}"
if [[ -n "${MATCH_RUN_DIR:-}" ]]; then
    run_dir=$(realpath -m "$MATCH_RUN_DIR")
    mkdir -p "$run_dir"
else
    run_dir=$(mktemp -d -t my3d-apollo-match.XXXXXX)
fi
server_pid=
player_pids=()
client_strategy_args=()
pass_sender_x=-3
pass_sender_y=0
pass_sender_qw=1
pass_sender_qz=0
pass_receiver_x=4
pass_reassert_delay=0.05
pass_reassert_settle_s=0.5
kickoff_command_delay=0.5
kick_calibration_scenario=0
procedural_dribble_scenario=0
procedural_shot_scenario=0
procedural_clear_scenario=0
scenario_ball_x=0
scenario_ball_y=0
staging_ball_x=-20
staging_ball_y=0
strong_kick_track_robot_x=0
strong_kick_track_robot_y=0
strong_kick_release_robot_x=0
strong_kick_release_robot_y=0
strong_kick_opponent_x=0
strong_kick_kicker_team=My3D-A
strong_kick_opponent_team=My3D-B
strong_kick_qw=1
strong_kick_qz=0
parameterized_kick_mode=${APOLLO_ENABLE_PARAMETERIZED_KICK:-1}
learned_kick_mode=${APOLLO_LEARNED_KICK_MODE:-}
if [[ -z "$learned_kick_mode" ]]; then
    learned_kick_mode=$(
        [[ "$parameterized_kick_mode" == 1 ]] && echo active || echo off
    )
fi
# Exercise the best available specialist composition by default. The stable
# walk remains the fallback for unsupported commands and recovery cooldowns.
fast_walk_mode=${APOLLO_ENABLE_FAST_WALK:-1}
rapid_turn_mode=${APOLLO_ENABLE_RAPID_TURN:-1}

case "${APOLLO_ENABLE_PASS_STRATEGY:-1}" in
    1) ;;
    0) client_strategy_args+=(--disable-pass-strategy) ;;
    *)
        echo "APOLLO_ENABLE_PASS_STRATEGY must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${MATCH_PARAMETERIZED_KICK_SCENARIO:-0}" in
    1)
        # Start outside the robot's near-field camera occlusion. The decision
        # layer must perceive and commit the pass, then perform its validated
        # 0.35 m setup before the residual kick is eligible.
        pass_sender_x=-1.0
        pass_sender_y=-0.5
        pass_sender_qw=0.9238795
        pass_sender_qz=0.3826834
        # The planner's 1 m leading offset turns this into the table's
        # validated 2 m target while the 1 m direct candidate is rejected as
        # too near.
        pass_receiver_x=1.0
        ;;
    0) ;;
    *)
        echo "MATCH_PARAMETERIZED_KICK_SCENARIO must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${MATCH_KICK_CALIBRATION_SCENARIO:-0}" in
    1)
        # Motion-identification scene: preserve the real 7v7 strategy and
        # Ready handshake, but remove most long-approach gait-state variance.
        # Enter PlayOn before the passer walks into the torso-occlusion zone,
        # so the world model can establish a new-mode self-vision track.
        pass_sender_x=-0.8
        pass_sender_y=0
        pass_sender_qw=1
        pass_sender_qz=0
        pass_receiver_x=1.0
        pass_reassert_delay=0.01
        pass_reassert_settle_s=0.05
        kickoff_command_delay=0.05
        kick_calibration_scenario=1
        ;;
    0) ;;
    *)
        echo "MATCH_KICK_CALIBRATION_SCENARIO must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${MATCH_PROCEDURAL_DRIBBLE_SCENARIO:-0}" in
    1)
        # Establish vision first, then place the AP in the exact independent
        # trajectory slot: ball local x=0.32 m, y=+0.04 m. Pass planning must
        # be disabled by the caller so the decision layer requests
        # DribbleTouch rather than a TargetedPass.
        pass_sender_x=-0.8
        pass_sender_y=-0.04
        pass_sender_qw=1
        pass_sender_qz=0
        pass_reassert_delay=0.01
        pass_reassert_settle_s=0.05
        kickoff_command_delay=0.05
        procedural_dribble_scenario=1
        ;;
    0) ;;
    *)
        echo "MATCH_PROCEDURAL_DRIBBLE_SCENARIO must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${MATCH_PROCEDURAL_SHOT_SCENARIO:-0}" in
    1)
        # Establish vision outside near-field occlusion, then place the AP in
        # the exact held-out 4 m shot slot: ball local x=0.326 m, y=+0.04 m.
        pass_sender_x=22.7
        pass_sender_y=-0.04
        pass_sender_qw=1
        pass_sender_qz=0
        pass_reassert_delay=0.01
        pass_reassert_settle_s=0.05
        kickoff_command_delay=0.05
        scenario_ball_x=23.5
        staging_ball_x=23.5
        strong_kick_track_robot_x=22.7
        strong_kick_track_robot_y=0
        strong_kick_release_robot_x=23.174
        strong_kick_release_robot_y=-0.04
        strong_kick_opponent_x=-24
        procedural_shot_scenario=1
        ;;
    0) ;;
    *)
        echo "MATCH_PROCEDURAL_SHOT_SCENARIO must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${MATCH_PROCEDURAL_CLEAR_SCENARIO:-0}" in
    1)
        # Defensive-third safety-clear scene. The AP faces +x and enters the
        # independently held-out 6 m clear slot at local x=0.326 m, y=+0.04 m.
        pass_sender_x=-0.8
        pass_sender_y=0
        pass_sender_qw=1
        pass_sender_qz=0
        pass_reassert_delay=0.01
        pass_reassert_settle_s=0.05
        kickoff_command_delay=0.05
        # Use the right-side team so the monitor places the ball at positive
        # server x. Frame normalization maps this to the same canonical -20 m
        # defensive-third state and +x attacking direction used by the policy.
        scenario_ball_x=20
        staging_ball_x=20
        strong_kick_track_robot_x=20.8
        strong_kick_track_robot_y=0
        strong_kick_release_robot_x=20.326
        strong_kick_release_robot_y=0.04
        strong_kick_opponent_x=-24
        strong_kick_kicker_team=My3D-B
        strong_kick_opponent_team=My3D-A
        strong_kick_qw=0
        strong_kick_qz=1
        procedural_clear_scenario=1
        ;;
    0) ;;
    *)
        echo "MATCH_PROCEDURAL_CLEAR_SCENARIO must be 0 or 1" >&2
        exit 2
        ;;
esac

scenario_count=$((
    kick_calibration_scenario + procedural_dribble_scenario +
    procedural_shot_scenario + procedural_clear_scenario
))
if [[ "$scenario_count" -gt 1 ]]; then
    echo "kick calibration scenarios are mutually exclusive" >&2
    exit 2
fi
if [[ ( "$procedural_dribble_scenario" == 1 ||
        "$procedural_shot_scenario" == 1 ||
        "$procedural_clear_scenario" == 1 ) &&
      "${MATCH_PASS_SCENARIO:-0}" != 1 ]]; then
    echo "procedural kick scenarios require MATCH_PASS_SCENARIO=1" >&2
    exit 2
fi
if [[ ( "$procedural_dribble_scenario" == 1 ||
        "$procedural_shot_scenario" == 1 ||
        "$procedural_clear_scenario" == 1 ) &&
      "${APOLLO_ENABLE_PASS_STRATEGY:-1}" != 0 ]]; then
    echo "procedural kick calibration requires APOLLO_ENABLE_PASS_STRATEGY=0" >&2
    exit 2
fi

case "$parameterized_kick_mode" in
    1) client_strategy_args+=(--enable-parameterized-kick) ;;
    0) ;;
    *)
        echo "APOLLO_ENABLE_PARAMETERIZED_KICK must be 0 or 1" >&2
        exit 2
        ;;
esac

case "$learned_kick_mode" in
    active|shadow)
        if [[ "$parameterized_kick_mode" != 1 ]]; then
            echo "learned kick requires APOLLO_ENABLE_PARAMETERIZED_KICK=1" >&2
            exit 2
        fi
        learned_kick_model=${APOLLO_LEARNED_KICK_MODEL:-$HOME/rl_runs/kick-transition-dagger-r2-bc-s10002/policy.onnx}
        # Current best transition candidate. Active control remains bounded by
        # LearnedKickRunner's narrow measured envelope and same-cycle fallback.
        learned_kick_sha256=${APOLLO_LEARNED_KICK_SHA256:-b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1}
        if [[ -z "$learned_kick_model" || ! -f "$learned_kick_model" ]]; then
            echo "APOLLO_LEARNED_KICK_MODEL must name a kick_policy_v3 ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$learned_kick_model" | cut -d " " -f 1)" != "$learned_kick_sha256" ]]; then
            echo "APOLLO_LEARNED_KICK_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        if [[ "$learned_kick_mode" == active ]]; then
            client_strategy_args+=(--enable-learned-kick)
        else
            client_strategy_args+=(--shadow-learned-kick)
        fi
        client_strategy_args+=(--learned-kick-model "$learned_kick_model")
        ;;
    off) ;;
    *)
        echo "APOLLO_LEARNED_KICK_MODE must be off, shadow, or active" >&2
        exit 2
        ;;
esac

if [[ ( "$procedural_dribble_scenario" == 1 ||
        "$procedural_shot_scenario" == 1 ||
        "$procedural_clear_scenario" == 1 ) &&
      "$parameterized_kick_mode" != 1 ]]; then
    echo "procedural kick scenario requires parameterized kick support" >&2
    exit 2
fi

case "$fast_walk_mode" in
    1)
        fast_walk_model=${APOLLO_FAST_WALK_MODEL:-$HOME/rl_runs/stable-motion/fast-walk-recovery-s20261130-v1/policy.onnx}
        fast_walk_sha256=${APOLLO_FAST_WALK_SHA256:-778614c0af7995e2b50d5f677ecbf27b1026c98942e07a22e85ddf2595b21337}
        if [[ -z "$fast_walk_model" || ! -f "$fast_walk_model" ]]; then
            echo "APOLLO_FAST_WALK_MODEL must name the phase-v2 ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$fast_walk_model" | cut -d " " -f 1)" != "$fast_walk_sha256" ]]; then
            echo "APOLLO_FAST_WALK_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        client_strategy_args+=(--enable-fast-walk --fast-walk-model "$fast_walk_model")
        ;;
    0) ;;
    *)
        echo "APOLLO_ENABLE_FAST_WALK must be 0 or 1" >&2
        exit 2
        ;;
esac

case "$rapid_turn_mode" in
    1)
        rapid_turn_model=${APOLLO_RAPID_TURN_MODEL:-$HOME/rl_runs/stable-motion/rapid-turn-s20261101-v1/policy.onnx}
        rapid_turn_sha256=${APOLLO_RAPID_TURN_SHA256:-c086b819d3ffa3dbb971dbcc2bb2e40c949864a4a702546f678d89414c510cca}
        if [[ -z "$rapid_turn_model" || ! -f "$rapid_turn_model" ]]; then
            echo "APOLLO_RAPID_TURN_MODEL must name the validated run-policy ONNX file" >&2
            exit 2
        fi
        if [[ "$(sha256sum "$rapid_turn_model" | cut -d " " -f 1)" != "$rapid_turn_sha256" ]]; then
            echo "APOLLO_RAPID_TURN_MODEL failed the locked SHA-256 check" >&2
            exit 2
        fi
        client_strategy_args+=(--enable-rapid-turn --rapid-turn-model "$rapid_turn_model")
        ;;
    0) ;;
    *)
        echo "APOLLO_ENABLE_RAPID_TURN must be 0 or 1" >&2
        exit 2
        ;;
esac

cleanup() {
    for pid in "${player_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -z "${MATCH_RUN_DIR:-}" && "${KEEP_MATCH_LOGS:-0}" != 1 \
        && "$run_dir" == /tmp/my3d-apollo-match.* ]]; then
        rm -rf -- "$run_dir"
    fi
}
trap cleanup EXIT INT TERM

if ! [[ "$max_cycles" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-cycle-count]" >&2
    exit 2
fi
if ! [[ "$launch_stagger" =~ ^(0|[0-9]+([.][0-9]+)?)$ ]]; then
    echo "APOLLO_LAUNCH_STAGGER must be a non-negative number of seconds" >&2
    exit 2
fi

if [[ -z "${APOLLO_BINARY:-}" ]]; then
    "$repo_dir/scripts/build_apollo_runtime.sh" >/dev/null
elif [[ ! -x "$binary" ]]; then
    echo "Apollo binary is not executable: $binary" >&2
    exit 2
fi
if [[ ! -d "$asset_root" ]]; then
    echo "Apollo asset root does not exist: $asset_root" >&2
    exit 2
fi

(
    cd "$run_dir"
    render_mode=headless
    if [[ "${MATCH_RENDER:-0}" == 1 ]]; then
        render_mode=render
    fi
    exec "$repo_dir/scripts/run_server.sh" realtime "$agent_port" "$monitor_port" "$render_mode"
) >"$run_dir/server.log" 2>&1 &
server_pid=$!
sleep 2

for team in My3D-A My3D-B; do
    for number in $(seq 1 7); do
        "$binary" \
            --team "$team" \
            --player-number "$number" \
            --host 127.0.0.1 \
            --port "$agent_port" \
            --asset-root "$asset_root" \
            --max-cycles "$max_cycles" \
            --status-interval "${APOLLO_STATUS_INTERVAL:-100}" \
            "${client_strategy_args[@]}" \
            >"$run_dir/${team}-${number}.log" 2>&1 &
        player_pids+=("$!")
        # RCSSServerMJ 0.2.1 can reset connections when fourteen ONNX-backed
        # clients initialize in the same scheduler slice. This still starts
        # both complete teams well inside the official three-second limit.
        sleep "$launch_stagger"
    done
done

sleep 4
if [[ "${MATCH_PASS_SCENARIO:-0}" == 1 ]]; then
    # Deterministic open-lane scene for the communication-to-contact pass
    # contract. Both complete teams remain connected; only their poses and the
    # ball are reset through the official monitor protocol.
    "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay 0.05 \
        "(agent (unum 1) (team My3D-A) (move3d -26 0 0.8 1 0 0 0))" \
        "(agent (unum 2) (team My3D-A) (move3d -10 -5 0.8 1 0 0 0))" \
        "(agent (unum 3) (team My3D-A) (move3d -10 5 0.8 1 0 0 0))" \
        "(agent (unum 4) (team My3D-A) (move3d -6 -7 0.8 1 0 0 0))" \
        "(agent (unum 5) (team My3D-A) (move3d -6 7 0.8 1 0 0 0))" \
        "(agent (unum 6) (team My3D-A) (move3d $pass_receiver_x 0 0.8 1 0 0 0))" \
        "(agent (unum 7) (team My3D-A) (move3d $pass_sender_x $pass_sender_y 0.8 $pass_sender_qw 0 0 $pass_sender_qz))" \
        "(agent (unum 1) (team My3D-B) (move3d 26 0 0.8 0 0 0 1))" \
        "(agent (unum 2) (team My3D-B) (move3d 18 -7 0.8 0 0 0 1))" \
        "(agent (unum 3) (team My3D-B) (move3d 18 -4 0.8 0 0 0 1))" \
        "(agent (unum 4) (team My3D-B) (move3d 18 -1 0.8 0 0 0 1))" \
        "(agent (unum 5) (team My3D-B) (move3d 18 1 0.8 0 0 0 1))" \
        "(agent (unum 6) (team My3D-B) (move3d 18 4 0.8 0 0 0 1))" \
        "(agent (unum 7) (team My3D-B) (move3d 18 7 0.8 0 0 0 1))" \
        "(ball (pos $staging_ball_x $staging_ball_y 0.11) (vel 0 0 0))"
    sleep 1
    # The client-side beam command can race the first monitor reset while all
    # fourteen ONNX sessions finish initialization. Re-assert only the actors
    # that define the lane after every client has emitted its first action.
    "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay "$pass_reassert_delay" \
        "(agent (unum 6) (team My3D-A) (move3d $pass_receiver_x 0 0.8 1 0 0 0))" \
        "(agent (unum 7) (team My3D-A) (move3d $pass_sender_x $pass_sender_y 0.8 $pass_sender_qw 0 0 $pass_sender_qz))" \
        "(agent (unum 1) (team My3D-B) (move3d 26 0 0.8 0 0 0 1))" \
        "(agent (unum 2) (team My3D-B) (move3d 18 -7 0.8 0 0 0 1))" \
        "(agent (unum 3) (team My3D-B) (move3d 18 -4 0.8 0 0 0 1))" \
        "(agent (unum 4) (team My3D-B) (move3d 18 -1 0.8 0 0 0 1))" \
        "(agent (unum 5) (team My3D-B) (move3d 18 1 0.8 0 0 0 1))" \
        "(agent (unum 6) (team My3D-B) (move3d 18 4 0.8 0 0 0 1))" \
        "(agent (unum 7) (team My3D-B) (move3d 18 7 0.8 0 0 0 1))" \
        "(ball (pos -20 0 0.11) (vel 0 0 0))"
    sleep "$pass_reassert_settle_s"
elif [[ "${MATCH_PASS_SCENARIO:-0}" != 0 ]]; then
    echo "MATCH_PASS_SCENARIO must be 0 or 1" >&2
    exit 2
fi

"$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
    --host 127.0.0.1 \
    --port "$monitor_port" \
    --delay "$kickoff_command_delay" \
    "(kickOff Left)"

if [[ "${MATCH_PASS_SCENARIO:-0}" == 1 ]]; then
    "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        --delay 0 \
        "(agent (unum 6) (team My3D-A) (move3d $pass_receiver_x 0 0.8 1 0 0 0))" \
        "(agent (unum 7) (team My3D-A) (move3d $pass_sender_x $pass_sender_y 0.8 $pass_sender_qw 0 0 $pass_sender_qz))" \
        "(agent (unum 1) (team My3D-B) (move3d 26 0 0.8 0 0 0 1))" \
        "(agent (unum 2) (team My3D-B) (move3d 18 -7 0.8 0 0 0 1))" \
        "(agent (unum 3) (team My3D-B) (move3d 18 -4 0.8 0 0 0 1))" \
        "(agent (unum 4) (team My3D-B) (move3d 18 -1 0.8 0 0 0 1))" \
        "(agent (unum 5) (team My3D-B) (move3d 18 1 0.8 0 0 0 1))" \
        "(agent (unum 6) (team My3D-B) (move3d 18 4 0.8 0 0 0 1))" \
        "(agent (unum 7) (team My3D-B) (move3d 18 7 0.8 0 0 0 1))"
    sleep 0.1
    "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        "(dropBall)"
    sleep 0.1
    "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        "(ball (pos $scenario_ball_x $scenario_ball_y 0.11) (vel 0 0 0))"
    if [[ "$kick_calibration_scenario" == 1 ]]; then
        # First allow a post-transition camera update at 0.8 m, then remove
        # approach gait phase and accidental pre-kick contacts. Decision,
        # Ready handshake, residual selection, and physics remain unchanged.
        sleep 0.2
        "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
            --host 127.0.0.1 \
            --port "$monitor_port" \
            --delay 0 \
            "(agent (unum 6) (team My3D-A) (move3d $pass_receiver_x 0 0.8 1 0 0 0))" \
            "(agent (unum 7) (team My3D-A) (move3d -0.33 0 0.8 1 0 0 0))"
        sleep 0.05
    elif [[ "$procedural_dribble_scenario" == 1 ]]; then
        # Rebeam at rest so the standalone runner's measured-joint and body
        # velocity guards, not an approach gait phase, own this calibration.
        sleep 0.2
        # Complete a short five-sample static placement inside the 0.08 s
        # release debounce. Unlike the former long rebeam loop, this stops
        # before the trajectory can start and therefore cannot overwrite the
        # strike itself.
        "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
            --host 127.0.0.1 \
            --port "$monitor_port" \
            --delay 0.01 \
            "(agent (unum 7) (team My3D-A) (move3d -0.32 -0.04 0.8 1 0 0 0))" \
            "(agent (unum 7) (team My3D-A) (move3d -0.32 -0.04 0.8 1 0 0 0))" \
            "(agent (unum 7) (team My3D-A) (move3d -0.32 -0.04 0.8 1 0 0 0))" \
            "(agent (unum 7) (team My3D-A) (move3d -0.32 -0.04 0.8 1 0 0 0))" \
            "(agent (unum 7) (team My3D-A) (move3d -0.32 -0.04 0.8 1 0 0 0))"
        sleep 0.05
    elif [[ "$procedural_shot_scenario" == 1 ||
            "$procedural_clear_scenario" == 1 ]]; then
        sleep 0.2
        # Establish a fresh near-contact track at 0.8 m before entering the
        # torso-occluded release slot. Keep both bodies fixed for ten camera
        # updates so monitor-command latency cannot consume the whole stage.
        "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
            --host 127.0.0.1 \
            --port "$monitor_port" \
            --delay 0.02 \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_track_robot_x $strong_kick_track_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))"
        sleep 1.0
        "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
            --host 127.0.0.1 \
            --port "$monitor_port" \
            --delay 0.01 \
            "(agent (unum 1) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x -8 0.8 1 0 0 0))" \
            "(agent (unum 2) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x -6 0.8 1 0 0 0))" \
            "(agent (unum 3) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x -4 0.8 1 0 0 0))" \
            "(agent (unum 4) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x -2 0.8 1 0 0 0))" \
            "(agent (unum 5) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x 2 0.8 1 0 0 0))" \
            "(agent (unum 6) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x 4 0.8 1 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_opponent_team) (move3d $strong_kick_opponent_x 6 0.8 1 0 0 0))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_release_robot_x $strong_kick_release_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_release_robot_x $strong_kick_release_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_release_robot_x $strong_kick_release_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_release_robot_x $strong_kick_release_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_release_robot_x $strong_kick_release_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))" \
            "(ball (pos $scenario_ball_x 0 0.11) (vel 0 0 0))" \
            "(agent (unum 7) (team $strong_kick_kicker_team) (move3d $strong_kick_release_robot_x $strong_kick_release_robot_y 0.8 $strong_kick_qw 0 0 $strong_kick_qz))"
        sleep 0.05
    fi
else
    "$python_bin" "$repo_dir/scripts/send_monitor_command.py" \
        --host 127.0.0.1 \
        --port "$monitor_port" \
        "(dropBall)"
fi

clean_exits=0
for pid in "${player_pids[@]}"; do
    if wait "$pid"; then
        clean_exits=$((clean_exits + 1))
    fi
done

failures=$(
    { grep -Ehi \
        "Failed to start agent|segmentation fault|core dumped|terminate called" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
server_errors=$(grep -Eci "Traceback|ERROR|Segmentation fault|core dumped" \
    "$run_dir/server.log" 2>/dev/null || true)
connections=$(grep -c "New agent connection" "$run_dir/server.log" || true)
joins=$(grep -c "joined the game" "$run_dir/server.log" || true)
play_on=$(
    { grep -El "MY3D_STATUS.*play_on=1" "$run_dir"/My3D-*.log \
        2>/dev/null || true; } | wc -l
)
illegal_defense=$(grep -c "Illegal defense" "$run_dir/server.log" || true)
kick_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=((Parameterized(Residual)?)?Kick(Forward|Stabilize|Hold)|ProceduralKick(Execute|Hold))" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
parameterized_kick_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=(Parameterized(Residual)?Kick(Forward|Stabilize|Hold)|ProceduralKick(Execute|Hold))" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
learned_kick_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=LearnedKick(Execute|Hold)" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
learned_kick_shadow_samples=$(
    { grep -Eh "MY3D_STATUS.*learned_kick_shadow_valid=1" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
procedural_kick_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=ProceduralKick(Execute|Hold).*kick_mode=DribbleTouch" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
procedural_shot_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=ProceduralKick(Execute|Hold).*kick_mode=Shot" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
procedural_clear_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=ProceduralKick(Execute|Hold).*kick_mode=Clear" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
getup_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=GetUp" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
fast_walk_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=FastWalkV2" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
rapid_turn_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=RapidTurnV1" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
rapid_turn_left_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=RapidTurnV1Left" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
rapid_turn_right_samples=$(
    { grep -Eh "MY3D_STATUS.*motion=RapidTurnV1RightMirror" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
pass_plan_samples=$(
    { grep -Eh "MY3D_STATUS.*strategy=Pass" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
pass_ready_samples=$(
    { grep -Eh "MY3D_STATUS.*strategy=Pass.*pass_ready=1" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
targeted_pass_kick_samples=$(
    { grep -Eh \
        "MY3D_STATUS.*motion=(Parameterized(Residual)?)?Kick(Forward|Stabilize|Hold).*kick_mode=TargetedPass" \
        "$run_dir"/My3D-*.log 2>/dev/null || true; } | wc -l
)
pass_contact_events=$("$python_bin" "$repo_dir/scripts/analyze_apollo_pass.py" \
    --metric contacts "$run_dir"/My3D-*.log)
procedural_contact_events=$("$python_bin" "$repo_dir/scripts/analyze_apollo_pass.py" \
    --kick-mode DribbleTouch --metric contacts "$run_dir"/My3D-*.log)
procedural_shot_contact_events=$("$python_bin" "$repo_dir/scripts/analyze_apollo_pass.py" \
    --kick-mode Shot --metric contacts "$run_dir"/My3D-*.log)
procedural_clear_contact_events=$("$python_bin" "$repo_dir/scripts/analyze_apollo_pass.py" \
    --kick-mode Clear --metric contacts "$run_dir"/My3D-*.log)
activation_warnings=0
if [[ -f "$run_dir/MUJOCO_LOG.TXT" ]]; then
    activation_warnings=$(grep -c "Nan, Inf or huge value in CTRL" \
        "$run_dir/MUJOCO_LOG.TXT" || true)
fi

kick_requirement_failed=0
if [[ "${MATCH_REQUIRE_KICK:-0}" == 1 && $kick_samples -eq 0 ]]; then
    kick_requirement_failed=1
fi
parameterized_kick_requirement_failed=0
if [[ "${MATCH_REQUIRE_PARAMETERIZED_KICK:-0}" == 1 && \
    $parameterized_kick_samples -eq 0 ]]; then
    parameterized_kick_requirement_failed=1
fi
procedural_kick_requirement_failed=0
if [[ "${MATCH_REQUIRE_PROCEDURAL_KICK:-0}" == 1 &&
    ( $procedural_kick_samples -eq 0 || $procedural_contact_events -eq 0 ) ]]; then
    procedural_kick_requirement_failed=1
fi
procedural_shot_requirement_failed=0
if [[ "${MATCH_REQUIRE_PROCEDURAL_SHOT:-0}" == 1 &&
    ( $procedural_shot_samples -eq 0 ||
      $procedural_shot_contact_events -eq 0 ) ]]; then
    procedural_shot_requirement_failed=1
fi
procedural_clear_requirement_failed=0
if [[ "${MATCH_REQUIRE_PROCEDURAL_CLEAR:-0}" == 1 &&
    ( $procedural_clear_samples -eq 0 ||
      $procedural_clear_contact_events -eq 0 ) ]]; then
    procedural_clear_requirement_failed=1
fi
pass_requirement_failed=0
if [[ "${MATCH_REQUIRE_PASS:-0}" == 1 ]] && \
    { [[ $pass_plan_samples -eq 0 ]] || [[ $pass_ready_samples -eq 0 ]] || \
      [[ $targeted_pass_kick_samples -eq 0 ]] || [[ $pass_contact_events -eq 0 ]]; }; then
    pass_requirement_failed=1
fi
fast_walk_requirement_failed=0
if [[ "${MATCH_REQUIRE_FAST_WALK:-0}" == 1 && $fast_walk_samples -eq 0 ]]; then
    fast_walk_requirement_failed=1
fi
rapid_turn_requirement_failed=0
if [[ "${MATCH_REQUIRE_RAPID_TURN:-0}" == 1 && $rapid_turn_samples -eq 0 ]]; then
    rapid_turn_requirement_failed=1
fi

if [[ $clean_exits -ne 14 || $connections -ne 14 || $joins -ne 14 \
    || $play_on -ne 14 || $failures -ne 0 || $server_errors -ne 0 \
    || $illegal_defense -ne 0 || $kick_requirement_failed -ne 0 \
    || $parameterized_kick_requirement_failed -ne 0 \
    || $procedural_kick_requirement_failed -ne 0 \
    || $procedural_shot_requirement_failed -ne 0 \
    || $procedural_clear_requirement_failed -ne 0 \
    || $pass_requirement_failed -ne 0 || $fast_walk_requirement_failed -ne 0 \
    || $rapid_turn_requirement_failed -ne 0 ]]; then
    echo "Apollo 7v7 acceptance failed: cycles=$max_cycles clean_exits=$clean_exits " \
        "connections=$connections joins=$joins play_on=$play_on failures=$failures " \
        "server_errors=$server_errors illegal_defense=$illegal_defense " \
        "kick_samples=$kick_samples parameterized_kick_samples=$parameterized_kick_samples " \
        "learned_kick_samples=$learned_kick_samples " \
        "learned_kick_shadow_samples=$learned_kick_shadow_samples " \
        "procedural_kick_samples=$procedural_kick_samples " \
        "procedural_contact_events=$procedural_contact_events " \
        "procedural_shot_samples=$procedural_shot_samples " \
        "procedural_shot_contact_events=$procedural_shot_contact_events " \
        "procedural_clear_samples=$procedural_clear_samples " \
        "procedural_clear_contact_events=$procedural_clear_contact_events " \
        "getup_samples=$getup_samples " \
        "fast_walk_samples=$fast_walk_samples " \
        "rapid_turn_samples=$rapid_turn_samples " \
        "rapid_turn_left_samples=$rapid_turn_left_samples " \
        "rapid_turn_right_samples=$rapid_turn_right_samples " \
        "pass_plan_samples=$pass_plan_samples pass_ready_samples=$pass_ready_samples " \
        "targeted_pass_kick_samples=$targeted_pass_kick_samples " \
        "pass_contact_events=$pass_contact_events " \
        "activation_warnings=$activation_warnings" >&2
    tail -100 "$run_dir/server.log" >&2
    if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
        echo "Logs preserved at $run_dir" >&2
    fi
    exit 1
fi

echo "Apollo 7v7 acceptance passed: cycles=$max_cycles clean_exits=$clean_exits " \
    "connections=$connections joins=$joins play_on=$play_on failures=$failures " \
    "server_errors=$server_errors illegal_defense=$illegal_defense " \
    "kick_samples=$kick_samples parameterized_kick_samples=$parameterized_kick_samples " \
    "learned_kick_samples=$learned_kick_samples " \
    "learned_kick_shadow_samples=$learned_kick_shadow_samples " \
    "procedural_kick_samples=$procedural_kick_samples " \
    "procedural_contact_events=$procedural_contact_events " \
    "procedural_shot_samples=$procedural_shot_samples " \
    "procedural_shot_contact_events=$procedural_shot_contact_events " \
    "procedural_clear_samples=$procedural_clear_samples " \
    "procedural_clear_contact_events=$procedural_clear_contact_events " \
    "getup_samples=$getup_samples " \
    "fast_walk_samples=$fast_walk_samples " \
    "rapid_turn_samples=$rapid_turn_samples " \
    "rapid_turn_left_samples=$rapid_turn_left_samples " \
    "rapid_turn_right_samples=$rapid_turn_right_samples " \
    "pass_plan_samples=$pass_plan_samples pass_ready_samples=$pass_ready_samples " \
    "targeted_pass_kick_samples=$targeted_pass_kick_samples " \
    "pass_contact_events=$pass_contact_events " \
    "activation_warnings=$activation_warnings"
if [[ "${KEEP_MATCH_LOGS:-0}" == 1 ]]; then
    echo "Logs preserved at $run_dir"
fi
