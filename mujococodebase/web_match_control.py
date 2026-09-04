"""Thread-safe state and validated controls for the local web match console."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from threading import Condition, RLock
from typing import Any


ALLOWED_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class ControlMessage:
    """One validated browser control request."""

    action: str
    values: dict[str, float | str]


def _bounded_number(
    payload: dict[str, Any], name: str, *, minimum: float, maximum: float
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def parse_control(payload: Any) -> ControlMessage:
    """Validate a JSON control payload without accepting arbitrary commands."""

    if not isinstance(payload, dict):
        raise ValueError("control payload must be an object")
    action = payload.get("action")
    if not isinstance(action, str):
        raise ValueError("action must be a string")

    no_argument_actions = {
        "pause",
        "resume",
        "toggle_pause",
        "step",
        "camera_cycle",
        "drop_ball",
        "reset_ball",
    }
    if action in no_argument_actions:
        return ControlMessage(action, {})

    if action == "speed":
        speed = _bounded_number(payload, "speed", minimum=0.25, maximum=4.0)
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"speed must be one of {ALLOWED_SPEEDS}")
        return ControlMessage(action, {"speed": speed})

    if action == "camera_mode":
        mode = payload.get("mode")
        if mode not in {"static", "follow"}:
            raise ValueError("camera mode must be static or follow")
        return ControlMessage(action, {"mode": mode})

    if action == "camera_drag":
        mode = payload.get("mode")
        if mode not in {"orbit", "pan"}:
            raise ValueError("camera drag mode must be orbit or pan")
        return ControlMessage(
            action,
            {
                "mode": mode,
                "dx": _bounded_number(payload, "dx", minimum=-500, maximum=500),
                "dy": _bounded_number(payload, "dy", minimum=-500, maximum=500),
            },
        )

    if action == "camera_zoom":
        return ControlMessage(
            action,
            {
                "delta": _bounded_number(
                    payload, "delta", minimum=-10, maximum=10
                )
            },
        )

    if action == "kickoff":
        side = payload.get("side")
        if side not in {"left", "right"}:
            raise ValueError("kickoff side must be left or right")
        return ControlMessage(action, {"side": side})

    raise ValueError(f"unsupported control action: {action}")


class MatchHub:
    """Share frames, match state, pacing, and controls across server threads."""

    def __init__(self, *, initial_speed: float = 1.0) -> None:
        if initial_speed not in ALLOWED_SPEEDS:
            raise ValueError(f"initial speed must be one of {ALLOWED_SPEEDS}")
        self._lock = RLock()
        self._frame_ready = Condition(self._lock)
        self._run_state = Condition(self._lock)
        self._latest_frame = b""
        self._frame_sequence = 0
        self._controls: deque[ControlMessage] = deque()
        self._paused = False
        self._step_tokens = 0
        self._speed = initial_speed
        self._stopped = False
        self._stream_clients = 0
        self._status: dict[str, Any] = {
            "server_state": "starting",
            "camera_mode": "static",
            "frame_id": 0,
            "left_team": "Waiting",
            "right_team": "Waiting",
            "left_score": 0,
            "right_score": 0,
            "play_time": 0.0,
            "play_mode": "WaitingForServer",
            "render_fps": 0.0,
        }

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._status,
                "paused": self._paused,
                "speed": self._speed,
                "frame_sequence": self._frame_sequence,
                "stream_clients": self._stream_clients,
                "stopped": self._stopped,
            }

    def update_status(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._status.update(values)

    def publish_frame(self, jpeg: bytes, status: dict[str, Any]) -> int:
        if not jpeg:
            raise ValueError("a published frame cannot be empty")
        with self._frame_ready:
            self._latest_frame = jpeg
            self._frame_sequence += 1
            self._status.update(status)
            self._frame_ready.notify_all()
            return self._frame_sequence

    def wait_for_frame(
        self, after_sequence: int, *, timeout: float = 2.0
    ) -> tuple[int, bytes] | None:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._frame_sequence > after_sequence or self._stopped,
                timeout=timeout,
            )
            if self._frame_sequence <= after_sequence:
                return None
            return self._frame_sequence, self._latest_frame

    def stream_client(self, delta: int) -> None:
        with self._lock:
            self._stream_clients = max(0, self._stream_clients + delta)

    def apply_control(self, control: ControlMessage) -> None:
        with self._run_state:
            if control.action == "pause":
                self._paused = True
            elif control.action == "resume":
                self._paused = False
                self._step_tokens = 0
            elif control.action == "toggle_pause":
                self._paused = not self._paused
                if not self._paused:
                    self._step_tokens = 0
            elif control.action == "step":
                self._paused = True
                self._step_tokens = min(5, self._step_tokens + 1)
            elif control.action == "speed":
                self._speed = float(control.values["speed"])
            else:
                self._controls.append(control)
            self._run_state.notify_all()

    def poll_controls(self) -> list[ControlMessage]:
        with self._lock:
            controls = list(self._controls)
            self._controls.clear()
            return controls

    def wait_for_turn(self) -> str | None:
        """Wait for running, step, control-only, or stopped state."""

        with self._run_state:
            self._run_state.wait_for(
                lambda: (
                    not self._paused
                    or self._step_tokens > 0
                    or bool(self._controls)
                    or self._stopped
                )
            )
            if self._stopped:
                return None
            if self._paused and self._controls:
                return "control"
            if self._paused:
                self._step_tokens -= 1
                return "step"
            return "run"

    def wait_delay(self, seconds: float) -> bool:
        """Wait for simulation pacing, interrupting promptly on pause/stop."""

        if seconds <= 0:
            return True
        with self._run_state:
            self._run_state.wait(timeout=seconds)
            return not self._stopped and not self._paused

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._status["server_state"] = "stopped"
            self._run_state.notify_all()
            self._frame_ready.notify_all()
