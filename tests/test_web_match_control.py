import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mujococodebase.web_match_control import MatchHub, parse_control
from mujococodebase.web_match_http import WebMatchConsole


def test_control_parser_accepts_bounded_native_controls() -> None:
    assert parse_control({"action": "kickoff", "side": "left"}).values == {
        "side": "left"
    }
    drag = parse_control(
        {"action": "camera_drag", "mode": "orbit", "dx": 12, "dy": -4}
    )
    assert drag.values == {"mode": "orbit", "dx": 12.0, "dy": -4.0}
    assert parse_control({"action": "speed", "speed": 4}).values["speed"] == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "speed", "speed": 3},
        {"action": "kickoff", "side": "either"},
        {"action": "camera_drag", "mode": "orbit", "dx": 1000, "dy": 0},
        {"action": "raw_command", "command": "(agent ...)"},
        ["pause"],
    ],
)
def test_control_parser_rejects_unbounded_or_arbitrary_input(payload) -> None:
    with pytest.raises(ValueError):
        parse_control(payload)


def test_match_hub_tracks_frames_pause_step_and_queued_controls() -> None:
    hub = MatchHub()
    sequence = hub.publish_frame(b"jpeg", {"frame_id": 4})
    assert sequence == 1
    assert hub.wait_for_frame(0, timeout=0) == (1, b"jpeg")

    hub.apply_control(parse_control({"action": "pause"}))
    hub.apply_control(parse_control({"action": "step"}))
    assert hub.wait_for_turn() == "step"
    assert hub.status()["paused"] is True

    hub.apply_control(parse_control({"action": "camera_mode", "mode": "follow"}))
    assert hub.wait_for_turn() == "control"
    controls = hub.poll_controls()
    assert [control.action for control in controls] == ["camera_mode"]
    assert hub.poll_controls() == []


def test_web_console_contains_native_interaction_contract() -> None:
    html = (
        Path(__file__).parents[1]
        / "mujococodebase"
        / "web_match_index.html"
    ).read_text(encoding="utf-8")
    for contract in (
        "/stream.mjpg",
        "/api/control",
        "camera_drag",
        "camera_zoom",
        "toggle_pause",
        "requestFullscreen",
    ):
        assert contract in html


def test_http_console_exposes_status_and_rejects_raw_commands() -> None:
    hub = MatchHub()
    index = (
        Path(__file__).parents[1]
        / "mujococodebase"
        / "web_match_index.html"
    )
    console = WebMatchConsole(hub, "127.0.0.1", 0, index)
    console.start()
    base_url = f"http://127.0.0.1:{console.port}"
    try:
        with urlopen(f"{base_url}/api/status", timeout=2) as response:
            assert json.load(response)["paused"] is False

        request = Request(
            f"{base_url}/api/control",
            data=json.dumps({"action": "pause"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 202
        assert hub.status()["paused"] is True

        raw_request = Request(
            f"{base_url}/api/control",
            data=json.dumps(
                {"action": "raw_command", "command": "(agent ...)"}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(raw_request, timeout=2)
        assert error.value.code == 400
    finally:
        hub.stop()
        console.stop()
