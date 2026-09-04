"""Local-only HTTP and MJPEG transport for the web match console."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from mujococodebase.web_match_control import MatchHub, parse_control


logger = logging.getLogger(__name__)


class _MatchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler_class(hub: MatchHub, index_html: bytes) -> type[BaseHTTPRequestHandler]:
    class MatchHandler(BaseHTTPRequestHandler):
        server_version = "My3DMatchConsole/1.0"

        def log_message(self, format_string: str, *args: Any) -> None:
            logger.debug("web console: " + format_string, *args)

        def _common_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")

        def _send_bytes(
            self, status: HTTPStatus, content_type: str, body: bytes
        ) -> None:
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", encoded)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                self.send_response(HTTPStatus.OK)
                self._common_headers()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(index_html)))
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self'; style-src 'self' "
                    "'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                    "connect-src 'self'",
                )
                self.end_headers()
                self.wfile.write(index_html)
                return
            if path == "/api/status":
                self._send_json(HTTPStatus.OK, hub.status())
                return
            if path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": not hub.stopped, "frame": hub.status()["frame_sequence"]},
                )
                return
            if path == "/stream.mjpg":
                self._stream_frames()
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.split("?", 1)[0] != "/api/control":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = -1
            if size < 2 or size > 4096:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": "invalid request size"}
                )
                return
            try:
                payload = json.loads(self.rfile.read(size))
                control = parse_control(payload)
                hub.apply_control(control)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._send_json(
                HTTPStatus.ACCEPTED,
                {"accepted": control.action, "status": hub.status()},
            )

        def _stream_frames(self) -> None:
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            sequence = 0
            hub.stream_client(1)
            try:
                while not hub.stopped:
                    frame = hub.wait_for_frame(sequence, timeout=2.0)
                    if frame is None:
                        continue
                    sequence, jpeg = frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.stream_client(-1)

    return MatchHandler


class WebMatchConsole:
    """Own the local browser transport independently from the simulation loop."""

    def __init__(self, hub: MatchHub, host: str, port: int, index_path: Path):
        index_html = index_path.read_bytes()
        self._server = _MatchHTTPServer(
            (host, port), _handler_class(hub, index_html)
        )
        self._thread = Thread(
            target=self._server.serve_forever,
            name="web_match_console",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
