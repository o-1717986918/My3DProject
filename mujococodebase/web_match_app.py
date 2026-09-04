"""Run RCSSServerMJ with an EGL-to-browser match console."""

from __future__ import annotations

import argparse
import logging
import os
import signal
from pathlib import Path
from types import FrameType


# This must be selected before importing MuJoCo through RCSSServerMJ.
os.environ.setdefault("MUJOCO_GL", "egl")

from rcsssmj.games.soccer.game_phase import GamePhase  # noqa: E402
from rcsssmj.games.soccer.server.soccer_server import (  # noqa: E402
    SoccerSimServer,
)
from rcsssmj.games.soccer.sim.soccer_referee import SoccerReferee  # noqa: E402
from rcsssmj.games.soccer.sim.soccer_sim import SoccerSimulation  # noqa: E402
from rcsssmj.games.soccer.soccer_fields import (  # noqa: E402
    SoccerFieldVersions,
    create_soccer_field,
)
from rcsssmj.games.soccer.soccer_rules import (  # noqa: E402
    SoccerRuleBooks,
    create_soccer_rule_book,
)
import rcsssmj.server.server as server_module  # noqa: E402

from mujococodebase.web_match_control import (  # noqa: E402
    ALLOWED_SPEEDS,
    MatchHub,
)
from mujococodebase.web_match_http import WebMatchConsole  # noqa: E402
from mujococodebase.web_match_runtime import (  # noqa: E402
    WebMujocoMonitor,
    WebPacedServerMixin,
)


logger = logging.getLogger(__name__)


class WebSoccerSimServer(WebPacedServerMixin, SoccerSimServer):
    def __init__(self, *args, hub: MatchHub, **kwargs) -> None:
        self.hub = hub
        super().__init__(*args, **kwargs)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoboCup 3D server with a local Windows browser console"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--aport", type=int, default=60000)
    parser.add_argument("--mport", type=int, default=60001)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--field", default="fifa7vs7")
    parser.add_argument("--rules", default="ssim26")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--time", type=float)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--render-interval", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()

    fields = {str(version.value) for version in SoccerFieldVersions}
    rules = {str(book.value) for book in SoccerRuleBooks}
    phases = {phase.value for phase in GamePhase}
    if args.field not in fields:
        parser.error(f"unknown field: {args.field}")
    if args.rules not in rules:
        parser.error(f"unknown rules: {args.rules}")
    if args.phase not in phases:
        parser.error(f"unknown phase: {args.phase}")
    if args.speed not in ALLOWED_SPEEDS:
        parser.error(f"speed must be one of {ALLOWED_SPEEDS}")
    if not 320 <= args.width <= 1920 or not 240 <= args.height <= 1080:
        parser.error("render size must be within 320x240 and 1920x1080")
    if not 1 <= args.render_interval <= 20:
        parser.error("render interval must be between 1 and 20 cycles")
    if not 50 <= args.jpeg_quality <= 95:
        parser.error("JPEG quality must be between 50 and 95")
    return args


def main() -> int:
    args = _arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    hub = MatchHub(initial_speed=args.speed)
    index_path = Path(__file__).with_name("web_match_index.html")
    console = WebMatchConsole(hub, args.web_host, args.web_port, index_path)

    rule_book = create_soccer_rule_book(args.rules)
    field = create_soccer_field(args.field)
    simulation = SoccerSimulation(
        field=field,
        rules=rule_book,
        referee=SoccerReferee(),
        initial_game_phase=GamePhase.from_value(args.phase),
        initial_play_time=args.time,
    )

    server_module.MujocoMonitor = lambda model, _interval: WebMujocoMonitor(
        model,
        hub,
        render_interval=args.render_interval,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
    )
    server = WebSoccerSimServer(
        sim=simulation,
        host=args.host,
        agent_port=args.aport,
        monitor_port=args.mport,
        sync_mode=args.sync,
        real_time=True,
        render=True,
        hub=hub,
    )

    def stop(_signal: int, _frame: FrameType | None) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    console.start()
    logger.info(
        "Web match console ready at http://%s:%d/",
        args.web_host,
        console.port,
    )
    try:
        server.run()
    finally:
        hub.stop()
        console.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
