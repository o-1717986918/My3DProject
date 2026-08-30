import argparse
import logging

from mujococodebase.agent import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an RCSSServerMJ agent.")
    parser.add_argument("-t", "--team", default="Default", help="Team name")
    parser.add_argument("-n", "--number", type=int, default=1, help="Player number")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=60000, help="Agent TCP port")
    parser.add_argument("-f", "--field", default="fifa", help="Client field model")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after this many perception/action cycles (for tests)",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=500,
        help="Log a compact state snapshot every N cycles; 0 disables it",
    )
    args = parser.parse_args()
    if args.number < 1 or args.number > 11:
        parser.error("--number must be between 1 and 11")
    if args.max_cycles is not None and args.max_cycles < 1:
        parser.error("--max-cycles must be positive")
    return args


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(handlers=[handler], level=logging.INFO)


def main() -> int:
    args = parse_args()
    configure_logging()
    player = Agent(
        team_name=args.team,
        number=args.number,
        host=args.host,
        port=args.port,
        field=args.field,
    )
    try:
        player.run(max_cycles=args.max_cycles, status_interval=args.status_interval)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
