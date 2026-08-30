"""Send one or more length-prefixed commands to the RCSSServerMJ monitor port."""

import argparse
import socket
import struct
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("commands", nargs="+", help='e.g. "(playMode PlayOn)"')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=60001)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between commands so the monitor queue preserves each one",
    )
    args = parser.parse_args()

    for command in args.commands:
        with socket.create_connection(
            (args.host, args.port), timeout=3.0
        ) as connection:
            payload = command.encode()
            connection.sendall(struct.pack(">I", len(payload)) + payload)
        time.sleep(args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
