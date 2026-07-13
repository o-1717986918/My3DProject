import argparse
import logging
from mujococodebase.agent import Agent

ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setLevel(logging.INFO)
logging.basicConfig(handlers=[ch], level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Start a MuJoCo Agent.")

parser.add_argument("-t", "--team", type=str, default="Default", help="Team name")
parser.add_argument("-n", "--number", type=int, default=1, help="Player number")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
parser.add_argument("--port", type=int, default=60000, help="Server port")
parser.add_argument("-f", "--field", type=str, default='fifa', help="Field to be played")

args = parser.parse_args()

player = Agent(
    team_name=args.team,
    number=args.number,
    host=args.host,
    port=args.port,
    field=args.field
)

player.run()
