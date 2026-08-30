import logging

from mujococodebase.decision_maker import DecisionMaker
from mujococodebase.robot import T1, Robot
from mujococodebase.skills.skills_manager import SkillsManager
from mujococodebase.world.world import World
from mujococodebase.server import Server
from mujococodebase.world_parser import WorldParser

logger = logging.getLogger(__file__)


class Agent:
    def __init__(
        self,
        team_name: str = "Default",
        number: int = 1,
        host: str = "localhost",
        port: int = 60000,
        field: str = "fifa",
    ):
        """
        Initializes the agent and all its main components.

        Args:
            team_name (str): The name of the team the agent belongs to.
            number (int): The player number assigned to this agent.
            host (str): The host address of the simulator server.
            port (int): The port number of the simulator server.
            field (str): The name of the field configuration to use.
        """

        self.world: World = World(
            agent=self, team_name=team_name, number=number, field_name=field
        )
        self.world_parser: WorldParser = WorldParser(agent=self)
        self.server: Server = Server(
            host=host, port=port, world_parser=self.world_parser
        )
        self.robot: Robot = T1(agent=self)
        self.skills_manager: SkillsManager = SkillsManager(agent=self)
        self.decision_maker: DecisionMaker = DecisionMaker(agent=self)

    def run(self, max_cycles: int | None = None, status_interval: int = 500):
        """
        Starts the agent’s main control loop.

        This method:
          1. Connects to the simulator server.
          2. Sends the initial configuration (init message).
          3. Enters the main loop, where it:
             - Receives and parses world updates.
             - Updates internal world representation.
             - Executes the decision-making process.
             - Sends the next set of commands to the server.
        """
        self.server.connect()

        self.server.send_immediate(
            f"(init {self.robot.name} {self.world.team_name} {self.world.number})"
        )

        try:
            while max_cycles is None or self.world.cycle_count < max_cycles:
                self.server.receive()

                self.world.update()

                self.decision_maker.update_current_behavior()

                self.server.send()

                if (
                    status_interval > 0
                    and self.world.cycle_count % status_interval == 0
                ):
                    logger.info(
                        "[#%d] cycle=%d time=%.2f pos=(%.2f, %.2f, %.2f) "
                        "up=%.2f ball=(%.2f, %.2f) fresh=%s skill=%s phase=%s",
                        self.world.number,
                        self.world.cycle_count,
                        self.world.server_time,
                        *self.world.global_position,
                        self.world.torso_up_component(),
                        *self.world.ball_pos[:2],
                        self.world.is_ball_fresh(),
                        self.skills_manager.current_skill_name,
                        self.decision_maker.attack_phase.name,
                    )
        finally:
            self.shutdown()

    def shutdown(self):
        """
        Safely shuts down the agent.

        Logs a shutdown message and closes the server connection.
        """
        logger.info("Shutting down.")
        self.server.shutdown()
