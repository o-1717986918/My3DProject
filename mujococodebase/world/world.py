from dataclasses import Field
import numpy as np
from mujococodebase.world.other_robot import OtherRobot
from mujococodebase.world.field import FIFAField, HLAdultField
from mujococodebase.world.play_mode import PlayModeEnum, PlayModeGroupEnum

class World:
    """
    Represents the current simulation world, containing all relevant
    information about the environment, the ball, and the robots.
    """

    MAX_PLAYERS_PER_TEAM = 11
    
    def __init__(self, agent, team_name: str, number: int, field_name: str):
        """
        Initializes the world state.

        Args:
            agent: Reference to the agent that owns this world.
            team_name (str): The name of the agent's team.
            number (int): The player's number within the team.
            field_name (str): The name of the field to initialize
                              (e.g., 'fifa' or 'hl_adult').
        """
        
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent
        self.team_name: str = team_name
        self.number: int = number
        self.playmode: PlayModeEnum = PlayModeEnum.NOT_INITIALIZED
        self.playmode_group: PlayModeGroupEnum = PlayModeGroupEnum.NOT_INITIALIZED
        self.is_left_team: bool = None
        self.game_time: float = None
        self.server_time: float = None
        self.score_left: int = None
        self.score_right: int = None
        self.their_team_name: str = None
        self.last_server_time: str = None
        self._global_cheat_position: np.ndarray = np.zeros(3)
        self.global_position: np.ndarray = np.zeros(3)
        self.ball_pos: np.ndarray = np.zeros(3)
        self.is_ball_pos_updated: bool = False
        self.our_team_players: list[OtherRobot] = [OtherRobot() for _ in range(self.MAX_PLAYERS_PER_TEAM)]
        self.their_team_players: list[OtherRobot] = [OtherRobot(is_teammate=False) for _ in range(self.MAX_PLAYERS_PER_TEAM)]
        self.field: Field = self.__initialize_field(field_name=field_name)

    def update(self) -> None:
        """
        Updates the world state
        """
        self.playmode_group = PlayModeGroupEnum.get_group_from_playmode(
            playmode=self.playmode, is_left_team=self.is_left_team
        )

    def is_fallen(self) -> bool:
        return self.global_position[2] < 0.3
    
    def __initialize_field(self, field_name: str) -> Field:
        if field_name in ('hl_adult', 'hl_adult_2020', 'hl_adult_2019',):
            return HLAdultField(world=self)
        else:
            return FIFAField(world=self)