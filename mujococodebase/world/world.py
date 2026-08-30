from dataclasses import Field
import numpy as np
from mujococodebase.utils.math_ops import MathOps
from mujococodebase.world.other_robot import OtherRobot
from mujococodebase.world.field import FIFAField, HLAdultField, MyField
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
        self.ball_velocity: np.ndarray = np.zeros(3)
        self.ball_last_seen_time: float | None = None
        self.is_ball_pos_updated: bool = False
        self.cycle_count: int = 0
        self.our_team_players: list[OtherRobot] = [
            OtherRobot() for _ in range(self.MAX_PLAYERS_PER_TEAM)
        ]
        self.their_team_players: list[OtherRobot] = [
            OtherRobot(is_teammate=False) for _ in range(self.MAX_PLAYERS_PER_TEAM)
        ]
        self.field: Field = self.__initialize_field(field_name=field_name)

    def update(self) -> None:
        """
        Updates the world state
        """
        self.playmode_group = PlayModeGroupEnum.get_group_from_playmode(
            playmode=self.playmode, is_left_team=self.is_left_team
        )
        self.cycle_count += 1

    def is_fallen(self) -> bool:
        return self.global_position[2] < 0.3

    def torso_up_component(self) -> float:
        """Return the world-z component of the torso's local up axis."""
        quaternion = np.asarray(self.agent.robot.global_orientation_quat, dtype=float)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            return -1.0
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6:
            return -1.0
        x, y, _, _ = quaternion / norm
        return float(1.0 - 2.0 * (x * x + y * y))

    def is_upright(
        self, min_height: float = 0.45, min_up_component: float = 0.65
    ) -> bool:
        """Whether the torso is high enough and points predominantly upward."""
        return (
            self.global_position[2] >= min_height
            and self.torso_up_component() >= min_up_component
        )

    def is_ball_fresh(self, max_age: float = 0.5) -> bool:
        """Whether the current ball estimate was observed recently enough to act on."""
        if self.ball_last_seen_time is None or self.server_time is None:
            return False
        return 0.0 <= self.server_time - self.ball_last_seen_time <= max_age

    def to_simulator_pose(
        self, position_2d: np.ndarray | list | tuple, orientation_deg: float
    ) -> tuple[np.ndarray, float]:
        """Convert a canonical team-frame pose to the simulator's global frame."""
        position = np.asarray(position_2d, dtype=float).copy()
        orientation = float(orientation_deg)
        if self.is_left_team is False:
            position = -position
            orientation = MathOps.normalize_deg(orientation - 180.0)
        return position, orientation

    def __initialize_field(self, field_name: str) -> Field:
        if field_name in (
            "hl_adult",
            "hl_adult_2020",
            "hl_adult_2019",
        ):
            return HLAdultField(world=self)
        elif field_name == "my_field":
            return MyField(world=self)
        else:
            return FIFAField(world=self)
