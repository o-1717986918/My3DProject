from abc import ABC, abstractmethod


class Skill(ABC):
    """
    Base interface for skills.
    Each skill must implement:
      - execute(reset: bool, *args) -> bool
      - is_ready(*args) -> bool
    """

    def __init__(self, agent):
        from mujococodebase.agent import Agent  # type hinting

        self.agent: Agent = agent

    @abstractmethod
    def execute(self, reset: bool, *args, **kwargs) -> bool:
        """
        Executes one step of the skill.

        Parameters
        ----------
        reset : bool
            Indicates if this is the first invocation of this skill (should reset internal state).
        *args
            Skill-specific arguments.

        Returns
        -------
        finished : bool
            True if the skill has finished execution.
        """
        raise NotImplementedError("Method 'execute' must be implemented in the Skill.")

    @abstractmethod
    def is_ready(self, *args) -> bool:
        """
        Checks if the current conditions allow starting the skill.

        Returns
        -------
        ready : bool
        """
        raise NotImplementedError("Method 'is_ready' must be implemented in the Skill.")
