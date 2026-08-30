import numpy as np

from mujococodebase.skills.skill import Skill


class KickRight(Skill):
    """Stable learned-policy kick: short forward burst followed by stabilization."""

    FORWARD_SECONDS = 0.65
    STABILIZE_SECONDS = 0.35

    def __init__(self, agent):
        super().__init__(agent)
        self.start_time = None

    def execute(self, reset: bool, *args, **kwargs) -> bool:
        server_time = self.agent.world.server_time
        if reset or self.start_time is None:
            self.start_time = server_time

        elapsed = server_time - self.start_time
        if elapsed < self.FORWARD_SECONDS:
            self.agent.skills_manager.execute_sub_skill(
                "Walk",
                reset=reset,
                target_2d=np.array([0.50, -0.04]),
                is_target_absolute=False,
                orientation=0.0,
                is_orientation_absolute=False,
            )
            return False

        if elapsed < self.FORWARD_SECONDS + self.STABILIZE_SECONDS:
            self.agent.skills_manager.execute_sub_skill(
                "Walk",
                reset=False,
                target_2d=np.zeros(2),
                is_target_absolute=False,
                orientation=0.0,
                is_orientation_absolute=False,
            )
            return False

        return True

    def is_ready(self, *args, **kwargs) -> bool:
        return not self.agent.world.is_fallen()
