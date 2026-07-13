import numpy as np
from mujococodebase.utils.math_ops import MathOps

class FieldLandmarks:
    def __init__(self, world):
        from mujococodebase.world.world import World  # type hinting
        
        self.world: World = world
        
        self.landmarks: dict = {}
    
    def update_from_perception(self, landmark_id: str, landmark_pos: np.ndarray) -> None:
        """
        Calculates the global position of all currently visible landmarks.
        """
        
        world = self.world
        local_cart_3d = MathOps.deg_sph2cart(landmark_pos)
        
        
        global_pos_3d = MathOps.rel_to_global_3d(
            local_pos_3d=local_cart_3d,
            global_pos_3d=world.global_position,
            global_orientation_quat=world.agent.robot.global_orientation_quat
        )
        
        self.landmarks[landmark_id] = global_pos_3d

    def get_landmark_position(self, landmark_id: str) -> np.ndarray | None:
        """
        Returns the calculated 2d global position for a given landmark ID.
        Returns None if the landmark is not currently visible or processed.
        """
        return self.global_positions.get(landmark_id)