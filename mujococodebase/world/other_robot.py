import numpy as np

class OtherRobot:
    def __init__(self, is_teammate: bool=True):
        self.is_teammate = is_teammate
        self.position = np.zeros(3)
        self.last_seen_time = None