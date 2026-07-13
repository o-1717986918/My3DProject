import os
from mujococodebase.skills.keyframe.keyframe import KeyframeSkill


class Neutral(KeyframeSkill):
    def __init__(self, agent):
        super().__init__(agent, os.path.join(os.path.dirname(__file__), "neutral.yaml"))
