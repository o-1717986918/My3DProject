import os
from mujococodebase.skills.keyframe.keyframe import KeyframeSkill


class KickRight(KeyframeSkill):
    """右脚踢球——先蓄力后摆，再前摆击球。"""

    def __init__(self, agent):
        super().__init__(agent, os.path.join(os.path.dirname(__file__), "kick_right.yaml"))
