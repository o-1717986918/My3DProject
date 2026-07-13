from mujococodebase.skills.keyframe.get_up.get_up import GetUp
from mujococodebase.skills.keyframe.keyframe import KeyframeSkill
from mujococodebase.skills.keyframe.poses.neutral.neutral import Neutral
from mujococodebase.skills.skill import Skill
from mujococodebase.skills.walk.walk import Walk


class SkillsManager:

    def __init__(self, agent) -> None:
        self.current_skill_name: str | None = None
        self.previous_skill_name: str | None = None
        self.current_sub_skill_name: str | None = None
        self.skills: dict[str, Skill] = {}
        self.agent = agent

        self.create_skills()

    def create_skills(self):
        """
        Loads all Skill classes and instantiates them.
        Each skill is indexed by its class name.
        """

        classes: list[type[Skill]] = [Walk, Neutral, GetUp]

        # instantiate each Skill and store in the skills dictionary
        self.skills = {cls.__name__: cls(agent=self.agent) for cls in classes}

    def get_skill_object(self, name: str) -> Skill:
        """Returns the Skill instance corresponding to the given name."""
        if name not in self.skills:
            raise KeyError(f"No skill found with the name '{name}'")
        return self.skills[name]

    def execute(self, skill_name: str, *args, **kwargs) -> bool:
        """
        Executes one step of the specified skill.

        - On the first call for a skill, `reset=True` is sent.
        - Returns True when the skill has finished execution.
        """
        skill = self.get_skill_object(skill_name)

        # detect if the skill has changed to trigger automatic reset
        reset = self.current_skill_name != skill_name
        if reset:
            # previous skill was interrupted
            if self.current_skill_name is not None:
                self.previous_skill_name = self.current_skill_name
            self.current_skill_name = skill_name

        # call the Skill's own execute method
        finished = skill.execute(reset, *args, **kwargs)
        if not finished:
            return False

        # skill finished execution
        self.previous_skill_name = self.current_skill_name
        self.current_skill_name = None
        return True

    def execute_sub_skill(self, skill_name: str, reset: bool, *args, **kwargs) -> bool:
        """
        Executes a step of a sub-skill within another skill.

        - Does not change `current_skill_name`.
        - `current_sub_skill_name` reflects the sub-skill currently being executed.
        """
        skill = self.get_skill_object(skill_name)
        self.current_sub_skill_name = skill_name
        return skill.execute(reset, *args, **kwargs)

    def is_ready(self, skill_name: str, *args, **kwargs) -> bool:
        """
        Checks if the specified skill is ready to start based on current conditions.
        """
        skill = self.get_skill_object(skill_name)
        return skill.is_ready(*args, **kwargs)
