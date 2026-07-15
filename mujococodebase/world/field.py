from abc import ABC, abstractmethod
from typing import override
from mujococodebase.world.field_landmarks import FieldLandmarks


class Field(ABC):
    def __init__(self, world):
        from mujococodebase.world.world import World  # type hinting
        self.world: World = world
        self.field_landmarks: FieldLandmarks = FieldLandmarks(world=self.world)
    
    def get_our_goal_position(self):
        if self.world.is_left_team is False:
            return (self.get_length()/2, 0)
        return (-self.get_length()/2, 0)

    def get_their_goal_position(self):
        if self.world.is_left_team is False:
            return (-self.get_length()/2, 0)
        return (self.get_length()/2, 0)
    
    @abstractmethod
    def get_width(self):
        raise NotImplementedError()

    @abstractmethod
    def get_length(self):
        raise NotImplementedError()
    

class FIFAField(Field):
    def __init__(self, world):
        super().__init__(world)
    
    @override
    def get_width(self):
        return 68
    
    @override
    def get_length(self):
        return 105
    

class HLAdultField(Field):
    def __init__(self, world):
        super().__init__(world)
    
    @override
    def get_width(self):
        return 9
    
    @override
    def get_length(self):
        return 14



class MyField(Field):
    def __init__(self, world):
        super().__init__(world)
    
    @override
    def get_width(self):
        return 36
    
    @override
    def get_length(self):
        return 55