from enum import Enum, auto


class PlayModeEnum(Enum):
    NOT_INITIALIZED = auto()

    """Enum specifying possible play modes."""

    BEFORE_KICK_OFF = auto()
    """The game hasn't started yet."""

    OUR_KICK_OFF = auto()
    """Our team has kick off."""

    THEIR_KICK_OFF = auto()
    """Their team has kick off."""

    PLAY_ON = auto()
    """The game is running normal."""

    OUR_THROW_IN = auto()
    """The ball left the field and our team has throw in."""

    THEIR_THROW_IN = auto()
    """The ball left the field and their team has throw in."""

    OUR_CORNER_KICK = auto()
    """Our team has corner kick."""

    THEIR_CORNER_KICK = auto()
    """Their team has corner kick."""

    OUR_GOAL_KICK = auto()
    """Our team has goal kick."""

    THEIR_GOAL_KICK = auto()
    """Their team has goal kick."""

    OUR_OFFSIDE = auto()
    """Their team violated the offside rule."""

    THEIR_OFFSIDE = auto()
    """Our team violated the offside rule."""

    GAME_OVER = auto()
    """The game has ended."""

    OUR_GOAL = auto()
    """Our team scored a goal."""

    THEIR_GOAL = auto()
    """Their team scored a goal."""

    OUR_FREE_KICK = auto()
    """Our team has a free kick."""

    THEIR_FREE_KICK = auto()
    """Their team has a free kick."""

    OUR_DIRECT_FREE_KICK = auto()
    """Our team has a direct free kick."""

    THEIR_DIRECT_FREE_KICK = auto()
    """Their team has a direct free kick."""

    OUR_PENALTY_KICK = auto()
    """Our team has a penalty kick (from the penalty spot)."""

    THEIR_PENALTY_KICK = auto()
    """Their team has a penalty kick (from the penalty spot)."""

    OUR_PENALTY_SHOOT = auto()
    """Our team has a penalty shoot (starting from somewhere on the field, allowed to touch the ball more than once)."""

    THEIR_PENALTY_SHOOT = auto()
    """Their team has a penalty shoot (starting from somewhere on the field, allowed to touch the ball more than once)."""

    @classmethod
    def get_playmode_from_string(
        cls, playmode: str, is_left_team: bool
    ) -> "PlayModeEnum":
        assert isinstance(is_left_team, bool)

        playmode_ids = {
            "BeforeKickOff": (PlayModeEnum.BEFORE_KICK_OFF,),
            "KickOff_Left": (
                PlayModeEnum.OUR_KICK_OFF,
                PlayModeEnum.THEIR_KICK_OFF,
            ),
            "KickOff_Right": (
                PlayModeEnum.THEIR_KICK_OFF,
                PlayModeEnum.OUR_KICK_OFF,
            ),
            "PlayOn": (PlayModeEnum.PLAY_ON,),
            "KickIn_Left": (
                PlayModeEnum.OUR_THROW_IN,
                PlayModeEnum.THEIR_THROW_IN,
            ),
            "KickIn_Right": (
                PlayModeEnum.THEIR_THROW_IN,
                PlayModeEnum.OUR_THROW_IN,
            ),
            "corner_kick_left": (
                PlayModeEnum.OUR_CORNER_KICK,
                PlayModeEnum.THEIR_CORNER_KICK,
            ),
            "corner_kick_right": (
                PlayModeEnum.THEIR_CORNER_KICK,
                PlayModeEnum.OUR_CORNER_KICK,
            ),
            "goal_kick_left": (
                PlayModeEnum.OUR_GOAL_KICK,
                PlayModeEnum.THEIR_GOAL_KICK,
            ),
            "goal_kick_right": (
                PlayModeEnum.THEIR_GOAL_KICK,
                PlayModeEnum.OUR_GOAL_KICK,
            ),
            "offside_left": (
                PlayModeEnum.OUR_OFFSIDE,
                PlayModeEnum.THEIR_OFFSIDE,
            ),
            "offside_right": (
                PlayModeEnum.THEIR_OFFSIDE,
                PlayModeEnum.OUR_OFFSIDE,
            ),
            "GameOver": (PlayModeEnum.GAME_OVER,),
            "Goal_Left": (
                PlayModeEnum.OUR_GOAL,
                PlayModeEnum.THEIR_GOAL,
            ),
            "Goal_Right": (
                PlayModeEnum.THEIR_GOAL,
                PlayModeEnum.OUR_GOAL,
            ),
            "free_kick_left": (
                PlayModeEnum.OUR_FREE_KICK,
                PlayModeEnum.THEIR_FREE_KICK,
            ),
            "free_kick_right": (
                PlayModeEnum.THEIR_FREE_KICK,
                PlayModeEnum.OUR_FREE_KICK,
            ),
            "direct_free_kick_left": (
                PlayModeEnum.OUR_DIRECT_FREE_KICK,
                PlayModeEnum.THEIR_DIRECT_FREE_KICK,
            ),
            "direct_free_kick_right": (
                PlayModeEnum.THEIR_DIRECT_FREE_KICK,
                PlayModeEnum.OUR_DIRECT_FREE_KICK,
            ),
            "penalty_kick_left": (
                PlayModeEnum.OUR_PENALTY_KICK,
                PlayModeEnum.THEIR_PENALTY_KICK,
            ),
            "penalty_kick_right": (
                PlayModeEnum.THEIR_PENALTY_KICK,
                PlayModeEnum.OUR_PENALTY_KICK,
            ),
            "penalty_shoot_left": (
                PlayModeEnum.OUR_PENALTY_SHOOT,
                PlayModeEnum.THEIR_PENALTY_SHOOT,
            ),
            "penalty_shoot_right": (
                PlayModeEnum.THEIR_PENALTY_SHOOT,
                PlayModeEnum.OUR_PENALTY_SHOOT,
            ),
        }[playmode]

        playmode = None
        if len(playmode_ids) > 1:
            playmode = playmode_ids[0 if is_left_team else 1]
        else:
            playmode = playmode_ids[0]

        return playmode


class PlayModeGroupEnum(Enum):
    NOT_INITIALIZED = auto()
    OTHER = auto()
    OUR_KICK = auto()
    THEIR_KICK = auto()
    ACTIVE_BEAM = auto()
    PASSIVE_BEAM = auto()

    @classmethod
    def get_group_from_playmode(
        cls, playmode: PlayModeEnum, is_left_team: bool
    ) -> "PlayModeGroupEnum":
        playmode_group: PlayModeGroupEnum = None

        if playmode in (PlayModeEnum.PLAY_ON, PlayModeEnum.GAME_OVER):
            playmode_group = cls.OTHER
        elif playmode in (
            PlayModeEnum.OUR_CORNER_KICK,
            PlayModeEnum.OUR_DIRECT_FREE_KICK,
            PlayModeEnum.OUR_FREE_KICK,
            PlayModeEnum.OUR_GOAL_KICK,
            PlayModeEnum.OUR_KICK_OFF,
            PlayModeEnum.OUR_OFFSIDE,
            PlayModeEnum.OUR_PENALTY_KICK,
            PlayModeEnum.OUR_PENALTY_SHOOT,
            PlayModeEnum.OUR_THROW_IN,
        ):
            playmode_group = cls.OUR_KICK
        elif playmode in (
            PlayModeEnum.THEIR_CORNER_KICK,
            PlayModeEnum.THEIR_DIRECT_FREE_KICK,
            PlayModeEnum.THEIR_FREE_KICK,
            PlayModeEnum.THEIR_GOAL_KICK,
            PlayModeEnum.THEIR_KICK_OFF,
            PlayModeEnum.THEIR_OFFSIDE,
            PlayModeEnum.THEIR_PENALTY_KICK,
            PlayModeEnum.THEIR_PENALTY_SHOOT,
            PlayModeEnum.THEIR_THROW_IN,
        ):
            playmode_group = cls.THEIR_KICK
        elif (playmode is PlayModeEnum.THEIR_GOAL) or (
            is_left_team and playmode is PlayModeEnum.BEFORE_KICK_OFF
        ):
            playmode_group = cls.ACTIVE_BEAM
        elif (playmode is PlayModeEnum.OUR_GOAL) or (
            not is_left_team and playmode is PlayModeEnum.BEFORE_KICK_OFF
        ):
            playmode_group = cls.PASSIVE_BEAM
        else:
            raise NotImplementedError(
                f"Not implemented playmode group for playmode {playmode}"
            )

        return playmode_group
