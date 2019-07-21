import typing
from typing import List

from pos import Pos


if typing.TYPE_CHECKING:
    from .game import Game


class Dir:
    ALL: List['Dir'] = []

    def __init__ (self, offset: Pos) -> None:
        self.offset: Pos = offset

    @classmethod
    def static_init (cls, game: 'Game') -> None:
        cls.UP: Dir = Dir (Pos (game, 0, -1))
        cls.RIGHT: Dir = Dir (Pos (game, 1, 0))
        cls.DOWN: Dir = Dir (Pos (game, 0, 1))
        cls.LEFT: Dir = Dir (Pos (game, -1, 0))
        cls.ALL: List[Dir] = [cls.UP, cls.RIGHT, cls.DOWN, cls.LEFT]