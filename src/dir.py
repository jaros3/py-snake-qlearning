import typing
from typing import List
import random

from pos import Pos


class Dir:
    ALL: List['Dir'] = []

    def __init__ (self, offset: Pos, arrow: str) -> None:
        self.offset: Pos = offset
        self.arrow = arrow

    @classmethod
    def static_init (cls) -> None:
        cls.UP: Dir = Dir (Pos (0, -1), '↑')
        cls.RIGHT: Dir = Dir (Pos (1, 0), '→')
        cls.DOWN: Dir = Dir (Pos (0, 1), '↓')
        cls.LEFT: Dir = Dir (Pos (-1, 0), '←')
        cls.ALL: List[Dir] = [cls.UP, cls.RIGHT, cls.DOWN, cls.LEFT]

    @classmethod
    def random (cls) -> 'Dir':
        return random.choice (cls.ALL)

    def is_opposite (self, other: 'Dir') -> bool:
        return self.offset + other.offset == Pos (0, 0)
