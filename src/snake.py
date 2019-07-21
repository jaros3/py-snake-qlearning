import typing
from typing import Deque
from collections import deque
from tkinter import Canvas

from pos import Pos
from dir import Dir

if typing.TYPE_CHECKING:
    from .game import Game


class Snake:
    def __init__ (self, game: 'Game') -> None:
        self.game: 'Game' = game
        pos = Pos.random (game)
        self.body: Deque[Pos] = deque ([pos for i in range (6)])

    @property
    def length (self) -> int:
        return len (self.body)

    @property
    def head (self) -> Pos:
        return self.body[0]
    @property
    def tail (self) -> Pos:
        return self.body[-1]

    def draw (self, canvas: Canvas) -> None:
        for cell in self.body:
            cell.draw (canvas, 'white')

    def move (self, dir: Dir) -> bool:
        head = self.body[0] + dir.offset
        self.body.pop ()
        isAlive = True
        if head.is_outside or head in self.body:
            isAlive = False
        self.body.appendleft (head)
        return isAlive
