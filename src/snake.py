import typing
from typing import List, Tuple
from tkinter import Canvas

from pos import Pos
from dir import Dir


class Snake:
    def __init__ (self, body: List[Pos]) -> None:
        self.body: List[Pos] = body

    @classmethod
    def random (cls) -> 'Snake':
        pos = Pos.random ()
        return Snake ([pos for i in range (6)])

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
            cell.draw (canvas, self.head, 'white')

    def move (self, dir: Dir) -> Tuple['Snake', bool]:
        next_head = self.body[0] + dir.offset
        is_alive = not next_head.is_outside and next_head not in self.body
        next_body = [next_head] + self.body[:-1]
        return Snake (next_body), is_alive

    def grow (self) -> None:
        self.body.append (self.body[-1])

    def __str__ (self) -> str:
        return str (self.body)
    def __repr__ (self) -> str:
        return str (self)
