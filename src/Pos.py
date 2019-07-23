import random
import typing
from tkinter import Canvas

from const import *


class Pos:
    def __init__ (self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y

    @classmethod
    def random (cls) -> 'Pos':
        return Pos (x = random.randrange (WIDTH - 1), y = random.randrange (HEIGHT - 1))

    def __add__ (self, other: 'Pos') -> 'Pos':
        return Pos (self.x + other.x, self.y + other.y)
    def __sub__ (self, other: 'Pos') -> 'Pos':
        return Pos (self.x - other.x, self.y - other.y)

    def draw (self, canvas: Canvas, head: 'Pos', color: str) -> None:
        x = (self.x - head.x + SIGHT_RADIUS) * SCALE
        y = (self.y - head.y + SIGHT_RADIUS) * SCALE
        canvas.create_rectangle (x, y, x + SCALE, y + SCALE, fill = color)

    @property
    def is_outside (self) -> bool:
        return not self.is_in (0, 0, WIDTH - 1, HEIGHT - 1)

    def is_in (self, min_x: int, min_y: int, max_x: int, max_y: int) -> bool:
        return min_x <= self.x <= max_x and min_y <= self.y <= max_y

    def __eq__ (self, other) -> bool:
        return isinstance (other, Pos) and self.x == other.x and self.y == other.y
    def __ne__ (self, other) -> bool:
        return not (self == other)

    def __str__ (self) -> str:
        return f'({self.x}, {self.y})'
    def __repr__ (self) -> str:
        return str (self)
