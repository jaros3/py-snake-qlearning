import random
import typing
from tkinter import Canvas

if typing.TYPE_CHECKING:
    from .game import Game


class Pos:
    def __init__ (self, game: 'Game', x: int, y: int) -> None:
        self.game: 'Game' = game
        self.x: int = x
        self.y: int = y

    @classmethod
    def random (cls, game: 'Game') -> 'Pos':
        return Pos (game, x = random.randrange (game.width - 1), y = random.randrange (game.height - 1))

    def __add__ (self, other: 'Pos') -> 'Pos':
        return Pos (self.game, self.x + other.x, self.y + other.y)

    def draw (self, canvas: Canvas, color: str) -> None:
        x = self.x * self.game.SCALE
        y = self.y * self.game.SCALE
        canvas.create_rectangle (x, y, x + self.game.SCALE, y + self.game.SCALE, fill = color)

    @property
    def is_outside (self) -> bool:
        return self.x < 0 or self.y < 0 or self.x >= self.game.width or self.y >= self.game.height

    def __eq__ (self, other) -> bool:
        return isinstance (other, Pos) and self.x == other.x and self.y == other.y
    def __ne__ (self, other) -> bool:
        return not (self == other)