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
        return Pos (game, x = random.randrange (game.width), y = random.randrange (game.height))

    def __add__ (self, other: 'Pos') -> 'Pos':
        return Pos (self.game, self.x + other.x, self.y + other.y)

    def draw (self, canvas: Canvas, color: str) -> None:
        x = self.x * self.game.SCALE
        y = self.y * self.game.SCALE
        canvas.create_rectangle (x, y, x + self.game.SCALE, y + self.game.SCALE, fill = color)
