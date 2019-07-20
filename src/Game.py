import random

from pos import Pos
from snake import Snake
from tkinter import Canvas, Tk
from dir import Dir


class Game:
    SCALE = 20

    def __init__ (self):
        self.width = 20
        self.height = 20
        self.snake = Snake (self)
        self.apple = Pos.random (self)

    def draw (self, canvas: Canvas) -> None:
        canvas.delete ('all')
        canvas.create_rectangle (0, 0, self.width * self.SCALE, self.height * self.SCALE, fill = 'black')
        self.snake.draw (canvas)
        self.apple.draw (canvas, 'red')

    def step (self) -> None:
        self.snake.move (random.choice (Dir.ALL))
