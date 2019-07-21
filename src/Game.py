import random

from pos import Pos
from snake import Snake
from tkinter import Canvas, Label, StringVar
from dir import Dir
from brain import Brain


class Game:
    SCALE = 20

    def __init__ (self) -> None:
        self.width = 40
        self.height = 40
        self.brain = Brain (self)
        self.snake = Snake (self)
        self.apple = Pos.random (self)
        self.score = 0
        self.attempt = 0

    def draw (self, canvas: Canvas, text: StringVar) -> None:
        canvas.delete ('all')
        canvas.create_rectangle (0, 0, self.width * self.SCALE, self.height * self.SCALE, fill = 'black')
        self.snake.draw (canvas)
        self.apple.draw (canvas, 'red')

        text.set (f'Attempt: {self.attempt}\nScore: {self.score}')

    def step (self) -> int:
        if not self.snake.move (random.choice (Dir.ALL)):
            self.reset ()
            return -1
        if self.snake.head == self.apple:
            self.apple = Pos.random (self)
            return 1
        return 0

    def reset (self) -> None:
        self.snake = Snake (self)
        self.apple = Pos.random (self)
        self.score = 0
        self.attempt += 1
