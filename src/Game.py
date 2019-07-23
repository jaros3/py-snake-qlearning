import numpy as np
from numpy import ndarray
from typing import Tuple
from tkinter import Canvas, Label, StringVar

from pos import Pos
from snake import Snake
from dir import Dir
from brain import Brain


class Game:
    WIDTH = 40
    HEIGHT = 40
    SCALE = 20
    SIGHT_CHANNELS = 2
    SIGHT_RADIUS = 20
    SIGHT_DIAMETER = SIGHT_RADIUS * 2 + 1
    CHANNEL_OBSTACLE = 0
    CHANNEL_APPLE = 1

    def __init__ (self) -> None:
        self.score = 0
        self.attempt = 0
        self.age = 0
        self.apple = Pos.random (self)
        self.snake = Snake (self)
        self.brain = Brain (self)

    def draw (self, canvas: Canvas) -> None:
        canvas.delete ('all')
        canvas.create_rectangle (0, 0, self.WIDTH * self.SCALE, self.HEIGHT * self.SCALE, fill = 'black')
        self.snake.draw (canvas)
        self.brain.draw (canvas)
        self.apple.draw (canvas, 'red')
        self.draw_walls (canvas)

    def draw_walls (self, canvas: Canvas) -> None:
        for x in range (-1, self.WIDTH + 1):
            Pos (self, x, -1).draw (canvas, 'white')
            Pos (self, x, self.HEIGHT).draw (canvas, 'white')
        for y in range (-1, self.HEIGHT + 1):
            Pos (self, -1, y).draw (canvas, 'white')
            Pos (self, self.WIDTH, y).draw (canvas, 'white')

    def set_text (self, text: StringVar) -> None:
        text.set (f'Attempt: {self.attempt}\nScore: {self.score}\nAge: {self.age}')

    def step_and_learn (self) -> None:
        last_sight = self.sight ()
        action_index = self.brain.think ()
        reward, is_alive = self.step (action_index)
        self.brain.learn (reward, is_alive, last_sight, action_index)

    def step (self, action_index: int) -> Tuple[int, bool]:
        action = Dir.ALL[action_index]
        if not self.snake.move (action):
            print ('Death!')
            self.reset ()
            return -1, False
        if self.snake.head == self.apple:
            print ('Apple!')
            self.apple = Pos.random (self)
            self.score += 1
            return 1, True
        return 0, True

    def reset (self) -> None:
        self.snake = Snake (self)
        self.apple = Pos.random (self)
        self.score = 0
        self.age = 0
        self.attempt += 1
        self.brain.on_death ()

    def sight (self) -> ndarray:
        sight = np.zeros ((1, self.SIGHT_CHANNELS, self.SIGHT_DIAMETER, self.SIGHT_DIAMETER))

        for cell in self.snake.body:
            self.set_pixel (sight, cell, self.CHANNEL_OBSTACLE)
        self.set_pixel (sight, self.apple, self.CHANNEL_APPLE)

        for x in range (-1, self.WIDTH + 1):
            self.set_pixel (sight, Pos (self, x, -1), self.CHANNEL_OBSTACLE)
            self.set_pixel (sight, Pos (self, x, self.HEIGHT), self.CHANNEL_OBSTACLE)
        for y in range (-1, self.HEIGHT + 1):
            self.set_pixel (sight, Pos (self, -1, y), self.CHANNEL_OBSTACLE)
            self.set_pixel (sight, Pos (self, self.WIDTH, y), self.CHANNEL_OBSTACLE)
        return sight

    def set_pixel (self, sight: ndarray, cell: Pos, layer: int) -> None:
        relative = cell - self.snake.head + Pos (self, self.SIGHT_RADIUS, self.SIGHT_RADIUS)
        if relative.is_in (0, 0, 2 * self.SIGHT_RADIUS, 2 * self.SIGHT_RADIUS):
            sight[0, layer, relative.y, relative.x] = 1
