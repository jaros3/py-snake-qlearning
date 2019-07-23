import typing
from typing import List, Tuple
from tkinter import Canvas
import numpy as np

from pos import Pos
from dir import Dir
from snake import Snake
from const import *


class Board:
    CHANNEL_OBSTACLE = 0
    CHANNEL_APPLE = 1

    def __init__ (self, apple: Pos, snake: Snake) -> None:
        self.apple = apple
        self.snake = snake

    def reset (self) -> None:
        self.apple = Pos.random ()
        self.snake = Snake.random ()

    def draw (self, canvas: Canvas) -> None:
        canvas.delete ('all')
        canvas.create_rectangle (0, 0, WIDTH * SCALE, HEIGHT * SCALE, fill = 'black')
        self.snake.draw (canvas)
        self.apple.draw (canvas, self.snake.head, 'red')
        self.draw_walls (canvas)

    def draw_walls (self, canvas: Canvas) -> None:
        head = self.snake.head
        for x in range (-1, WIDTH + 1):
            Pos (x, -1).draw (canvas, head, 'white')
            Pos (x, HEIGHT).draw (canvas, head, 'white')
        for y in range (-1, HEIGHT + 1):
            Pos (-1, y).draw (canvas, head, 'white')
            Pos (WIDTH, y).draw (canvas, head, 'white')

    def step (self, action_index: int) -> Tuple['Board', int, bool]:
        action = Dir.ALL[action_index]
        next_snake, is_alive = self.snake.move (action)
        next_board = Board (self.apple, next_snake)
        if not is_alive:
            print ('Death!')
            next_board.reset ()
            reward = -1
        elif self.snake.head == self.apple:
            print ('Apple!')
            next_board.apple = Pos.random ()
            next_snake.grow ()
            reward = 1
        else:
            reward = 0
        return next_board, reward, is_alive

    @classmethod
    def make_buffer (cls, batch: int) -> np.ndarray:
        return np.zeros ((batch, SIGHT_CHANNELS, SIGHT_DIAMETER, SIGHT_DIAMETER))

    def observe (self, buffer_slice: np.ndarray) -> None:
        for cell in self.snake.body:
            self.set_pixel (buffer_slice, cell, self.CHANNEL_OBSTACLE)
        self.set_pixel (buffer_slice, self.apple, self.CHANNEL_APPLE)

        for x in range (-1, WIDTH + 1):
            self.set_pixel (buffer_slice, Pos (x, -1), self.CHANNEL_OBSTACLE)
            self.set_pixel (buffer_slice, Pos (x, HEIGHT), self.CHANNEL_OBSTACLE)
        for y in range (-1, HEIGHT + 1):
            self.set_pixel (buffer_slice, Pos (-1, y), self.CHANNEL_OBSTACLE)
            self.set_pixel (buffer_slice, Pos (WIDTH, y), self.CHANNEL_OBSTACLE)

    def set_pixel (self, buffer_slice: np.ndarray, cell: Pos, channel: int) -> None:
        relative = cell - self.snake.head + Pos (SIGHT_RADIUS, SIGHT_RADIUS)
        if relative.is_in (0, 0, 2 * SIGHT_RADIUS, 2 * SIGHT_RADIUS):
            buffer_slice[0, channel, relative.y, relative.x] = 1
