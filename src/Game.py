from collections import defaultdict
from typing import Dict, Tuple
from tkinter import Canvas, StringVar
import statistics

from pos import Pos
from snake import Snake
from board import Board
from brain import Brain
from memories import Memory, Memories


class Game:
    LEARN_PERIOD = 1

    def __init__ (self) -> None:
        self.score = 0
        self.attempt = 0
        self.age = 0
        self.seconds = 0
        self.current_step = 0
        self.lifetime: Dict[int, int] = defaultdict (int)
        self.turns: Dict[int, int] = defaultdict (int)
        self.board = Board (Pos.random (), Snake.random ())
        self.memories = Memories ()
        self.brain = Brain (self)

    def draw (self, canvas: Canvas) -> None:
        self.board.draw (canvas)
        self.brain.draw (canvas, self.board.snake.head)

    def set_text (self, text: StringVar) -> None:
        hours = self.seconds // 3600
        minutes = self.seconds // 60 % 60
        seconds = self.seconds % 60
        lifetime = statistics.mean (self.lifetime.values ()) if self.lifetime else 0
        turns = statistics.mean (self.turns.values ()) if self.turns else 0
        text.set (f'Attempt: {self.attempt} | Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n' +
                  f'Apples: {self.score}\nAge: {self.age} | Lifetime: {lifetime:.2f} | Turns: {turns:.2f}\n' +
                  f'Step: {self.current_step}')

    def step_and_learn (self) -> None:
        action_index, turned = self.brain.think ()
        if turned:
            self.turns[self.attempt] += 1
        reward, is_alive, next_board = self.step (action_index)
        self.memories.items.append (Memory (len (self.memories.items), self.board, action_index, reward, is_alive, next_board))
        self.current_step += 1
        if self.current_step % self.LEARN_PERIOD == 0:
            self.brain.learn (self.memories)
        self.board = next_board

    def step (self, action_index: int) -> Tuple[int, bool, Board]:
        self.age += 1
        self.lifetime[self.attempt] += 1
        next_board, reward, is_alive = self.board.step (action_index, real_snake = True)
        if reward > 0:
            self.score += 1
        elif not is_alive:
            self.reset ()
        return reward, is_alive, next_board

    def reset (self) -> None:
        self.score = 0
        self.age = 0
        self.attempt += 1
        self.lifetime[self.attempt] = 0
        self.turns[self.attempt] = 0
        self.brain.on_death ()
