from collections import defaultdict
from typing import Dict, Tuple
from tkinter import Canvas, StringVar
import statistics

from pos import Pos
from snake import Snake
from board import Board
from brain import Brain
from memories import Memory, Memories
from display import Displays


class Game:
    LEARN_PERIOD = 5

    def __init__ (self) -> None:
        self.score = 0
        self.attempt = 0
        self.age = 0
        self.seconds = 0
        self.current_step = 0
        self.attempt_ages: Dict[int, int] = defaultdict (int)
        self.attempt_turns: Dict[int, int] = defaultdict (int)
        self.attempt_apples: Dict[int, int] = defaultdict (int)
        self.max_apples = 0
        self.board = Board (Board.random_apples (), Snake.random ())
        self.memories = Memories ()
        self.displays: Displays = None
        self.brain = Brain (self)

    def draw (self, canvas: Canvas) -> None:
        self.board.draw (canvas)
        self.brain.draw (canvas, self.board.snake.head)

    def set_text (self, text: StringVar) -> None:
        hours = self.seconds // 3600
        minutes = self.seconds // 60 % 60
        seconds = self.seconds % 60
        avg_ages = statistics.mean (self.attempt_ages.values ()) if self.attempt_ages else 0
        avg_turns = statistics.mean (self.attempt_turns.values ()) if self.attempt_turns else 0
        avg_apples = statistics.mean (self.attempt_apples.values ()) if self.attempt_apples else 0
        text.set (f'Attempt: {self.attempt} | Time: {hours:02d}:{minutes:02d}:{seconds:02d}\n' +
                  f'Apples: {self.score} | Avg apples: {avg_apples:.2f} | Max apples: {self.max_apples}\n' +
                  f'Age: {self.age} | Avg age: {avg_ages:.2f} | Avg turns: {avg_turns:.2f}\n' +
                  f'Step: {self.current_step}')

    def step_and_learn (self) -> None:
        action_index, turned = self.brain.think ()
        if turned:
            self.attempt_turns[self.attempt] += 1
        reward, is_alive, next_board = self.step (action_index)
        self.memories.items.append (Memory (len (self.memories.items), self.board, action_index, reward, is_alive, next_board))
        self.current_step += 1
        if self.current_step % self.LEARN_PERIOD == 0:
            self.brain.learn (self.memories)
        self.board = next_board

    def step (self, action_index: int) -> Tuple[int, bool, Board]:
        self.age += 1
        self.attempt_ages[self.attempt] += 1
        next_board, reward, is_alive = self.board.step (action_index, real_snake = True)
        if reward > 0:
            self.score += 1
            self.attempt_apples[self.attempt] += 1
            if self.score > self.max_apples:
                self.max_apples = self.score
        elif not is_alive:
            self.reset ()
        return reward, is_alive, next_board

    def reset (self) -> None:
        self.score = 0
        self.age = 0
        self.attempt += 1
        self.attempt_ages[self.attempt] = 0
        self.attempt_turns[self.attempt] = 0
        self.attempt_apples[self.attempt] = 0
        self.brain.on_death ()
