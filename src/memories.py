from typing import List

from dir import Dir
from board import Board


class Memory:
    def __init__ (self, prev_board: Board, action_index: int, reward: int, is_alive: bool, next_board: Board) -> None:
        self.prev_board: Board = prev_board
        self.action_index: int = action_index
        self.reward: int = reward
        self.is_alive: bool = is_alive
        self.next_board: Board = next_board

    def __str__ (self) -> str:
        return f'{self.prev_board.snake.head} {Dir.ALL[self.action_index]} {self.next_board.snake.head} {self.reward:+} {self.is_alive}'
    def __repr__ (self) -> str:
        return str (self)


class Memories:
    def __init__ (self) -> None:
        self.items: List[Memory] = []

    def __str__ (self) -> str:
        return str (self.items)
    def __repr__ (self) -> str:
        return str (self)
