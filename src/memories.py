from typing import List

from board import Board


class Memory:
    def __init__ (self, prev_board: Board, action_index: int, reward: int, is_alive: bool, next_board: Board) -> None:
        self.prev_board: Board = prev_board
        self.action_index: int = action_index
        self.reward: int = reward
        self.is_alive: bool = is_alive
        self.next_board: Board = next_board


class Memories:
    def __init__ (self) -> None:
        self.items: List[Memory] = []
