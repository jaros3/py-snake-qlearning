from abc import ABC, abstractmethod
from typing import List, NamedTuple
import math


class WidthHeight (NamedTuple):
    width: int
    height: int


class Resizer (ABC):
    def __init__ (self, name: str):
        self.name = name

    @abstractmethod
    def forward (self, side: int) -> int:
        pass

    @abstractmethod
    def backward (self, side: int) -> int:
        pass


class Padding (Resizer):
    def __init__ (self, name: str, amount: int):
        super ().__init__ (name + '_padding')
        self.amount = 2 * amount

    def forward (self, side: int) -> int:
        return side + self.amount

    def backward (self, side: int) -> int:
        return side - self.amount

    @staticmethod
    def make (name: str, kind: str, filterSide: int) -> 'Padding':
        if kind == 'same':
            return Padding (name, 0)
        elif kind == 'valid':
            if filterSide % 2 != 1:
                raise Exception (f'filterSide {filterSide} must be odd for same padding ({name})')
            return Padding (name, -(filterSide - 1) // 2)
        else:
            raise Exception (f'invalid kind: {kind}')


class ConvResizer (Resizer):
    def __init__ (self, name: str, filterSide: int = 3, stride: int = 1, padding: str = 'same'):
        super ().__init__ (name)
        self.filterSide = filterSide
        self.stride = stride
        self.padding = Padding.make (name, padding, filterSide)

    def forward (self, side: int) -> int:
        paddedSide = self.padding.forward (side)
        paddedSide -= 1
        if paddedSide % self.stride != 0:
            raise Exception (
                f'Cannot fit integer number of strides: {side} -> {paddedSide} % {self.stride} at {self.name}')
        return paddedSide // self.stride + 1

    def backward (self, side: int) -> int:
        side = (side - 1) * self.stride + 1
        side = self.padding.backward (side)
        return side


class PoolResizer (ConvResizer):
    def __init__ (self, name: str, filterSide: int = 3, stride: int = 2, padding: str = 'valid'):
        super ().__init__ (name, filterSide, stride, padding)


class Resizers (Resizer):
    def __init__ (self, name: str, resizers: List[Resizer]):
        super ().__init__ (name)
        self.resizers = resizers

    def forward (self, side: int) -> int:
        for resizer in self.resizers:
            before = side
            after = resizer.forward (before)
            print (f'{resizer.name}: {before} -> {after}')

            side = resizer.forward (side)
        return side

    def backward (self, side: int) -> int:
        for resizer in reversed (self.resizers):
            after = side
            before = resizer.backward (after)
            print (f'{resizer.name}: {before} -> {after}')
            side = before
        return side

    def find (self, optimal: int, chooseLarger: bool) -> int:
        lower = 0
        upper = optimal
        while upper - lower > 1:
            middle = (lower + upper) // 2
            fromMiddle = self.backward (middle)
            if fromMiddle < optimal:
                lower = middle
            elif fromMiddle > optimal:
                upper = middle
            else:
                return fromMiddle
        fromLower = self.backward (lower)
        fromUpper = self.backward (upper)
        if fromLower == optimal:
            return fromLower
        elif fromUpper == optimal:
            return fromUpper
        return fromUpper if chooseLarger else fromLower

    def findSize (self, size: WidthHeight, chooseLarger: bool) -> WidthHeight:
        return WidthHeight (self.find (size.width, chooseLarger),
                            self.find (size.height, chooseLarger))
