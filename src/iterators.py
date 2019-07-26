from typing import Iterable, TypeVar

T = TypeVar ('T')


def first (collection: Iterable[T]) -> T:
    return next (iter (collection))
