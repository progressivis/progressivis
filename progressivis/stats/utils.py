import numpy as np
from typing import Iterable


class OnlineMean:
    """
    Welford's algorithm
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.delta: float = 0

    def add(self, iterable: Iterable[float]) -> None:
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum: float) -> None:
        if np.isnan(datum):
            return
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
