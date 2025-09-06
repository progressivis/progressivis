# Management of quality indicators in ProgressiVis
import abc
from typing import Any

import numpy as np


class QualityLiteral(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def quality(self, val: float) -> float:
        pass


class QualityL1(QualityLiteral):
    def __init__(self) -> None:
        self.previous: float | None = None

    def quality(self, val: float) -> float:
        try:
            ret = -abs(self.previous - val)  # type: ignore
        except Exception:
            ret = 0
        self.previous = val
        return ret


class QualitySqrtSumSquarredDiffs:
    def __init__(self) -> None:
        self.previous: np.ndarray[Any, Any] | None = None

    def quality(self, val: np.ndarray[Any, Any]) -> float:
        try:
            ret = -np.sqrt(np.sum((self.previous - val) ** 2))
        except Exception:
            ret = 0
        self.previous = val
        return float(ret)
