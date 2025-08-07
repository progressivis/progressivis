# Online statistical operators
# Try to be consistent with river https://riverml.xyz/
# https://github.com/online-ml/river/tree/main/river/stats

import numpy as np
from typing import (
    Collection,
    Union,
    Any,
    Set,
    Type,
)

from abc import abstractmethod, ABC

from ..table.column_base import BasePColumn


Num = Union[float, int]
# Column = np.typing.ArrayLike  # Union[Collection[Num], BasePColumn]
Column = Union[Collection[Num], BasePColumn]


class Statistic(ABC):
    name = "statistic"

    @abstractmethod
    def reset(self) -> None:
        """Reset the online statistic"""

    @abstractmethod
    def get(self) -> float:
        """Return the current value of the statistic."""

    def clone(self) -> Any:
        return self.__class__()


class Univariate(Statistic):
    """A univariate statistic measures a property of a variable."""

    name = "Univariate"

    @abstractmethod
    def update_many(self, X: Column) -> None:
        """Update the called instance."""
        raise NotImplementedError

    def update(self, v: float) -> None:
        self.update_many([v])


aggr_registry: dict[str, Type[Univariate]] = {}


class Bivariate(Statistic):
    """A bivariate statistic measures a relationship between two variables."""

    name = "Bivariate"

    @abstractmethod
    def update_many(self, X: Column, Y: Column) -> None:
        """Update the called instance."""

    def update(self, x: float, y: float) -> None:
        self.update_many([x], [y])


class Count(Univariate):  # Keep this functor first!
    """
    """

    name = "count"

    def __init__(self) -> None:
        self.count: int
        self.reset()

    def reset(self) -> None:
        self.count = 0

    def update_many(self, X: Column) -> None:
        if X is not None:
            self.count += len(X)

    def get(self) -> float:
        return self.count


aggr_registry[Count.name] = Count


class NUnique(Univariate):
    """ """

    name = "nunique"

    def __init__(self) -> None:
        self._set: Set[Any] = set()

    def reset(self) -> None:
        self._set = set()

    def update_many(self, X: Column) -> None:
        if X is not None:
            self._set.update(set(X))

    def get(self) -> Any:
        return len(self._set)


aggr_registry[NUnique.name] = NUnique


class Sum(Univariate):
    """ """

    name = "sum"

    def __init__(self) -> None:
        self.sum: Num
        self.reset()

    def reset(self) -> None:
        self.sum = 0

    def update(self, v: Num) -> None:
        self.sum += v

    def update_many(self, X: Column) -> None:
        if X is not None:
            self.sum += np.sum(X)  # type: ignore

    def get(self) -> float:
        return self.sum


aggr_registry[Sum.name] = Sum


class Mean(Sum):
    """ """

    name = "mean"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.n: int = 0
        self.mean: float = 0.0

    def update(self, v: Num) -> None:
        super().update(v)
        self.n += 1
        self.mean = self.sum / self.n

    def update_many(self, X: Column) -> None:
        if X is not None:
            super().update_many(X)
            self.n += len(X)
            self.mean = self.sum / self.n

    def get(self) -> float:
        return self.mean


aggr_registry[Mean.name] = Mean


# Should translate that to Cython eventually
class Var(Univariate):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    name = "variance"

    def __init__(self, ddof: int = 1) -> None:
        self.ddof: int = ddof
        self.mean = Mean()
        self.M2: float = 0.0
        self.reset()

    def reset(self) -> None:
        self.mean.reset()
        self.M2 = 0

    def clone(self) -> Any:
        return self.__class__(self.ddof)

    def update_many(self, X: Column) -> None:
        if X is not None:
            old = self.mean.get()
            self.mean.update_many(X)
            new = self.mean.get()
            self.M2 += np.sum(
                np.multiply(np.subtract(X, old),  # type: ignore
                            np.subtract(X, new))  # type: ignore
            ).item()

    @property
    def n(self) -> int:
        return self.mean.n

    @property
    def variance(self) -> float:
        if self.n <= self.ddof:
            return 0.0
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self) -> float:
        return float(self.variance ** 0.5)

    def get(self) -> float:
        return self.variance


aggr_registry[Var.name] = Var


class Std(Var):
    name = "stddev"

    def get(self) -> float:
        return self.std


aggr_registry[Std.name] = Std


class Cov(Bivariate):
    def __init__(self, ddof: int = 1) -> None:
        self.ddof: int = ddof
        self.mean_x = Mean()
        self.mean_y = Mean()
        self.cov: float = 0

    @property
    def n(self) -> int:
        return self.mean_x.n

    def reset(self) -> None:
        self.mean_x.reset()
        self.mean_y.reset()
        self.cov = 0

    def clone(self) -> Any:
        return self.__class__(self.ddof)

    def update_many(self, X: Column, Y: Column) -> None:
        dx = np.subtract(X, self.mean_x.get())  # type: ignore
        self.mean_x.update_many(X)
        self.mean_y.update_many(Y)
        self.cov += np.sum(
            dx * np.subtract(Y, self.mean_y.get())  # type: ignore
            - self.cov
        ) / max(self.n - self.ddof, 1)

    def get(self) -> float:
        return self.cov


# aggr_registry[Covariance.name] = Covariance

class Corr(Bivariate):
    def __init__(self, ddof: int = 1) -> None:
        self.var_x = Var(ddof=ddof)
        self.var_y = Var(ddof=ddof)
        self.cov_xy = Cov(ddof=ddof)

    @property
    def ddof(self) -> int:
        return self.cov_xy.ddof

    def reset(self) -> None:
        self.var_x.reset()
        self.var_y.reset()
        self.cov_xy.reset()

    def update_many(self, X: Column, Y: Column) -> None:
        self.var_x.update_many(X)
        self.var_y.update_many(Y)
        self.cov_xy.update_many(X, Y)

    def get(self) -> float:
        var_x: float = self.var_x.get()
        var_y: float = self.var_y.get()
        if var_x and var_y:
            return self.cov_xy.get() / (var_x * var_y) ** 0.5  # type: ignore
        return 0.0
