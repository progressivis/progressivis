# Online statistical operators
# Try to be consistent with river https://riverml.xyz/
# https://github.com/online-ml/river/tree/main/river/stats

import numpy as np
import scipy as sp

from typing import (
    Collection,
    Dict,
    Union,
    Any,
    Set,
    Type,
    Tuple,
)

from abc import abstractmethod, ABC
import itertools

from ..table.column_base import BasePColumn

import pandas as pd

Num = Union[float, int]
# Column = np.typing.ArrayLike  # Union[Collection[Num], BasePColumn]
Column = Union[Collection[Num], BasePColumn, np.ndarray[Any, Any]]


class Statistic(ABC):
    name = "statistic"

    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    @abstractmethod
    def reset(self) -> None:
        """Reset the online statistic"""

    @abstractmethod
    def get(self) -> float:
        """Return the current value of the statistic."""

    def clone(self) -> Any:
        return self.__class__()

    def __repr__(self) -> str:
        try:
            value = self.get()
        except NotImplementedError:
            value = None
        fmt_value = None if value is None else f"{value:{self._fmt}}".rstrip("0")
        return f"{self.__class__.__name__}: {fmt_value}"

    def __str__(self) -> str:
        return repr(self)


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
        self.n: int = 0

    def reset(self) -> None:
        self.n = 0

    def update_many(self, col: Column) -> None:
        self.n += len(col)

    def get(self) -> float:
        return self.n


aggr_registry[Count.name] = Count


class NUnique(Univariate):
    """ """

    name = "nunique"

    def __init__(self) -> None:
        self._set: Set[Any] = set()

    def reset(self) -> None:
        self._set = set()

    @property
    def n(self) -> int:
        return len(self._set)

    def update_many(self, col: Column) -> None:
        self._set.update(set(col))

    def get(self) -> Any:
        return len(self._set)


aggr_registry[NUnique.name] = NUnique


class Mean(Univariate):
    """ """

    name = "mean"

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.n: int = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, v: Num) -> None:
        self.sum += v
        self.n += 1

    def update_many(self, col: Column) -> None:
        self.sum += np.sum(col)  # type: ignore
        self.n += len(col)

    @property
    def mean(self) -> float:
        return 0 if self.n == 0 else self.sum / self.n

    def get(self) -> float:
        return self.mean


aggr_registry[Mean.name] = Mean


class Sum(Mean):
    """ """

    name = "sum"

    def get(self) -> float:
        return self.sum


aggr_registry[Sum.name] = Sum


class Var(Univariate):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    name = "variance"

    def __init__(self, ddof: int = 0) -> None:
        self.ddof: int = ddof
        self.mean = Mean()
        self.M2: float = 0.0

    def reset(self) -> None:
        self.mean.reset()
        self.M2 = 0

    def clone(self) -> Any:
        return self.__class__(self.ddof)

    def update_many(self, col: Column) -> None:
        old = self.mean.get()
        self.mean.update_many(col)
        new = self.mean.get()
        self.M2 += np.sum(
            np.multiply(np.subtract(col, old),  # type: ignore
                        np.subtract(col, new))  # type: ignore
        ).item()

    @property
    def n(self) -> int:
        return self.mean.n

    @property
    def sum(self) -> float:
        return self.mean.sum

    @property
    def variance(self) -> float:
        if self.n <= self.ddof:
            return 0.0
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self) -> float:
        return float(self.variance ** 0.5)

    def get(self) -> float:
        if self.n <= self.ddof:
            return 0.0
        return self.variance

    def mean_ci_central_limit(self, alpha: float = 0.05) -> float:
        if self.n <= self.ddof:
            return 0.0
        z = sp.stats.t.ppf(1-alpha, self.n-1)
        return float(z * (self.variance / self.n) ** 0.5)


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

    def update_many(self, col_x: Column, col_y: Column) -> None:
        dx = np.subtract(col_x, self.mean_x.get())  # type: ignore
        self.mean_x.update_many(col_x)
        self.mean_y.update_many(col_y)
        self.cov += np.sum(
            dx * np.subtract(col_y, self.mean_y.get())  # type: ignore
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

    def update_many(self, col_x: Column, col_y: Column) -> None:
        self.var_x.update_many(col_x)
        self.var_y.update_many(col_y)
        self.cov_xy.update_many(col_x, col_y)

    def get(self) -> float:
        var_x: float = self.var_x.get()
        var_y: float = self.var_y.get()
        if var_x and var_y:
            return self.cov_xy.get() / (var_x * var_y) ** 0.5  # type: ignore
        return 0.0


class CovarianceMatrix:
    def __init__(self, ddof: int = 1) -> None:
        self.ddof = ddof
        self._cov: Dict[Tuple[Any, Any], Cov] = {}
        self._var: Dict[Any, Var] = {}

    def update_many(self, table: Dict[Any, Column]) -> None:
        for i, j in itertools.combinations(sorted(table), r=2):
            key = (i, j)
            try:
                cov = self._cov[key]
            except KeyError:
                cov = Cov(self.ddof)
                self._cov[key] = cov
            cov.update_many(table[i], table[j])

        for i, col_i in table.items():
            try:
                var = self._var[i]
            except KeyError:
                var = Var(self.ddof)
                self._var[i] = var
            var.update_many(col_i)

    def reset(self) -> None:
        for cov in self._cov.values():
            cov.reset()
        for var in self._var.values():
            var.reset()

    @property
    def cov(self) -> Dict[Tuple[Any, Any], float]:
        res: Dict[Tuple[Any, Any], float] = {}
        for key, cov in self._cov.items():
            res[key] = float(cov.get())
        for i, var in self._var.items():
            res[(i, i)] = float(var.get())
        return res

    @property
    def corr(self) -> Dict[Tuple[Any, Any], float]:
        res: Dict[Tuple[Any, Any], float] = {}
        vars: Dict[Any, float] = {}

        for i, var in self._var.items():
            vars[i] = var.get()
            res[(i, i)] = 1

        for key, cov in self._cov.items():
            (i, j) = key
            var_x: float = vars[i]
            var_y: float = vars[j]
            if var_x and var_y:
                res[key] = float(cov.get() / (var_x * var_y) ** 0.5)
            else:
                res[key] = 0.0

        return res

    def get(self) -> Dict[Tuple[Any, Any], float]:
        return self.cov

    def get_cov(self, row: Any, col: Any) -> float:
        key = (row, col)
        try:
            cov = self._cov[key]
        except KeyError:
            key = (col, row)
            cov = self._cov[key]
        return cov.get()

    def get_var(self, col: Any) -> float:
        var = self._var[col]
        return var.get()

    @staticmethod
    def columns(
            d: Dict[Tuple[Any, Any], float]
    ) -> Set[Any]:
        keys: Set[Any] = set()
        for key in d:
            keys.update(key)
        return keys

    @staticmethod
    def as_matrix(
            d: Dict[Tuple[Any, Any], float]
    ) -> np.typing.NDArray[np.float64]:
        keys = CovarianceMatrix.columns(d)
        for key in d:
            keys.update(key)
        res = np.ndarray((len(keys), len(keys)), dtype=np.float64)
        index = {key: i for i, key in enumerate(sorted(keys))}
        for key, val in d.items():
            c1, c2 = key
            i = index[c1]
            j = index[c2]
            res[i, j] = res[j, i] = val
        return res

    def cov_matrix(self) -> np.typing.NDArray[np.float64]:
        return self.as_matrix(self.cov)

    def corr_matrix(self) -> np.typing.NDArray[np.float64]:
        return self.as_matrix(self.corr)

    @staticmethod
    def as_pandas(
            d: Dict[Tuple[Any, Any], float],
            columns: Collection[str] | None = None
    ) -> pd.DataFrame:
        keys = CovarianceMatrix.columns(d)
        if columns is not None:
            keys &= set(columns)
        cols = sorted(keys)
        arr = CovarianceMatrix.as_matrix(d)
        return pd.DataFrame(
            arr,
            index=cols,
            columns=cols,
            dtype="float64"
        )
