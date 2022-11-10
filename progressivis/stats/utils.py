import numpy as np
from numbers import Number
from typing import Iterable, Dict, Optional, Union, Any, Set, Type
from collections import defaultdict
from abc import abstractmethod
from datasketches import (
    kll_floats_sketch,
    kll_ints_sketch,
    frequent_strings_sketch,
    frequent_items_error_type,
)
from ..table.column_base import BaseColumn
from ..core.utils import nn, is_str, is_dict

Num = Union[float, int]


class OnlineFunctor:
    name = "functor"

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("reset not defined")

    @abstractmethod
    def add(self, iterable: Iterable[Any]) -> None:
        raise NotImplementedError("add not defined")

    @abstractmethod
    def get_value(self) -> Any:
        raise NotImplementedError("add not defined")


aggr_registry: Dict[str, Type[OnlineFunctor]] = {}


class OnlineSum(OnlineFunctor):
    """ """

    name = "sum"

    def __init__(self) -> None:
        self._sum: Num
        self.reset()

    def reset(self) -> None:
        self._sum = 0

    def add(self, iterable: Iterable[Num]) -> None:
        if iterable is not None:
            self._sum += sum(iterable)

    def get_value(self) -> Num:
        return self._sum


aggr_registry[OnlineSum.name] = OnlineSum


class OnlineCount(OnlineFunctor):
    """ """

    name = "count"

    def __init__(self) -> None:
        self.count: int
        self.reset()

    def reset(self) -> None:
        self.count = 0

    def add(self, iterable: Iterable[Any]) -> None:
        if iterable is not None:
            self.count += len(iterable)  # type: ignore

    def get_value(self) -> float:
        return self.count


aggr_registry[OnlineCount.name] = OnlineCount


class OnlineSet(OnlineFunctor):
    """ """

    name = "set"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._set: Set[Any] = set()

    def add(self, iterable: Iterable[Any]) -> None:
        if iterable is not None:
            self._set.update(set(iterable))

    def get_value(self) -> Any:
        return self._set


aggr_registry[OnlineSet.name] = OnlineSet


class OnlineUnique(OnlineSet):
    """ """

    name = "uniq"

    def get_value(self) -> Any:
        assert len(self._set) == 1
        return list(self._set)[0]


aggr_registry[OnlineUnique.name] = OnlineUnique


class OnlineMean(OnlineFunctor):
    """ """

    name = "mean"

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

    def get_value(self) -> float:
        return self.mean


aggr_registry[OnlineMean.name] = OnlineMean


# Should translate that to Cython eventually
class OnlineVariance(OnlineFunctor):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    name = "variance"

    def __init__(self, ddof: int = 1) -> None:
        self.reset()
        self.ddof: int = ddof

    def reset(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0
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
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self) -> float:
        n_ddof = self.n - self.ddof
        return self.M2 / n_ddof if n_ddof else np.nan

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)  # type: ignore

    def get_value(self) -> float:
        return self.variance


aggr_registry[OnlineVariance.name] = OnlineVariance


class OnlineStd(OnlineVariance):
    name = "stddev"

    def get_value(self) -> float:
        return self.std


aggr_registry[OnlineStd.name] = OnlineStd


class OnlineCovariance:  # not an OnlineFuctor
    def __init__(self, ddof: int = 1) -> None:
        self.reset()
        self.ddof = ddof

    def reset(self) -> None:
        self.n: float = 0
        self.mean_x: float = 0
        self.sum_x: float = 0
        self.mean_y: float = 0
        self.sum_y: float = 0
        self.cm: float = 0

    def include(self, x: float, y: float) -> None:
        self.n += 1
        dx = x - self.mean_x
        self.sum_x += x
        self.sum_y += y
        self.mean_x = self.sum_x / self.n
        self.mean_y = self.sum_y / self.n
        self.cm += dx * (y - self.mean_y)

    def add(self, array_x: BaseColumn, array_y: BaseColumn) -> None:
        for x, y in zip(array_x, array_y):
            self.include(x, y)

    @property
    def cov(self) -> float:
        div_ = self.n - self.ddof
        return self.cm / div_ if div_ else np.nan


class SimpleImputer:
    def __init__(
        self,
        strategy: Optional[Union[str, Dict[str, str]]] = None,
        default_strategy: Optional[str] = None,
        fill_values: Optional[Union[str, Number, Dict[str, Union[str, Number]]]] = None,
    ):
        if nn(default_strategy):
            assert is_str(default_strategy)
            if not is_dict(strategy):
                raise ValueError(
                    "'default_strategy' is allowed only when strategy is a dict"
                )
        _strategy: Union[str, Dict[str, str]] = (
            {} if (strategy is None or is_str(strategy)) else strategy
        )  # type: ignore
        assert isinstance(_strategy, dict)
        _default_strategy = default_strategy or "mean"
        self._impute_all: bool = False
        if isinstance(strategy, str):
            _default_strategy = strategy
        self._strategy: Dict = defaultdict(lambda: _default_strategy, **_strategy)
        if is_str(strategy) or strategy is None:
            self._impute_all = True
        if _default_strategy == "constant" or "constant" in _strategy:
            assert nn(fill_values)
            self._fill_values = (
                fill_values if is_dict(fill_values) else defaultdict(lambda: fill_values)  # type: ignore
            )
        self._means: Dict[str, OnlineMean] = {}
        self._medians: Dict[str, Union[kll_ints_sketch, kll_floats_sketch]] = {}
        self._frequent: Dict[str, frequent_strings_sketch] = {}
        self._dtypes: Dict[str, Union[str, np.dtype]] = {}
        self._k = 200  # for sketching

    def init(self, dtypes):
        if self._impute_all:
            self._dtypes = dtypes
        else:
            self._dtypes = {k: v for (k, v) in dtypes.items() if k in self._strategy}
        for col, ty in self._dtypes.items():
            strategy = self.get_strategy(col)
            if strategy == "mean":
                if not np.issubdtype(ty, np.number):
                    raise ValueError(f"{strategy = } not compatible with {ty}")
                self._means[col] = OnlineMean()
            elif strategy == "median":
                if np.issubdtype(ty, np.floating):
                    self._medians[col] = kll_floats_sketch(self._k)
                elif np.issubdtype(ty, np.integer):
                    self._medians[col] = kll_ints_sketch(self._k)
                else:
                    raise ValueError(f"{strategy = } not compatible with {ty}")
            elif strategy == "most_frequent":
                self._frequent[col] = frequent_strings_sketch(self._k)
            elif strategy != "constant":
                raise ValueError(f"Unknown imputation {strategy = }")

    def get_strategy(self, col):
        return self._strategy[col]

    def add_df(self, df):
        for col, dt in self._dtypes.items():
            strategy = self.get_strategy(col)
            if strategy == "constant":
                continue
            add_strategy = getattr(self, f"add_{strategy}")
            add_strategy(col, df[col], dt)

    def add_mean(self, col, val, dt):
        self._means[col].add(val)

    def add_median(self, col, val, dt):
        if np.issubdtype(dt, np.integer):
            sk = kll_ints_sketch(self._k)
        else:
            assert np.issubdtype(dt, np.floating)
            sk = kll_floats_sketch(self._k)
        sk.update(val)
        self._medians[col].merge(sk)

    def add_most_frequent(self, col, val, dt):
        fi = frequent_strings_sketch(self._k)
        for s in val.astype(str):
            fi.update(s)
        self._frequent[col].merge(fi)

    def add_constant(self, col, val, dt):
        pass

    def getvalue(self, col):
        strategy = self.get_strategy(col)
        get_val_strategy = getattr(self, f"get_val_{strategy}")
        return get_val_strategy(col)

    def get_val_mean(self, col):
        return self._means[col].mean

    def get_val_median(self, col):
        return self._medians[col].get_quantile(0.5)

    def get_val_most_frequent(self, col):
        return self._frequent[col].get_frequent_items(
            frequent_items_error_type.NO_FALSE_POSITIVES
        )[0][0]

    def get_val_constant(self, col):
        return self._fill_values[col]
