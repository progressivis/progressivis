import numpy as np
from numbers import Number
from typing import Iterable, Dict, Optional, Union
from collections import defaultdict
from datasketches import (
    kll_floats_sketch,
    kll_ints_sketch,
    frequent_strings_sketch,
    frequent_items_error_type,
)
from ..core.utils import nn, is_str, is_dict


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
                    "'default_strategy' is allowed" " only when strategy is a dict"
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
            self._dtypes = {k: v for (k, v) in dtypes.items() if k in self.strategy}
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
