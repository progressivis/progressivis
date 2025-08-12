import numpy as np
import pandas as pd
from numbers import Number
from typing import (
    Optional,
    Union,
    Any,
    Hashable as Key,
    cast,
    TYPE_CHECKING
)
from collections import defaultdict
from datasketches import (
    kll_floats_sketch,
    kll_ints_sketch,
    frequent_strings_sketch,
    frequent_items_error_type,
)
from progressivis.core.utils import nn, is_str, is_dict
from .online import Mean


if TYPE_CHECKING:
    DTypes = Union[pd.Series[Any], dict[str, Union[str, np.dtype[Any]]]]


class SimpleImputer:
    def __init__(
        self,
        strategy: Optional[Union[str, dict[Key, str]]] = None,
        default_strategy: Optional[str] = None,
        fill_values: Optional[
            Union[str, int, Number, dict[Key, Union[str, Number]]]
        ] = None,
    ) -> None:
        if nn(default_strategy):
            assert is_str(default_strategy)
            if not is_dict(strategy):
                raise ValueError(
                    "'default_strategy' is allowed" " only when strategy is a dict"
                )
        _strategy: Union[str, dict[Key, str]] = (
            {} if (strategy is None or is_str(strategy)) else strategy
        )
        assert isinstance(_strategy, dict)
        _default_strategy = default_strategy or "mean"
        self._impute_all: bool = False
        if isinstance(strategy, str):
            _default_strategy = strategy
        self._strategy: dict[Key, Any] = defaultdict(  # type: ignore
            lambda: _default_strategy, **_strategy
        )
        if is_str(strategy) or strategy is None:
            self._impute_all = True
        if _default_strategy == "constant" or "constant" in _strategy:
            assert nn(fill_values)
            self._fill_values = (
                fill_values
                if is_dict(fill_values)
                else defaultdict(lambda: fill_values)
            )
        self._means: dict[Key, Mean] = {}
        self._medians: dict[Key, Union[kll_ints_sketch, kll_floats_sketch]] = {}
        self._frequent: dict[Key, frequent_strings_sketch] = {}
        self._dtypes: DTypes = {}
        self._k = 200  # for sketching

    def init(self, dtypes: "DTypes") -> None:
        if self._impute_all:
            self._dtypes = dtypes
        else:
            self._dtypes = cast(
                DTypes, {k: v for (k, v) in dtypes.items() if k in self._strategy}
            )
        for col, ty in self._dtypes.items():
            strategy = self.get_strategy(col)
            if strategy == "mean":
                if not np.issubdtype(ty, np.number):
                    raise ValueError(f"{strategy = } not compatible with {ty}")
                self._means[col] = Mean()
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

    def get_strategy(self, col: Key) -> Any:
        return self._strategy[col]

    def add_df(self, df: pd.DataFrame) -> None:
        for col, dt in self._dtypes.items():
            strategy = self.get_strategy(col)
            if strategy == "constant":
                continue
            add_strategy = getattr(self, f"add_{strategy}")
            add_strategy(col, df[cast(str, col)], dt)

    def add_mean(self, col: str, val: Any, dt: np.dtype[Any]) -> None:
        self._means[col].update(val)

    def add_median(self, col: str, val: Any, dt: np.dtype[Any]) -> None:
        sk: Union[kll_ints_sketch, kll_floats_sketch]
        if np.issubdtype(dt, np.integer):
            sk = kll_ints_sketch(self._k)
        else:
            assert np.issubdtype(dt, np.floating)
            sk = kll_floats_sketch(self._k)
        sk.update(val)
        self._medians[col].merge(sk)  # type: ignore

    def add_most_frequent(self, col: str, val: Any, dt: np.dtype[Any]) -> None:
        fi = frequent_strings_sketch(self._k)
        for s in val.astype(str):
            fi.update(s)
        self._frequent[col].merge(fi)

    def add_constant(self, col: str, val: Any, dt: np.dtype[Any]) -> None:
        pass

    def getvalue(self, col: str) -> Any:
        strategy = self.get_strategy(col)
        get_val_strategy = getattr(self, f"get_val_{strategy}")
        return get_val_strategy(col)

    def get_val_mean(self, col: str) -> Any:
        return self._means[col].get()

    def get_val_median(self, col: str) -> Any:
        return self._medians[col].get_quantile(0.5)

    def get_val_most_frequent(self, col: str) -> Any:
        return self._frequent[col].get_frequent_items(
            frequent_items_error_type.NO_FALSE_POSITIVES
        )[0][0]

    def get_val_constant(self, col: str) -> Any:
        assert isinstance(self._fill_values, dict)
        return self._fill_values[col]
