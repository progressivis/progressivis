from __future__ import annotations

from itertools import product

import pandas as pd

from ..core.module import ReturnRunStep
from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any
from ..table.table_base import BasePTable
from ..table.table import PTable
from ..table.module import PDictModule
from ..utils.psdict import PDict
from .utils import OnlineVariance, OnlineCovariance

from typing import Any, Union, Literal, Dict, Optional, List


class Corr(PDictModule):
    """
    Compute the covariance matrix (a dict, actually) of the columns of an input table.
    """

    inputs = [SlotDescriptor("table", type=PTable, required=True)]

    def __init__(
        self,
        mode: Union[Literal["Pearson"], Literal["CovarianceOnly"]] = "Pearson",
        ignore_string_cols: bool = False,
        **kwds: Any
    ) -> None:
        assert mode in ("Pearson", "CovarianceOnly")
        super().__init__(**kwds)
        self._is_corr: bool = mode == "Pearson"
        self._data: Dict[frozenset[str], OnlineCovariance] = {}
        self._vars: Dict[str, OnlineVariance] = {}
        self._ignore_string_cols = ignore_string_cols
        self._num_cols: Optional[List[str]] = None
        self.default_step_size = 1000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def get_num_cols(self, input_df: BasePTable) -> List[str]:
        if self._num_cols is None:
            if not self._columns:
                self._num_cols = [
                    c.name for c in input_df._columns if str(c.dshape) != "string"
                ]
            else:
                self._num_cols = [
                    c.name
                    for c in input_df._columns
                    if c.name in self._columns and str(c.dshape) != "string"
                ]
        return self._num_cols

    def op(self, chunk: BasePTable) -> Dict[Any, Any]:
        cols = chunk.columns
        cov_: Dict[frozenset[str], float] = {}
        done_ = set()
        for cx, cy in product(cols, cols):
            key = frozenset([cx, cy])
            if key in done_:
                continue
            data = self._data.get(key)
            if data is None:
                data = OnlineCovariance()
                self._data[key] = data
            data.add(chunk[cx], chunk[cy])
            done_.add(key)
            cov_[key] = data.cov
        if not self._is_corr:
            return cov_  # covariance only
        std_: Dict[str, float] = {}
        for c in cols:
            vdata = self._vars.get(c)
            if vdata is None:
                vdata = OnlineVariance()
                self._vars[c] = vdata
            vdata.add(chunk[c])
            std_[c] = vdata.std
        corr_: Dict[frozenset[str], float] = {}
        for k, v in cov_.items():
            lk = list(k)
            if len(lk) == 1:
                kx = ky = lk[0]
            else:
                kx = lk[0]
                ky = lk[1]
            corr_[k] = v / (std_[kx] * std_[ky])
        return corr_

    def reset(self) -> None:
        if self.result is not None:
            self.result.clear()
        if self._data is not None:
            for oc in self._data.values():
                oc.reset()
        if self._vars is not None:
            for ov in self._vars.values():
                ov.reset()

    def result_as_df(self, columns: List[str]) -> pd.DataFrame:
        """
        Convenience method
        """
        res = pd.DataFrame(index=columns, columns=columns, dtype="float64")
        for kx, ky in product(columns, columns):
            res.loc[kx, ky] = self.result[frozenset([kx, ky])]  # type: ignore
        return res

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            cols = None
            if self._ignore_string_cols:
                cols = self.get_num_cols(input_df)
            cov_ = self.op(self.filter_columns(input_df, fix_loc(indices), cols=cols))
            if self.result is None:
                self.result = PDict(other=cov_)
            else:
                self.result.update(cov_)
            return self._return_run_step(self.next_state(dfslot), steps)
