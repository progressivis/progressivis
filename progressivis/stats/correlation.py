from __future__ import annotations

import pandas as pd

from ..core.module import ReturnRunStep, def_input, def_output
from ..core.utils import indices_len, fix_loc
from ..core.decorators import process_slot, run_if_any
from ..table.table_base import BasePTable
from ..table.table import PTable
from ..core.module import Module
from ..utils.psdict import PDict
# from .online import Var as OnlineVariance, Cov as OnlineCovariance
from .online import CovarianceMatrix, Column

from typing import Any, Union, Literal, Dict, Optional, List, Sequence


@def_input("table", PTable, hint_type=Sequence[str])
@def_output("result", PDict)
class Corr(Module):
    """
    Compute the covariance matrix (a dict, actually) of the columns of an input table.
    """

    def __init__(
        self,
        mode: Union[Literal["Pearson"], Literal["CovarianceOnly"]] = "Pearson",
        ignore_string_cols: bool = False,
        **kwds: Any,
    ) -> None:
        assert mode in ("Pearson", "CovarianceOnly")
        super().__init__(**kwds)
        self._is_corr: bool = mode == "Pearson"
        # self._data: Dict[frozenset[str], OnlineCovariance] = {}
        # self._vars: Dict[str, OnlineVariance] = {}
        self._cov = CovarianceMatrix()
        self._ignore_string_cols = ignore_string_cols
        self._num_cols: Optional[List[str]] = None
        self.default_step_size = 1000
        self._columns: Optional[Sequence[str]] = None

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
        X: Dict[Any, Column] = {col: chunk[col].value for col in cols}
        self._cov.update_many(X)
        if not self._is_corr:
            return self._cov.cov
        return self._cov.corr
        # cov_: Dict[frozenset[str], float] = {}
        # done_ = set()
        # for cx, cy in product(cols, cols):
        #     key = frozenset([cx, cy])
        #     if key in done_:
        #         continue
        #     data = self._data.get(key)
        #     if data is None:
        #         data = OnlineCovariance()
        #         self._data[key] = data
        #     data.update_many(chunk[cx], chunk[cy])
        #     done_.add(key)
        #     cov_[key] = data.get()
        # if not self._is_corr:
        #     return cov_  # covariance only
        # std_: Dict[str, float] = {}
        # for c in cols:
        #     vdata = self._vars.get(c)
        #     if vdata is None:
        #         vdata = OnlineVariance()
        #         self._vars[c] = vdata
        #     vdata.update_many(chunk[c])
        #     std_[c] = vdata.std
        # corr_: Dict[frozenset[str], float] = {}
        # for k, v in cov_.items():
        #     lk = list(k)
        #     if len(lk) == 1:
        #         kx = ky = lk[0]
        #     else:
        #         kx = lk[0]
        #         ky = lk[1]
        #     corr_[k] = v / (std_[kx] * std_[ky])
        # return corr_

    def reset(self) -> None:
        if self.result is not None:
            self.result.clear()
        # if self._data is not None:
        #     for oc in self._data.values():
        #         oc.reset()
        # if self._vars is not None:
        #     for ov in self._vars.values():
        #         ov.reset()
        self._cov.reset()

    def result_as_df(self, columns: List[str]) -> pd.DataFrame:
        """
        Convenience method
        """
        if self._is_corr:
            res = pd.DataFrame(self._cov.corr_matrix(),
                               index=columns,
                               columns=columns)
        else:
            res = pd.DataFrame(self._cov.cov_matrix(),
                               index=columns,
                               columns=columns)
        return res

    @property
    def columns(self) -> Sequence[str]:
        assert self._columns
        return self._columns

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            assert input_df is not None
            if self._columns is None:
                if (hint := dfslot.hint) is not None:
                    self._columns = hint
                else:
                    self._columns = input_df.columns
            cols = None
            if self._ignore_string_cols:
                cols = self.get_num_cols(input_df)
            cov_ = self.op(self.filter_slot_columns(dfslot, fix_loc(indices), cols=cols))
            if self.result is None:
                self.result = PDict(other=cov_)
            else:
                self.result.update(cov_)
            return self._return_run_step(self.next_state(dfslot), steps)
