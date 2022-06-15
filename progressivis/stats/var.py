from __future__ import annotations

import logging

import numpy as np

from ..core.module import ReturnRunStep
from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor, Slot
from ..core.decorators import process_slot, run_if_any
from ..table.module import TableModule
from ..table.table_base import BaseTable
from ..table.table import Table
from ..table.dshape import dshape_all_dtype
from ..utils.psdict import PsDict
from .utils import OnlineVariance
from typing import Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from progressivis.table.module import Columns

logger = logging.getLogger(__name__)


class VarH(TableModule):
    """
    Compute the variance of the columns of an input table.
    """

    parameters = [("history", np.dtype(int), 3)]
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, **kwds: Any) -> None:
        super().__init__(dataframe_slot="table", **kwds)
        self._data: Dict[str, OnlineVariance] = {}
        self.default_step_size = 1000

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super().is_ready()

    def op(self, chunk: BaseTable) -> Dict[str, float]:
        cols = chunk.columns
        ret: Dict[str, float] = {}
        for c in cols:
            data = self._data.get(c)
            if data is None:
                data = OnlineVariance()
                self._data[c] = data
            data.add(chunk[c])
            ret[c] = data.variance
        return ret

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot: Slot = ctx.table
            indices = dfslot.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            op = self.op(self.filter_columns(input_df, fix_loc(indices)))
            if self.result is None:
                ds = dshape_all_dtype(input_df.columns, np.dtype("float64"))
                self.result = Table(
                    self.generate_table_name("var"),
                    dshape=ds,  # input_df.dshape,
                    create=True,
                )
            self.table.append(op, indices=[run_number])
            return self._return_run_step(self.next_state(dfslot), steps)


class Var(TableModule):
    """
    Compute the variance of the columns of an input table.
    This variant didn't keep history
    """

    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, ignore_string_cols: bool = False, **kwds: Any) -> None:
        super().__init__(dataframe_slot="table", **kwds)
        self._data: Dict[str, OnlineVariance] = {}
        self._ignore_string_cols: bool = ignore_string_cols
        self._num_cols: Columns = None
        self.default_step_size = 1000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def get_num_cols(self, input_df: BaseTable) -> Columns:
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

    def op(self, chunk: BaseTable) -> Dict[str, float]:
        cols = chunk.columns
        ret: Dict[str, float] = {}
        for c in cols:
            data = self._data.get(c)
            if data is None:
                data = OnlineVariance()
                self._data[c] = data
            data.add(chunk[c])
            ret[c] = data.variance
        return ret

    def reset(self) -> None:
        if self.result is not None:
            self.psdict.clear()
        if self._data is not None:
            for ov in self._data.values():
                ov.reset()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            cols = None
            if self._ignore_string_cols:
                cols = self.get_num_cols(input_df)
            op = self.op(self.filter_columns(input_df, fix_loc(indices), cols=cols))
            if self.result is None:
                self.result = PsDict(op)
            else:
                self.psdict.update(op)
            return self._return_run_step(self.next_state(dfslot), steps)
