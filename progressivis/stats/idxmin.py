"""
Compute the index of the minimum value of one or many table columns.
"""
from __future__ import annotations

from collections import OrderedDict
import logging

import numpy as np

from ..core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from ..core.utils import indices_len, fix_loc
from ..table.table import PTable
from ..core.decorators import process_slot, run_if_any

from typing import Any, Sequence


logger = logging.getLogger(__name__)


@document
@def_parameter(
    "history",
    np.dtype(int),
    33,
    doc=("then number of successive results" " to be kept"),
)
@def_input("table", type=PTable, hint_type=Sequence[str], doc="the input table")
@def_output(
    "min", PTable, attr_name="_min", required=False, doc="minimum values output table"
)
@def_output("result", PTable, doc="indices in the input table of the minimum values")
class IdxMin(Module):
    """ """

    def __init__(
        self,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super(IdxMin, self).is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.resize(0)
        if self._min is not None:
            self._min.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_table = dfslot.data()
            op = self.filter_slot_columns(dfslot, fix_loc(indices)).idxmin()
            if self.result is None:
                self.result = PTable(
                    self.generate_table_name("table"),
                    dshape=input_table.dshape,
                    create=True,
                )
            if not self._min:  # None or len()==0
                min_ = OrderedDict(zip(op.keys(), [np.nan] * len(op.keys())))
                for col, ix in op.items():
                    min_[col] = input_table.at[
                        ix, col
                    ]  # lookup value, is there a better way?
                if self._min is None:
                    self._min = PTable(
                        self.generate_table_name("_min"),
                        dshape=input_table.dshape,
                        create=True,
                    )
                self._min.append(min_, indices=[run_number])
                self.result.append(op, indices=[run_number])
            else:
                prev_min = self._min.last()
                prev_idx = self.result.last()
                assert prev_min is not None and prev_idx is not None
                min_ = OrderedDict(prev_min.items())
                for col, ix in op.items():
                    val = input_table.at[ix, col]
                    if np.isnan(val):
                        pass
                    elif np.isnan(min_[col]) or val < min_[col]:
                        op[col] = prev_idx[col]
                        min_[col] = val
                self.result.append(op, indices=[run_number])
                self._min.append(min_, indices=[run_number])
                if len(self.result) > self.params.history:
                    data = self.result.loc[self.result.index[-self.params.history :]]
                    assert data is not None
                    row = data.to_dict(orient="list")
                    self.result.resize(0)
                    self.result.append(row)

                    data = self._min.loc[self._min.index[-self.params.history :]]
                    assert data is not None
                    row = data.to_dict(orient="list")
                    self._min.resize(0)
                    self._min.append(row)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
