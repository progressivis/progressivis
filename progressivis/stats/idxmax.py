"""
Compute the index of the maximum value of one or many table columns.
"""
from __future__ import annotations

from collections import OrderedDict
import logging

import numpy as np

from ..core.module import ReturnRunStep
from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.module import PTableModule
from ..table.table import PTable
from ..core.decorators import process_slot, run_if_any

from typing import Optional, Any


logger = logging.getLogger(__name__)


class IdxMax(PTableModule):
    parameters = [("history", np.dtype(int), 3)]
    inputs = [SlotDescriptor("table", type=PTable, required=True)]
    outputs = [SlotDescriptor("max", type=PTable, required=False)]

    def __init__(self, **kwds: Any) -> None:
        super(IdxMax, self).__init__(**kwds)
        self._max: Optional[PTable] = None
        self.default_step_size = 10000

    def max(self) -> Optional[PTable]:
        return self._max

    def get_data(self, name: str) -> Any:
        if name == "max":
            return self.max()
        return super(IdxMax, self).get_data(name)

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super(IdxMax, self).is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)
        if self._max is not None:
            self._max.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(length=step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            op = self.filter_columns(input_df, fix_loc(indices)).idxmax()
            if self.result is None:
                self.result = PTable(
                    self.generate_table_name("table"),
                    dshape=input_df.dshape,
                    create=True,
                )

            if not self._max:  # None or len()==0
                max_ = OrderedDict(zip(op.keys(), [np.nan] * len(op.keys())))
                for col, ix in op.items():
                    max_[col] = input_df.at[
                        ix, col
                    ]  # lookup value, is there a better way?
                if self._max is None:
                    self._max = PTable(
                        self.generate_table_name("_max"),
                        dshape=input_df.dshape,
                        create=True,
                    )
                self._max.append(max_, indices=[run_number])
                self.table.append(op, indices=[run_number])
            else:
                prev_max = self._max.last()
                prev_idx = self.table.last()
                assert prev_max is not None and prev_idx is not None
                max_ = OrderedDict(prev_max.items())
                for col, ix in op.items():
                    val = input_df.at[ix, col]
                    if np.isnan(val):
                        pass
                    elif np.isnan(max_[col]) or val > max_[col]:
                        op[col] = prev_idx[col]
                        max_[col] = val
                self.table.append(op, indices=[run_number])
                self._max.append(max_, indices=[run_number])
                if len(self.table) > self.params.history:
                    data = self.table.loc[self.table.index[-self.params.history :]]
                    assert data is not None
                    row = data.to_dict(orient="list")
                    self.table.resize(0)
                    self.table.append(row)
                    data = self._max.loc[self._max.index[-self.params.history :]]
                    assert data is not None
                    row = data.to_dict(orient="list")
                    self._max.resize(0)
                    self._max.append(row)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
