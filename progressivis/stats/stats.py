from __future__ import annotations

import logging

import numpy as np

from progressivis.core.module import (
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from progressivis.core.api import indices_len, fix_loc, get_random_name, Module

from progressivis.table.api import PTable

from typing import Optional, Any, Union

# TODO update with http://www.johndcook.com/blog/skewness_kurtosis/
# Use http://www.grantjenks.com/docs/runstats/


logger = logging.getLogger(__name__)


@document
@def_input("table", PTable, doc="the input table")
@def_parameter(
    "history", np.dtype(int), 3, doc=("then number of successive results to be kept")
)
@def_output(
    "result", PTable, doc="result table containing two columns (for min and max value)"
)
class Stats(Module):
    """
    Computes the minimum and the maximum of the values for a single column
    """

    def __init__(
        self,
        column: Union[str, int],
        min_column: Optional[str] = None,
        max_column: Optional[str] = None,
        reset_index: bool = False,
        **kwds: Any,
    ) -> None:
        """
        Args:
            column: the name or the position of the column to be processed
            min_column: the name of the minimum column in the ``result`` table.
                When missing, the given name is ``_<column>_min``
            man_column: the name of the maximum column in the ``result`` table.
                When missing, the given name is ``_<column>_max``
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self._column = column
        self.default_step_size = 10000

        if min_column is None:
            min_column = "_" + str(column) + "_min"
        if max_column is None:
            max_column = "_" + str(column) + "_max"
        self._min_column: str = min_column
        self._max_column: str = max_column
        self._reset_index = reset_index
        # self.schema = [(self._min_column, np.dtype(float), np.nan),
        #                (self._max_column, np.dtype(float), np.nan),]
        self.schema = (
            "{" + self._min_column + ": float64, " + self._max_column + ": float64}"
        )
        self.result = PTable(get_random_name("stats_"), dshape=self.schema)

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super(Stats, self).is_ready()

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        prev_min = prev_max = np.nan
        dfslot = self.get_input_slot("table")
        assert dfslot is not None
        assert self.result is not None
        if dfslot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if dfslot.updated.any() or dfslot.deleted.any():
            dfslot.reset()
            dfslot.update(run_number)
        else:
            df = self.result
            prev = df.last_id
            if prev > 0:
                prev_min = df.at[prev, self._min_column]
                prev_max = df.at[prev, self._max_column]

        indices = dfslot.created.next(length=step_size)  # returns a slice
        input_df = dfslot.data()
        steps = indices_len(indices)
        if steps > 0:
            x = input_df.to_array(locs=fix_loc(indices), columns=[self._column])
            new_min = np.nanmin(x)
            new_max = np.nanmax(x)
            row = {
                self._min_column: np.nanmin([prev_min, new_min]),
                self._max_column: np.nanmax([prev_max, new_max]),
            }
            if run_number in df.index:
                df.loc[run_number] = row
            else:
                df.add(row, index=run_number)
        return self._return_run_step(self.next_state(dfslot), steps)
