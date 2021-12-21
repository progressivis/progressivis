from __future__ import annotations

from progressivis.core.utils import indices_len, fix_loc, get_random_name
from progressivis.core.slot import SlotDescriptor
from progressivis.table.module import TableModule
from progressivis.table.table import Table

# TODO update with http://www.johndcook.com/blog/skewness_kurtosis/
# Use http://www.grantjenks.com/docs/runstats/

import numpy as np

import logging

logger = logging.getLogger(__name__)


class Stats(TableModule):
    parameters = [('history', np.dtype(int), 3)]
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self,
                 column: str,
                 min_column: str = None,
                 max_column: str = None,
                 reset_index=False,
                 **kwds):
        super(Stats, self).__init__(**kwds)
        self._column = column
        self.default_step_size = 10000

        if min_column is None:
            min_column = '_' + str(column) + '_min'
        if max_column is None:
            max_column = '_' + str(column) + '_max'
        self._min_column = min_column
        self._max_column = max_column
        self._reset_index = reset_index
        # self.schema = [(self._min_column, np.dtype(float), np.nan),
        #                (self._max_column, np.dtype(float), np.nan),]
        self.schema = ('{' +
                       self._min_column+': float64, ' +
                       self._max_column+': float64}')
        self.result = Table(get_random_name('stats_'), dshape=self.schema)

    def is_ready(self) -> bool:
        if self.get_input_slot('table').created.any():
            return True
        return super(Stats, self).is_ready()

    def run_step(self,
                 run_number: int,
                 step_size, howlong):
        prev_min = prev_max = np.nan
        dfslot = self.get_input_slot('table')
        # dfslot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():
            dfslot.reset()
            dfslot.update(run_number)
        else:
            df = self.result
            prev = df.last_id
            if prev > 0:
                prev_min = df.at[prev, self._min_column]
                prev_max = df.at[prev, self._max_column]

        indices = dfslot.created.next(step_size)  # returns a slice
        input_df = dfslot.data()
        steps = indices_len(indices)
        if steps > 0:
            x = input_df.to_array(locs=fix_loc(indices),
                                  columns=[self._column])
            new_min = np.nanmin(x)
            new_max = np.nanmax(x)
            row = {self._min_column: np.nanmin([prev_min, new_min]),
                   self._max_column: np.nanmax([prev_max, new_max])}
            if run_number in df.index:
                df.loc[run_number] = row
            else:
                df.add(row, index=run_number)
        return self._return_run_step(self.next_state(dfslot), steps)
