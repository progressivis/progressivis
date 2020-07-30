"""
Compute the index of the maximum value of one or many table columns.
"""

from collections import OrderedDict
import logging

import numpy as np

from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..core.decorators import *


logger = logging.getLogger(__name__)


class IdxMax(TableModule):
    parameters = [('history', np.dtype(int), 3)]
    inputs = [SlotDescriptor('table', type=Table, required=True)]
    outputs = [SlotDescriptor('max', type=Table, required=False)]

    def __init__(self, **kwds):
        super(IdxMax, self).__init__(**kwds)
        self._max = None
        self.default_step_size = 10000

    def max(self):
        return self._max

    def get_data(self, name):
        if name == 'max':
            return self.max()
        return super(IdxMax, self).get_data(name)

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(IdxMax, self).is_ready()

    @process_slot("table")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):

        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            op = self.filter_columns(input_df, fix_loc(indices)).idxmax()
            if self._max is None:
                max_ = OrderedDict(zip(op.keys(), [np.nan]*len(op.keys())))
                for col, ix in op.items():
                    max_[col] = input_df.at[ix, col]  # lookup value, is there a better way?
                self._max = Table(self.generate_table_name('_max'),
                                  dshape=input_df.dshape,
                                  create=True)
                self._max.append(max_, indices=[run_number])
                self._table = Table(self.generate_table_name('table'),
                                    dshape=input_df.dshape,
                                    create=True)
                self._table.append(op, indices=[run_number])
            else:
                prev_max = self._max.last()
                prev_idx = self._table.last()
                max_ = OrderedDict(prev_max.items())
                for col, ix in op.items():
                    val = input_df.at[ix, col]
                    if np.isnan(val):
                        pass
                    elif np.isnan(max_[col]) or val > max_[col]:
                        op[col] = prev_idx[col]
                        max_[col] = val
                self._table.append(op, indices=[run_number])
                self._max.append(max_, indices=[run_number])
                if len(self._table) > self.params.history:
                    data = self._table.loc[self._table.index[-self.params.history:]]
                    self._table = Table(self.
                                        generate_table_name('table'),
                                        data=data,
                                        create=True)
                    data = self._max.loc[self._max.index[-self.params.history:]]
                    self._max = Table(self.
                                      generate_table_name('_max'),
                                      data=data,
                                      create=True)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
