"""
Compute the index of the minimum value of one or many table columns.
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import logging

import numpy as np

from progressivis.utils.synchronized import synchronized
from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.slot import SlotDescriptor
from progressivis.table.module import TableModule
from progressivis.table.table import Table

logger = logging.getLogger(__name__)


class IdxMin(TableModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        self._add_slots(kwds, 'output_descriptors',
                        [SlotDescriptor('min', type=Table, required=False)])
        super(IdxMin, self).__init__(**kwds)
        self._min = None
        self.default_step_size = 10000

    def min(self):
        return self._min

    def get_data(self, name):
        if name == 'min':
            return self.min()
        return super(IdxMin, self).get_data(name)

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(IdxMin, self).is_ready()

    async def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():
            dfslot.reset()
            self._table = None
            dfslot.update(run_number)
        indices = dfslot.created.next(step_size) # returns a slice
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_table = dfslot.data()
        op = self.filter_columns(input_table, fix_loc(indices)).idxmin()
        #if not op.index.equals(self._columns):
        #    # some columns are not numerical
        #    self._columns = op.index


        if self._min is None:
            min_ = OrderedDict(zip(op.keys(), [np.nan]*len(op.keys())))
            for col, ix in op.items():
                min_[col] = input_table.at[ix, col] # lookup value, is there a better way?
            self._min = Table(self.generate_table_name('_min'),
                              dshape=input_table.dshape,
                              create=True)
            self._min.append(min_, indices=[run_number])
            self._table = Table(self.generate_table_name('_table'),
                                dshape=input_table.dshape,
                                create=True)
            self._table.append(op, indices=[run_number])
        else:
            prev_min = self._min.last()
            prev_idx = self._table.last()
            min_ = OrderedDict(prev_min.items())
            for col, ix in op.items():
                val = input_table.at[ix, col]
                if np.isnan(val):
                    pass
                elif np.isnan(min_[col]) or val < min_[col]:
                    op[col] = prev_idx[col]
                    min_[col] = val
            with self.lock:
                self._table.append(op, indices=[run_number])
                self._min.append(min_, indices=[run_number])
                if len(self._table) > self.params.history:
                    data = self._table.loc[self._table.index[-self.params.history:]]
                    self._table = Table(self.generate_table_name('_table'),
                                        data=data,
                                        create=True)
                    data = self._min.loc[self._min.index[-self.params.history:]]
                    self._min = Table(self.generate_table_name('_min'),
                                      data=data,
                                      create=True)

        return self._return_run_step(self.next_state(dfslot), steps_run=steps)
