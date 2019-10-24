from __future__ import absolute_import, division, print_function

from progressivis.utils.synchronized import synchronized
from progressivis.core.utils import indices_len, fix_loc
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor

import numpy as np

import logging
logger = logging.getLogger(__name__)


class Min(TableModule):
    #parameters = [('history', np.dtype(int), 3)]

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        super(Min, self).__init__(**kwds)
        self._columns = columns
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(Min, self).is_ready()

    async def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():        
            dfslot.reset()
            if self._table is not None:
                self._table.resize(0)
            dfslot.update(run_number)
        indices = dfslot.created.next(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = dfslot.data()
        op = self.filter_columns(input_df, fix_loc(indices)).min(keepdims=True)
        if self._table is None:
            self._table = Table(self.generate_table_name('min'),
                                data=op,
                                create=True)
        elif len(self._table)==0: # has been resetted
            self._table.append(op)
        else:
            last = self._table.last()
            for colname in last:
                current_max = op[colname]
                current_max[0] = np.minimum(current_max, last[colname])
            self._table.append(op)
        #TODO manage the history in a more efficient way
        #if len(self._table) > self.params.history:
        #    self._table = self._table.loc[self._df.index[-self.params.history:]]
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)
