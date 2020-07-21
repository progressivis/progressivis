from ..core.utils import indices_len, fix_loc
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..utils.psdict import PsDict
from ..core.decorators import *
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

    def reset(self):
        if self._table is not None:
            self._table.fill(np.inf)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    @check_slots  
    def run_step(self,run_number,step_size,howlong):
        with self.context as ctx:
            indices = ctx.table.created.next(step_size) # returns a slice
            steps = indices_len(indices)
            input_df = ctx.table.data()
            op = self.filter_columns(input_df, fix_loc(indices)).min(keepdims=False)
            if self._table is None:
                self._table = PsDict(op)
            else:
                for k, v in self._table.items():
                    self._table[k] = np.minimum(op[k], v)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

    def old_run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():        
            dfslot.reset()
            if self._table is not None:
                self._table.fill(np.inf)
            dfslot.update(run_number)
        indices = dfslot.created.next(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = dfslot.data()
        op = self.filter_columns(input_df, fix_loc(indices)).min(keepdims=False)
        if self._table is None:
            self._table = PsDict(op)
        else:
            for k, v in self._table.items():
                self._table[k] = np.minimum(op[k], v)
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)
