from ..core.utils import indices_len
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..core.decorators import *
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Counter(TableModule):
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super(Counter, self).__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(Counter, self).is_ready()

    def reset(self):
        if self._table is not None:
                self._table.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table        
            indices = dfslot.created.next(step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            data = pd.DataFrame(dict(counter=steps), index=[0])
            if self._table is None:
                self._table = Table(self.generate_table_name('counter'),
                                    data=data,
                                    create=True)
            elif len(self._table) == 0:
                self._table.append(data)
            else:
                self._table['counter'].loc[0] += steps
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
