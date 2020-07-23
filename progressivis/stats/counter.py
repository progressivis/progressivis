
from progressivis.utils.synchronized import synchronized
from progressivis.core.utils import indices_len
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor
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

    @synchronized
    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():
            dfslot.reset()
            if self._table is not None:
                self._table.resize(0)
            dfslot.update(run_number)
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
