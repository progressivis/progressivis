import logging

from . import Table
#from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule
from ..core.utils import indices_len, fix_loc
#from .filter_impl import FilterImpl
from ..core.bitmap import bitmap
from . import TableSelectedView
import numexpr as ne
import numpy as np
logger = logging.getLogger(__name__)


class FilterMod(TableModule):
    parameters = [('expr', str, "unknown"),
                  ('user_dict', object, None)]

    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        #self._impl = FilterImpl(self.params.expr, self.params.user_dict)

    def reset(self):
        if self._table is not None:
             self._table.mask = bitmap([])
    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        # input_slot.update(run_number)
        input_table = input_slot.data()        
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._table is None:
            self._table = TableSelectedView(input_table, bitmap([]))
        steps = 0
        if input_slot.updated.any():
            input_slot.reset()
            input_slot.update(run_number)
            self.reset()
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(step_size, as_slice=False)
            self._table.mask -= deleted
            steps += indices_len(deleted)
        if input_slot.created.any():
            created = input_slot.created.next(step_size, as_slice=False)
            indices = fix_loc(created)
            steps += indices_len(created)
            eval_idx = input_table.eval(expr=self.params.expr, locs=np.array(indices),
                                        as_slice=False,result_object='index')
            self._table.mask |=  bitmap(eval_idx)
        if not steps:
            return self._return_run_step(self.state_blocked, steps_run=0)
        return self._return_run_step(self.next_state(input_slot), steps)
