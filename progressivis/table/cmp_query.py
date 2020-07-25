
import operator
import logging

import numpy as np

from ..core.slot import SlotDescriptor
from .module import TableModule
from ..core.bitmap import bitmap
from .table import Table

logger = logging.getLogger(__name__)

ops = {"<":   operator.__lt__,
       "<=":  operator.__le__,
       ">":   operator.__gt__,
       ">=":  operator.__ge__,
       "and": operator.__and__,
       "or":  operator.__or__,
       "xor": operator.__xor__,
       "==":  operator.__eq__,
       "!=":  operator.__ne__
       }


class CmpQueryLast(TableModule):
    inputs = [SlotDescriptor('table', type=Table, required=True),
              SlotDescriptor('cmp', type=Table, required=True)]
    outputs = [SlotDescriptor('select', type=bitmap, required=False)]

    def __init__(self, op="<", combine="and", **kwds):
        super(CmpQueryLast, self).__init__(**kwds)
        self.default_step_size = 1000
        self.op = op
        self._op = ops[op]
        self.combine = combine
        self._combine = ops[combine]
        self._bitmap = None

    def get_data(self, name):
        if name == 'select':
            return self._bitmap
        if name == 'table':
            self.get_input_slot('table').data()
        return super(CmpQueryLast, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        table_slot = self.get_input_slot('table')
        # table_slot.update(run_number)
        table_data = table_slot.data()
        cmp_slot = self.get_input_slot('cmp')
        # cmp_slot.update(run_number)
        cmp_slot.clear_buffers()
        cmp_data = cmp_slot.data()

        if table_data is None \
           or len(table_data) == 0 \
           or cmp_data is None \
           or len(cmp_data) == 0:
            # nothing to do if no filter is specified
            self._bitmap = None
            return self._return_run_step(self.state_blocked, steps_run=1)
        if table_slot.deleted.any() or cmp_slot.deleted.any():
            # restart from scatch
            table_slot.reset()
            self._bitmap = None
            table_slot.update(run_number)
            cmp_slot.update(run_number)

        cr = table_slot.created.next(as_slice=False)
        if cr is None:
            cr = bitmap()
        up = table_slot.updated.next(as_slice=False)
        work = cr | up
        ids = work.pop(step_size)
        if cr:
            table_slot.created.push(cr - ids)
        if up:
            table_slot.updated.push(up - ids)
        steps = len(ids)
        ids = np.asarray(ids, dtype=np.int64)
        indices = table_data.id_to_index(ids)
        last = cmp_data.last()
        results = None
        for colname in last:
            if colname in table_data:
                arg1 = table_data._column(colname)
                arg2 = last[colname]
                res = self._op(arg1[indices], arg2)
                res = ids[res]
                if results is None:
                    results = bitmap(res)
                else:
                    results = self._combine(results, bitmap(res))

        if self._bitmap is None:
            self._bitmap = results
        else:
            self._bitmap -= bitmap(indices)
            self._bitmap |= results

        return self._return_run_step(self.next_state(table_slot),
                                     steps_run=steps)
