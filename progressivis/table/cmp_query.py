from __future__ import annotations

import operator
import logging

import numpy as np

from ..core.module import Module, ReturnRunStep, def_input, def_output
from ..core.pintset import PIntSet
from .table import PTable

from typing import Optional, Any

logger = logging.getLogger(__name__)

ops = {
    "<": operator.__lt__,
    "<=": operator.__le__,
    ">": operator.__gt__,
    ">=": operator.__ge__,
    "and": operator.__and__,
    "or": operator.__or__,
    "xor": operator.__xor__,
    "==": operator.__eq__,
    "!=": operator.__ne__,
}


@def_input("table", PTable)
@def_input("cmp", PTable)
@def_output("result", PTable, required=False)
@def_output("select", PIntSet, required=False)
class CmpQueryLast(Module):
    """ """

    def __init__(self, op: str = "<", combine: str = "and", **kwds: Any) -> None:
        super(CmpQueryLast, self).__init__(**kwds)
        self.default_step_size = 1000
        self.op = op
        self._op = ops[op]
        self.combine = combine
        self._combine = ops[combine]

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        table_slot = self.get_input_slot("table")
        assert table_slot is not None
        table_data = table_slot.data()
        cmp_slot = self.get_input_slot("cmp")
        assert cmp_slot is not None
        cmp_slot.clear_buffers()
        cmp_data = cmp_slot.data()

        if (
            table_data is None
            or len(table_data) == 0
            or cmp_data is None
            or len(cmp_data) == 0
        ):
            # nothing to do if no filter is specified
            if self.select is not None:
                self.select.clear()
            return self._return_run_step(self.state_blocked, steps_run=1)
        if table_slot.deleted.any() or cmp_slot.deleted.any():
            # restart from scatch
            table_slot.reset()
            if self.select is not None:
                self.select.clear()
            table_slot.update(run_number)
            cmp_slot.update(run_number)

        cr = table_slot.created.next(as_slice=False)
        if cr is None:
            cr = PIntSet()
        up = table_slot.updated.next(as_slice=False)
        work = cr | up
        ids = work.pop(step_size)
        if cr:
            table_slot.created.push(cr - ids)
        if up:
            table_slot.updated.push(up - ids)
        steps = len(ids)
        aids = np.asarray(ids, dtype=np.int64)
        indices = table_data.id_to_index(aids)
        last = cmp_data.last()
        results: Optional[PIntSet] = None
        for colname in last:
            if colname in table_data:
                arg1 = table_data._column(colname)
                arg2 = last[colname]
                res = self._op(arg1[indices], arg2)
                res = aids[res]
                if results is None:
                    results = PIntSet(res)
                else:
                    results = self._combine(results, PIntSet(res))

        if self.select is None:
            self.select = results
        else:
            self.select.difference_update(PIntSet(indices))
            self.select.update(results)

        return self._return_run_step(self.next_state(table_slot), steps_run=steps)
