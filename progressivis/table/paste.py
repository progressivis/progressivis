"Binary Join module."
from __future__ import annotations

from progressivis.core.module import Module, ReturnRunStep, def_input, def_output
from progressivis.utils.inspect import filter_kwds
from .table import PTable
from .join_by_id import join
from .dshape import dshape_join
from collections import OrderedDict

from typing import Any


@def_input("first", PTable)
@def_input("second", PTable)
@def_output("result", PTable)
class Paste(Module):
    """
    Binary join module to join two tables and return a third one.

    Slots:
        first : PTable module producing the first table to join
        second : PTable module producing the second table to join
    Args:
        kwds : argument to pass to the join function
    """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.join_kwds = filter_kwds(kwds, join)

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        first_slot = self.get_input_slot("first")
        # first_slot.update(run_number)
        second_slot = self.get_input_slot("second")
        first_table = first_slot.data()
        second_table = second_slot.data()
        if first_table is None or second_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # second_slot.update(run_number)
        if first_slot.deleted.any() or second_slot.deleted.any():
            first_slot.reset()
            second_slot.reset()
            if self.result is not None:
                self.result.resize(0)
            first_slot.update(run_number)
            second_slot.update(run_number)
        first_slot.created.next(length=step_size)
        second_slot.created.next(length=step_size)
        first_slot.updated.next(length=step_size)
        second_slot.updated.next(length=step_size)
        if self.result is None:
            dshape, rename = dshape_join(first_table.dshape, second_table.dshape)
            self.result = PTable(name=None, dshape=dshape)
        if len(first_table) == 0 or len(second_table) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        col_0 = first_table.columns[0]
        col_1 = second_table.columns[0]
        if len(self.result) == 0:
            self.result.append(
                OrderedDict(
                    [
                        (col_0, first_table.last(col_0)),
                        (col_1, second_table.last(col_1)),
                    ]
                ),
                indices=[0],
            )
        else:
            assert len(self.result) == 1
            if first_table.last(col_0) != self.result.last(col_0):
                self.result[col_0].loc[0] = first_table.last(col_0)
            if second_table.last(col_1) != self.result.last(col_1):
                self.result[col_1].loc[0] = second_table.last(col_1)
        return self._return_run_step(self.next_state(first_slot), steps_run=1)
