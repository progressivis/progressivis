"Binary Join module."
from __future__ import annotations

from progressivis.core.module import ReturnRunStep
from progressivis.core.utils import Dialog, indices_len
from progressivis.core.slot import SlotDescriptor
from progressivis.utils.inspect import filter_kwds
from progressivis.table.table import PTable
from progressivis.table.module import PTableModule
from progressivis.table.join_by_id import join, join_start, join_cont, join_reset

from typing import Any


class BinJoin(PTableModule):
    """
    Binary join module to join two tables and return a third one.

    Slots:
        first : PTable module producing the first table to join
        second : PTable module producing the second table to join
    Args:
        kwds : argument to pass to the join function
    """

    inputs = [
        SlotDescriptor("first", type=PTable, required=True),
        SlotDescriptor("second", type=PTable, required=True),
    ]

    def __init__(self, **kwds: Any) -> None:
        super(BinJoin, self).__init__(**kwds)
        self.join_kwds = filter_kwds(kwds, join)
        self._dialog = Dialog(self)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        first_slot = self.get_input_slot("first")
        # first_slot.update(run_number)
        second_slot = self.get_input_slot("second")
        # second_slot.update(run_number)
        steps = 0
        if first_slot.deleted.any() or second_slot.deleted.any():
            first_slot.reset()
            second_slot.reset()
            if self.result is not None:
                self.table.resize(0)
                join_reset(self._dialog)
            first_slot.update(run_number)
            second_slot.update(run_number)
        created = {}
        if first_slot.created.any():
            indices = first_slot.created.next(length=step_size)
            steps += indices_len(indices)
            created["table"] = indices
        if second_slot.created.any():
            indices = second_slot.created.next(length=step_size)
            steps += indices_len(indices)
            created["other"] = indices
        updated = {}
        if first_slot.updated.any():
            indices = first_slot.updated.next(length=step_size)
            steps += indices_len(indices)
            updated["table"] = indices
        if second_slot.updated.any():
            indices = second_slot.updated.next(length=step_size)
            steps += indices_len(indices)
            updated["other"] = indices
        first_table = first_slot.data()
        second_table = second_slot.data()
        if not self._dialog.is_started:
            join_start(
                first_table,
                second_table,
                dialog=self._dialog,
                created=created,
                updated=updated,
                **self.join_kwds
            )
        else:
            join_cont(
                first_table,
                second_table,
                dialog=self._dialog,
                created=created,
                updated=updated,
            )
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
