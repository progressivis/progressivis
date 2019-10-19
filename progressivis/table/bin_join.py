"Binary Join module."
from __future__ import absolute_import, division, print_function

from progressivis.core.utils import Dialog, indices_len
from progressivis.core.slot import SlotDescriptor
from .table import Table
from .module import TableModule
from .join import join, join_start, join_cont, join_reset


class BinJoin(TableModule):
    """
    Binary join module to join two tables and return a third one.

    Slots:
        first : Table module producing the first table to join
        second : Table module producing the second table to join
    Args:
        kwds : argument to pass to the join function
    """
    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('first', type=Table, required=True),
                         SlotDescriptor('second', type=Table, required=True)])
        super(BinJoin, self).__init__(**kwds)
        self.join_kwds = self._filter_kwds(kwds, join)
        self._dialog = Dialog(self)

    async def run_step(self, run_number, step_size, howlong):
        first_slot = self.get_input_slot('first')
        first_slot.update(run_number)
        second_slot = self.get_input_slot('second')
        second_slot.update(run_number)
        steps = 0
        if first_slot.deleted.any() or second_slot.deleted.any():
            first_slot.reset()
            second_slot.reset()
            if self._table is not None:
                self._table.resize(0)
                join_reset(self._dialog)
            first_slot.update(run_number)
            second_slot.update(run_number)
        created = {}
        if first_slot.created.any():
            indices = first_slot.created.next(step_size)
            steps += indices_len(indices)
            created["table"] = indices
        if second_slot.created.any():
            indices = second_slot.created.next(step_size)
            steps += indices_len(indices)
            created["other"] = indices
        updated = {}
        if first_slot.updated.any():
            indices = first_slot.updated.next(step_size)
            steps += indices_len(indices)
            updated["table"] = indices
        if second_slot.updated.any():
            indices = second_slot.updated.next(step_size)
            steps += indices_len(indices)
            updated["other"] = indices
        with first_slot.lock:
            first_table = first_slot.data()
        with second_slot.lock:
            second_table = second_slot.data()
        if not self._dialog.is_started:
            join_start(first_table, second_table,
                       dialog=self._dialog,
                       created=created, updated=updated,
                       **self.join_kwds)
        else:
            join_cont(first_table, second_table,
                      dialog=self._dialog,
                      created=created, updated=updated)
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
