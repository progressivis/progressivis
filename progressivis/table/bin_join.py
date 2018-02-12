from __future__ import absolute_import, division, print_function

from .nary import NAry
from .table import Table
from ..core.slot import SlotDescriptor
from progressivis.table.module import TableModule

from progressivis.core.utils import Dialog, indices_len, fix_loc

class BinJoin(TableModule):
    def __init__(self, scheduler=None, **kwds):
        """Join(on=None, how='left', lsuffix='', rsuffix='',sort=False,id=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])
        """
        self._add_slots(kwds,'input_descriptors',
                            [SlotDescriptor('first', type=Table, required=True),
                            SlotDescriptor('second', type=Table, required=True)])
        super(BinJoin, self).__init__(scheduler=scheduler, **kwds)
        self.join_kwds = self._filter_kwds(kwds, Table.join)
        self._dialog = Dialog(self)

    def run_step(self, run_number, step_size, howlong):
        first_slot = self.get_input_slot('first')
        first_slot.update(run_number)
        second_slot = self.get_input_slot('second')
        second_slot.update(run_number)
        steps = 0
        if first_slot.deleted.any() or second_slot.deleted.any():
            first_slot.reset(mid=self.id)
            second_slot.reset(mid=self.id)
            if self._table is not None:
                self._table.resize(0)
                first_slot.join_reset(self._dialog)
            first_slot.update(run_number)
            second_slot.update(run_number)
        created = {}
        if first_slot.created.any():
            indices = first_slot.created.next(step_size)
            steps += indices_len(indices)
            created["self"] = indices #fix_loc(indices)
        if second_slot.created.any():
            indices = second_slot.created.next(step_size)
            steps += indices_len(indices)
            created["other"] =indices  #fix_loc(indices)
        updated = {}
        if first_slot.updated.any():
            indices = first_slot.updated.next(step_size)
            steps += indices_len(indices)
            updated["self"] = indices #fix_loc(indices)
        if second_slot.updated.any():
            indices = second_slot.updated.next(step_size)
            steps += indices_len(indices)
            updated["other"] = indices #fix_loc(indices)
        with first_slot.lock:
            first_table = first_slot.data()
        with second_slot.lock:
            second_table = second_slot.data()
        if not self._dialog.is_started:
            status = first_table.join_start(second_table, dialog=self._dialog, created=created, updated=updated, **self.join_kwds)
            #self._dialog.set_started()
            #self._dialog.set_output(self._table)
        else:
            status = first_table.join_cont(second_table, dialog=self._dialog, created=created, updated=updated)
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
        
