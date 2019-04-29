"Binary Join module."
from __future__ import absolute_import, division, print_function

from progressivis.core.utils import Dialog, indices_len
from progressivis.core.slot import SlotDescriptor
from .table import Table
from .module import TableModule
from .join import join, join_start, join_cont, join_reset
from .dshape import dshape_join
from collections import OrderedDict

class Paste(TableModule):
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
        super(Paste, self).__init__(**kwds)
        self.join_kwds = self._filter_kwds(kwds, join)
        self._dialog = Dialog(self)

    def run_step(self, run_number, step_size, howlong):
        #import pdb;pdb.set_trace()
        first_slot = self.get_input_slot('first')
        first_slot.update(run_number)
        second_slot = self.get_input_slot('second')
        second_slot.update(run_number)
        if first_slot.deleted.any() or second_slot.deleted.any():
            first_slot.reset()
            second_slot.reset()
            if self._table is not None:
                self._table.resize(0)
            first_slot.update(run_number)
            second_slot.update(run_number)
        first_slot.created.next(step_size)
        second_slot.created.next(step_size)
        first_slot.updated.next(step_size)
        second_slot.updated.next(step_size)
        with first_slot.lock:
            first_table = first_slot.data()
        with second_slot.lock:
            second_table = second_slot.data()
        if self._table is None:
            dshape, rename = dshape_join(first_table.dshape, second_table.dshape)
            self._table = Table(name=None, dshape=dshape)
        if len(first_table)==0 or len(second_table)==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        col_0 = first_table.columns[0]
        col_1 = second_table.columns[0]
        if len(self._table)==0:
            self._table.append(OrderedDict([(col_0, first_table.last(col_0)), (col_1, second_table.last(col_1))]), indices=[0])
        else:
            assert len(self._table) == 1
            if first_table.last(col_0)!=self._table.last(col_0):
                self._table[col_0].loc[0] =  first_table.last(col_0)
            if second_table.last(col_1)!=self._table.last(col_1):
                self._table[col_1].loc[0] = second_table.last(col_1)
        return self._return_run_step(self.next_state(first_slot), steps_run=1)
