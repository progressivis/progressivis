# -*- coding: utf-8 -*-
"Dummy module for torturing progressivis."

import random
import numpy as np

from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.bitmap import bitmap
from progressivis.core.slot import SlotDescriptor
from .module import TableModule
from . import Table

class DummyMod(TableModule):
    parameters = [('update_column', str, ""),
                  ('update_rows', object, None),
                  ('delete_rows', object, None),
                  ('delete_threshold', object, None),
                  ('update_threshold', object, None),
                  ('del_twice', bool, False),
                  ('fixed_step_size', int, 0),
                  ('mode', str, "random"),]
    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        super(DummyMod, self).__init__(**kwds)
        self._update_column = self.params.update_column
        self._update_rows = self.params.update_rows
        self._delete_rows = self.params.delete_rows
        self._delete_threshold = self.params.delete_threshold
        self._update_threshold = self.params.update_threshold
        self._mode = self.params.mode

    def test_delete_threshold(self, val):
        if self._delete_threshold is None:
            return True
        return len(val) > self._delete_threshold

    def run_step(self, run_number, step_size, howlong):
        if self.params.fixed_step_size and False:
             step_size = self.params.fixed_step_size
        input_slot = self.get_input_slot('table')
        input_slot.update(run_number)
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(step_size)
        steps = indices_len(created)
        with input_slot.lock:
            input_table = input_slot.data()
        p = self.params
        if self._table is None:
            self._table = Table(self.generate_table_name('dummy'), dshape=input_table.dshape, )
        raw_ids = self._table.index.values
        before_ = bitmap(raw_ids[raw_ids >= 0])
        v = input_table.loc[fix_loc(created), :]
        #print("creations: ", created)
        self._table.append(v) # indices=bitmap(created))
        delete = []
        if self._delete_rows and self.test_delete_threshold(before_):
            if isinstance(self._delete_rows, int):
                delete = random.sample(tuple(before_), min(self._delete_rows, len(before_)))
            elif self._delete_rows == 'half':
                delete = random.sample(tuple(before_), len(before_)//2)
            elif self._delete_rows == 'all':
                delete = before_
            else:
                delete = self._delete_rows
            #print("deletions: ", delete)
            if self.params.del_twice:
                mid = len(delete)//2
                del self._table.loc[delete[:mid]]
                del self._table.loc[delete[mid:]]
            else:
                del self._table.loc[delete]
        if self._update_rows and len(before_):
            before_ -= bitmap(delete)
            if isinstance(self._update_rows, int):
                updated = random.sample(tuple(before_), min(self._update_rows, len(before_)))
            else:
                updated = self._update_rows
            v = np.random.rand(len(updated))
            if updated:
                self._table.loc[fix_loc(updated), [self._update_column]] = [v]
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)
