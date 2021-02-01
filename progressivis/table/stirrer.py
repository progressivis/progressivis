# -*- coding: utf-8 -*-
"Stirrer module for torturing progressivis in tests."

import random
import numpy as np

from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.bitmap import bitmap
from progressivis.core.slot import SlotDescriptor
from .module import TableModule
from . import Table, TableSelectedView


class Stirrer(TableModule):
    parameters = [('update_column', str, ""),
                  ('update_rows', object, None),
                  ('delete_rows', object, None),
                  ('delete_threshold', object, None),
                  ('update_threshold', object, None),
                  ('del_twice', bool, False),
                  ('fixed_step_size', int, 0),
                  ('mode', str, "random")]
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._update_column = self.params.update_column
        self._update_rows = self.params.update_rows
        self._delete_rows = self.params.delete_rows
        self._delete_threshold = self.params.delete_threshold
        self._update_threshold = self.params.update_threshold
        self._mode = self.params.mode
        self._steps = 0

    def test_delete_threshold(self, val):
        if self._delete_threshold is None:
            return True
        return len(val) > self._delete_threshold
    def predict_step_size(self, duration):
        p = super().predict_step_size(duration)
        return max(p, 1000)
    def run_step(self, run_number, step_size, howlong):
        if self.params.fixed_step_size and False:
            step_size = self.params.fixed_step_size
        input_slot = self.get_input_slot('table')
        # input_slot.update(run_number)
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(step_size)
        steps = indices_len(created)
        self._steps += steps
        input_table = input_slot.data()
        if self._table is None:
            self._table = Table(self.generate_table_name('stirrer'),
                                dshape=input_table.dshape, )
        raw_ids = self._table.index
        before_ = raw_ids # bitmap(raw_ids[raw_ids >= 0])
        v = input_table.loc[fix_loc(created), :]
        self._table.append(v)  # indices=bitmap(created))
        delete = []
        if self._delete_rows and self.test_delete_threshold(before_):
            if isinstance(self._delete_rows, int):
                delete = random.sample(tuple(before_), min(self._delete_rows,
                                                           len(before_)))
            elif self._delete_rows == 'half':
                delete = random.sample(tuple(before_), len(before_)//2)
            elif self._delete_rows == 'all':
                delete = before_
            else:
                delete = self._delete_rows
            if delete and self.params.del_twice:
                mid = len(delete)//2
                del self._table.loc[delete[:mid]]
                del self._table.loc[delete[mid:]]
            elif delete:
                steps += len(delete)
                del self._table.loc[delete]
        if self._update_rows and len(before_):
            before_ -= bitmap(delete)
            if isinstance(self._update_rows, int):
                updated = random.sample(tuple(before_), min(self._update_rows,
                                                            len(before_)))
            else:
                updated = self._update_rows
            v = np.random.rand(len(updated))
            if updated:
                steps += len(updated)
                self._table.loc[fix_loc(updated), [self._update_column]] = [v]
        return self._return_run_step(self.next_state(input_slot),
                                     steps_run=steps)


class StirrerView(TableModule):
    parameters = [('update_column', str, ""),
                  ('delete_rows', object, None),
                  ('delete_threshold', object, None),
                  ('fixed_step_size', int, 0),
                  ('mode', str, "random")]
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._update_column = self.params.update_column
        self._delete_rows = self.params.delete_rows
        self._delete_threshold = self.params.delete_threshold
        self._mode = self.params.mode

    def test_delete_threshold(self, val):
        if self._delete_threshold is None:
            return True
        return len(val) > self._delete_threshold

    def run_step(self, run_number, step_size, howlong):
        if self.params.fixed_step_size and False:
            step_size = self.params.fixed_step_size
        input_slot = self.get_input_slot('table')
        # input_slot.update(run_number)
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(step_size, as_slice=False)
        created = fix_loc(created)
        steps = indices_len(created)
        input_table = input_slot.data()
        if self._table is None:
            self._table = TableSelectedView(input_table, bitmap([]))
        before_ = bitmap(self._table.index)
        self._table.mask |= created
        print(len(self._table.index))
        delete = []
        if self._delete_rows and self.test_delete_threshold(before_):
            if isinstance(self._delete_rows, int):
                delete = random.sample(tuple(before_), min(self._delete_rows,
                                                           len(before_)))
            elif self._delete_rows == 'half':
                delete = random.sample(tuple(before_), len(before_)//2)
            elif self._delete_rows == 'all':
                delete = before_
            else:
                delete = self._delete_rows
            self._table.mask -= bitmap(delete)
        return self._return_run_step(self.next_state(input_slot),
                                     steps_run=steps)
