# -*- coding: utf-8 -*-
"Stirrer module for torturing progressivis in tests."
from __future__ import annotations

import random
import numpy as np

from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.bitmap import bitmap
from progressivis.core.slot import SlotDescriptor
from .module import TableModule
from . import Table, TableSelectedView

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep


class Stirrer(TableModule):
    parameters = [
        ("update_column", np.dtype(np.object_), ""),
        ("update_rows", np.dtype(np.object_), None),
        ("delete_rows", np.dtype(np.object_), None),
        ("delete_threshold", np.dtype(np.object_), None),
        ("update_threshold", np.dtype(np.object_), None),
        ("del_twice", np.dtype(np.bool_), False),
        ("fixed_step_size", np.dtype(np.int_), 0),
        ("mode", np.dtype(np.object_), "random"),
    ]
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._update_column: str = self.params.update_column
        self._update_rows: bool = self.params.update_rows is not None
        self._delete_rows: bool = self.params.delete_rows is not None
        self._delete_threshold: Optional[int] = self.params.delete_threshold
        self._update_threshold: Optional[int] = self.params.update_threshold
        self._mode = self.params.mode
        self._steps = 0

    def test_delete_threshold(self, val: bitmap) -> bool:
        if self._delete_threshold is None:
            return True
        return len(val) > self._delete_threshold

    def predict_step_size(self, duration: float) -> int:
        p = super().predict_step_size(duration)
        return max(p, 1000)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if self.params.fixed_step_size and False:
            step_size = self.params.fixed_step_size
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(length=step_size)
        steps = indices_len(created)
        self._steps += steps
        input_table = input_slot.data()
        if self.result is None:
            self.result = Table(
                self.generate_table_name("stirrer"), dshape=input_table.dshape,
            )
        raw_ids = self.table.index
        before_ = raw_ids  # bitmap(raw_ids[raw_ids >= 0])
        v = input_table.loc[fix_loc(created), :]
        self.table.append(v)  # indices=bitmap(created))
        delete = []
        if self._delete_rows and self.test_delete_threshold(before_):
            if isinstance(self._delete_rows, int):
                delete = random.sample(
                    tuple(before_), min(self._delete_rows, len(before_))
                )
            elif self._delete_rows == "half":
                delete = random.sample(tuple(before_), len(before_) // 2)
            elif self._delete_rows == "all":
                delete = before_
            else:
                delete = self._delete_rows
            if delete and self.params.del_twice:
                mid = len(delete) // 2
                del self.table.loc[delete[:mid]]
                del self.table.loc[delete[mid:]]
            elif delete:
                steps += len(delete)
                del self.table.loc[delete]
        if self._update_rows and len(before_):
            before_ -= bitmap(delete)
            if isinstance(self._update_rows, int):
                updated = random.sample(
                    tuple(before_), min(self._update_rows, len(before_))
                )
            else:
                updated = self._update_rows
            v = np.random.rand(len(updated))
            if updated:
                steps += len(updated)
                self.table.loc[fix_loc(updated), [self._update_column]] = [v]
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)


class StirrerView(TableModule):
    parameters = [
        ("update_column", np.dtype(object), ""),
        ("delete_rows", np.dtype(object), None),
        ("delete_threshold", np.dtype(object), None),
        ("fixed_step_size", np.dtype(np.int_), 0),
        ("mode", np.dtype(object), "random"),
    ]
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._update_column: str = self.params.update_column
        self._delete_rows: bool = self.params.delete_rows is not None
        self._delete_threshold: Optional[int] = self.params.delete_threshold
        self._mode: str = self.params.mode

    def test_delete_threshold(self, val: bitmap) -> bool:
        if self._delete_threshold is None:
            return True
        return len(val) > self._delete_threshold

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if self.params.fixed_step_size and False:
            step_size = self.params.fixed_step_size
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(length=step_size, as_slice=False)
        # created = fix_loc(created)
        steps = indices_len(created)
        input_table = input_slot.data()
        if self.result is None:
            self.result = TableSelectedView(input_table, bitmap([]))
        before_ = bitmap(self.table.index)
        self.selected.selection |= created
        # print(len(self.table.index))
        delete = []
        if self._delete_rows and self.test_delete_threshold(before_):
            if isinstance(self._delete_rows, int):
                delete = random.sample(
                    tuple(before_), min(self._delete_rows, len(before_))
                )
            elif self._delete_rows == "half":
                delete = random.sample(tuple(before_), len(before_) // 2)
            elif self._delete_rows == "all":
                delete = before_
            else:
                delete = self._delete_rows
            self.selected.selection -= bitmap(delete)
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)
