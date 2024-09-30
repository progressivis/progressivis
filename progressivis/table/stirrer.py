# -*- coding: utf-8 -*-
"Stirrer module for torturing progressivis in tests."
from __future__ import annotations

import random
import numpy as np

from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.pintset import PIntSet
from ..core.module import Module, def_parameter, def_input, def_output
from .api import PTable, PTableSelectedView

from typing import Optional, List, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep


@def_parameter("update_column", np.dtype(np.object_), "")
@def_parameter("update_rows", np.dtype(np.object_), None)
@def_parameter("delete_rows", np.dtype(np.object_), None)
@def_parameter("delete_threshold", np.dtype(np.object_), None)
@def_parameter("update_threshold", np.dtype(np.object_), None)
@def_parameter("del_twice", np.dtype(np.bool_), False)
@def_parameter("fixed_step_size", np.dtype(np.int_), 0)
@def_parameter("mode", np.dtype(np.object_), "random")
@def_input("table", type=PTable, required=True)
@def_output("result", PTable)
class Stirrer(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._update_column: str = self.params.update_column
        self._update_rows: Any = self.params.update_rows
        self._delete_rows: Any = self.params.delete_rows
        self._delete_threshold: Optional[int] = self.params.delete_threshold
        self._update_threshold: Optional[int] = self.params.update_threshold
        self._mode = self.params.mode
        self._steps = 0

    def test_delete_threshold(self, val: PIntSet) -> bool:
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
            self.result = PTable(
                self.generate_table_name("stirrer"),
                dshape=input_table.dshape,
            )
        raw_ids = self.result.index
        before_ = raw_ids  # PIntSet(raw_ids[raw_ids >= 0])
        v = input_table.loc[fix_loc(created), :]
        self.result.append(v)  # indices=PIntSet(created))
        delete: Union[List[int], PIntSet] = []
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
                del self.result.loc[delete[:mid]]
                del self.result.loc[delete[mid:]]
            elif delete:
                if PIntSet(delete) in self.result.index and PIntSet(
                    delete
                ) not in PIntSet(created):
                    steps += len(delete)
                    del self.result.loc[delete]
        if self._update_rows and len(before_):
            before_ -= PIntSet(delete)
            if isinstance(self._update_rows, int):
                updated = random.sample(
                    tuple(before_), min(self._update_rows, len(before_))
                )
            else:
                updated = self._update_rows
            v = np.random.rand(len(updated))
            if isinstance(updated, tuple):
                u_ids = PIntSet(updated[0])
                u_vals = updated[1]
                steps += len(u_ids)
                self.result.loc[u_ids, [self._update_column]] = u_vals
            elif updated:
                steps += len(updated)
                self.result.loc[fix_loc(updated), [self._update_column]] = [v]
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)


@def_parameter("update_column", np.dtype(object), "")
@def_parameter("delete_rows", np.dtype(object), None)
@def_parameter("delete_threshold", np.dtype(object), None)
@def_parameter("fixed_step_size", np.dtype(np.int_), 0)
@def_parameter("mode", np.dtype(object), "random")
@def_input("table", type=PTable, required=True)
@def_output("result", PTableSelectedView)
class StirrerView(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._update_column: str = self.params.update_column
        self._delete_rows: bool = self.params.delete_rows is not None
        self._delete_threshold: Optional[int] = self.params.delete_threshold
        self._mode: str = self.params.mode

    def test_delete_threshold(self, val: PIntSet) -> bool:
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
            self.result = PTableSelectedView(input_table, PIntSet([]))
        before_ = PIntSet(self.result.index)
        self.result.selection |= created
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
            self.result.selection -= PIntSet(delete)
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)
