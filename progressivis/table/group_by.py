from __future__ import annotations

import logging
from ..table.module import TableModule, ReturnRunStep
from ..core.slot import SlotDescriptor
from . import Table, TableSelectedView
from ..core.bitmap import bitmap
from progressivis.core.utils import indices_len, fix_loc
from collections import defaultdict
from functools import singledispatchmethod as dispatch
import types
from typing import Optional, List, Union, Any, Callable

logger = logging.getLogger(__name__)


class GroupBy(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, by: Union[List[str], Callable], **kwds: Any) -> None:
        super().__init__(**kwds)
        self.by = by
        self._index = defaultdict(bitmap)
        self._input_table = None

    @dispatch
    def process_created(self, by, indices):
        raise NotImplementedError(f"Wrong type for {by}")

    @process_created.register
    def _(self, by: str, indices: bitmap):
        for i in indices:
            key = self._input_table.loc[i, by]
            self._index[key].add(i)
        # print("index", len(self._index))

    @process_created.register
    def _(self, by: list, indices: bitmap):
        # import pdb;pdb.set_trace()
        for i in indices:
            gen = self._input_table.loc[i, by]
            self._index[tuple(gen)].add(i)

    @process_created.register
    def _(self, by: types.FunctionType, indices: bitmap):
        # import pdb;pdb.set_trace()
        for i in indices:
            self._index[by(self._input_table, i)].add(i)

    def process_deleted(indices):
        pass

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        self._input_table = input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            self.result = TableSelectedView(input_table, bitmap([]))
        deleted: Optional[bitmap] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            # steps += indices_len(deleted) # deleted are constant time
            steps = 1
            deleted = fix_loc(deleted)
            self.selected.selection -= deleted
        created: Optional[bitmap] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            created = fix_loc(created)
            steps += indices_len(created)
            self.selected.selection |= created
            self.process_created(self.by, created)
        updated: Optional[bitmap] = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            updated = fix_loc(updated)
            steps += indices_len(updated)
            # currently updates are ignored
            # NB: we assume that the updates do not concern the "grouped by" columns
        print("steps", steps)
        return self._return_run_step(self.next_state(input_slot), steps)
