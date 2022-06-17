from __future__ import annotations

import logging
from ..table.module import TableModule, ReturnRunStep
from ..core.slot import SlotDescriptor
from . import Table, TableSelectedView
from ..core.bitmap import bitmap
from progressivis.core.utils import indices_len, fix_loc, norm_slice
from collections import defaultdict
from functools import singledispatchmethod as dispatch
import types
from collections import abc
from typing import Optional, List, Union, Any, Callable, Dict, Sequence

logger = logging.getLogger(__name__)


class SubColumn:
    def __init__(self, column: str, selection: Union[Sequence[int], slice],
                 tag: Optional[str] = None) -> None:
        def _make_tag():
            if tag:
                return tag
            if isinstance(selection, slice):
                sl = norm_slice(selection)
                return f"s{sl.start}_{sl.stop}{sl.step}"
            if isinstance(selection, Sequence):
                return "_".join([str(s) for s in selection])
            raise ValueError(f"Tag error for {selection}")
        self.column = column
        self.tag = _make_tag()
        self.selection = selection


class DateTime(SubColumn):
    def __init__(self, column: str, selection: str, tag: Optional[str] = None) -> None:
        idx = {fld: i for (i, fld) in enumerate("YMDhms")}
        if not (set(list(selection)) < set(idx.keys())):
            raise ValueError(f"unknown format: {selection}")
        self.tag = tag or selection
        self.column = column
        self.selection = [i for (k, i) in idx.items() if k in selection]


class GroupBy(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(
        self, by: Union[str, List[str], Callable, SubColumn], **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self.by = by
        self._index: Dict[Any, bitmap] = defaultdict(bitmap)
        self._input_table = None

    @dispatch
    def process_created(self, by, indices) -> None:
        raise NotImplementedError(f"Wrong type for {by}")

    @process_created.register
    def _(self, by: str, indices: bitmap) -> None:
        assert self._input_table is not None
        for i in indices:
            key = self._input_table.loc[i, by]
            self._index[key].add(i)

    @process_created.register
    def _(self, by: list, indices: bitmap) -> None:
        assert self._input_table is not None
        for i in indices:
            gen = self._input_table.loc[i, by]
            self._index[tuple(gen)].add(i)

    @process_created.register
    def _(self, by: types.FunctionType, indices: bitmap) -> None:
        for i in indices:
            self._index[by(self._input_table, i)].add(i)

    @process_created.register
    def _(self, by: SubColumn, indices: bitmap) -> None:
        assert self._input_table is not None
        col = by.column
        val = by.selection
        for i in indices:
            dt_vect = self._input_table.loc[i, col]
            self._index[tuple(dt_vect[val])].add(i)

    def process_deleted(self, indices: bitmap) -> None:
        for k, b in self._index.items():
            self._index[k] -= indices

    def items(self) -> abc.ItemsView:
        return self._index.items()

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
            if deleted:
                self.process_deleted(deleted)
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
        return self._return_run_step(self.next_state(input_slot), steps)
