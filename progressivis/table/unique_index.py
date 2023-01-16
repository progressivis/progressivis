from __future__ import annotations

import logging
from ..table.module import PTableModule, ReturnRunStep
from ..core.slot import SlotDescriptor
from . import PTable, PTableSelectedView
from ..core.pintset import PIntSet
from progressivis.core.utils import indices_len, fix_loc
from functools import singledispatchmethod as dispatch
from collections import abc
from typing import Optional, List, Union, Any, Dict, Tuple

logger = logging.getLogger(__name__)


class UniqueIndex(PTableModule):
    inputs = [SlotDescriptor("table", type=PTable, required=True)]

    def __init__(self, on: Union[str, List[str]], **kwds: Any) -> None:
        super().__init__(**kwds)
        self.on = on
        self._index: Dict[Any, int] = {}
        self._inverse: Dict[int, Any] = {}
        self._deleted: Dict[int, Any] = {}
        self._input_table = None

    @dispatch
    def process_created(self, on, indices) -> None:
        raise NotImplementedError(f"Wrong type for {on}")

    @process_created.register
    def _(self, on: str, indices: PIntSet) -> None:
        assert self._input_table is not None
        ds = str(self._input_table._column(on).dshape)
        if ds[0] in "123456789":  # ex: 6*uint16
            def _key(i: int) -> Tuple[Any]:
                return tuple(self._input_table.loc[i, on])
        else:
            def _key(i: int) -> Any:
                return self._input_table.loc[i, on]
        for i in indices:
            key = _key(i)
            assert (key not in self._index) or self._index[key] == i
            self._index[key] = i
            self._inverse[i] = key
            if i in self._deleted:
                del self._deleted[i]

    @process_created.register
    def _(self, on: list, indices: PIntSet) -> None:
        assert self._input_table is not None
        for i in indices:
            gen = self._input_table.loc[i, on]
            key = tuple(gen)
            assert i not in self._index or self._index[key] == i
            self._index[key] = i
            self._inverse[i] = key
            if i in self._deleted:
                del self._deleted[i]

    def process_deleted(self, indices: PIntSet) -> None:
        for i in indices:
            key = self._inverse[i]
            del self._inverse[i]
            del self._index[key]
            self._deleted[i] = key

    def items(self) -> abc.ItemsView:
        return self._index.items()

    @property
    def index(self):
        return self._index

    @property
    def deleted(self):
        return self._deleted

    def get_deleted_entries(self, ids) -> Any:
        return [entry for (i, entry) in self._deleted.items() if i in ids]

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
            self.result = PTableSelectedView(input_table, PIntSet([]))
        deleted: Optional[PIntSet] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            # steps += indices_len(deleted) # deleted are constant time
            steps = 1
            deleted = fix_loc(deleted)
            if deleted:
                self.process_deleted(deleted)
                self.selected.selection -= deleted
        created: Optional[PIntSet] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            created = fix_loc(created)
            steps += indices_len(created)
            self.selected.selection |= created
            self.process_created(self.on, created)
        updated: Optional[PIntSet] = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            updated = fix_loc(updated)
            steps += indices_len(updated)
            # currently updates are ignored
            # NB: we assume that the updates do not concern the "on" columns
        return self._return_run_step(self.next_state(input_slot), steps)
