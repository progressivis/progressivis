from __future__ import annotations
import numpy as np
import logging
from ..table.module import TableModule, ReturnRunStep
from ..core.slot import SlotDescriptor
from . import Table, TableSelectedView
from ..core.bitmap import bitmap
from collections import defaultdict
from functools import singledispatchmethod as dispatch
import types
from collections import abc
from typing import Optional, List, Union, Any, Callable, Dict, Sequence
from abc import ABCMeta, abstractproperty
logger = logging.getLogger(__name__)

UTIME = ["year", "month", "day", "hour", "minute", "second"]
UTIME_SET = set(UTIME)
UTIME_D = {fld: i for (i, fld) in enumerate(UTIME)}
UTIME_SHORT_D = {fld: i for (i, fld) in enumerate("YMDhms")}
DT_MAX = 6


class SubColumnABC(metaclass=ABCMeta):
    def __init__(self, column: str) -> None:
        self.column = column

    @abstractproperty
    def tag(self):
        ...

    @abstractproperty
    def selection(self):
        ...


class SimpleSC(SubColumnABC):
    def __init__(self, column, selection, tag=None):
        super().__init__(column)
        self._selection = selection
        self._tag = tag

    @property
    def tag(self):
        return self._tag

    @property
    def selection(self):
        return self._selection


class DTChain(SubColumnABC):
    def __init__(self, column):
        super().__init__(column)
        self.bag = []

    def __getattr__(self, name):
        if len(self.bag) >= DT_MAX:
            raise ValueError(f"Cannot chain more than {DT_MAX} items")
        if name not in UTIME_D:
            raise ValueError(f"Unknown item{name}")
        i = UTIME_D[name]
        if i in self.bag:
            raise ValueError(f"Attempt to chain {name} twice")
        self.bag.append(UTIME_D[name])
        return self

    def __getitem__(self, item):
        if isinstance(item, str):
            if not (set(list(item)) < set(UTIME_SHORT_D.keys())):
                raise ValueError(f"unknown format: {item}")
            selection = [i for (k, i) in UTIME_SHORT_D.items() if k in item]
            return SimpleSC(self.column, selection, tag=item)

    @property
    def selection(self):
        return self.bag

    @property
    def tag(self):
        return "".join([k for (k, v) in UTIME_SHORT_D.items() if v in self.bag])


class SCIndex:
    def __init__(self, column: str) -> None:
        self._column = column

    def __getitem__(self, item):
        if isinstance(item, str):
            if not (set(list(item)) < set(UTIME_SHORT_D.keys())):
                raise ValueError(f"unknown format: {item}")
            selection = [i for (k, i) in UTIME_SHORT_D.items() if k in item]
            return SimpleSC(self._column, selection, tag=item)
        if isinstance(item, (slice, Sequence)):
            return SimpleSC(self._column, item, tag="ix")
        raise ValueError(f"Invalid item {item}")


class SubColumn:
    def __init__(self, column: str) -> None:
        self._column = column
        self._dt = None
        self._idx = None

    @property
    def dt(self):
        if self._dt is None:
            self._dt = DTChain(self._column)
        return self._dt

    @property
    def ix(self):
        if self._idx is None:
            self._idx = SCIndex(self._column)
        return self._idx


class GroupBy(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(
            self, by: Union[str, List[str], Callable, SubColumn],
            keepdims: bool = False, **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self._raw_by = by
        self.by = None
        self._keepdims = keepdims
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
    def _(self, by: SubColumnABC, indices: bitmap) -> None:
        assert self._input_table is not None
        col = by.column
        val = by.selection
        if self._keepdims:
            mask_ = np.zeros(6, dtype=int)
            mask_[val] = 1
            for i in indices:
                dt_vect = self._input_table.loc[i, col]
                self._index[tuple(dt_vect*mask_)].add(i)
        else:
            for i in indices:
                dt_vect = self._input_table.loc[i, col]
                self._index[tuple(dt_vect[val])].add(i)

    def process_deleted(self, indices: bitmap) -> None:
        for k in self._index.keys():
            self._index[k] -= indices

    def items(self) -> abc.ItemsView:
        return self._index.items()

    @property
    def index(self):
        return self._index

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        self._input_table = input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.by is None:
            if isinstance(self._raw_by, str):
                by_shape = input_table._column(self._raw_by).shape
                if len(by_shape) == 1:
                    self.by = self._raw_by
                elif len(by_shape) == 2:
                    self.by = SimpleSC(self._raw_by, list(range(by_shape[1])))
                else:
                    raise ValueError(f"Group by not allowed for the {by_shape} shaped column {self._raw_by}")
            elif isinstance(self._raw_by, list):
                for c in self._raw_by:
                    if not isinstance(c, str):
                        raise ValueError("Multiple group by requires plain typed columns")
                    by_shape = input_table._column(c).shape
                    if len(by_shape) != 1:
                        raise ValueError("Multiple group by requires plain typed columns")
                self.by = self._raw_by
            else:
                self.by = self._raw_by
        if self.result is None:
            self.result = TableSelectedView(input_table, bitmap([]))
        deleted: Optional[bitmap] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            # steps += indices_len(deleted) # deleted are constant time
            steps = 1
            if deleted:
                self.process_deleted(deleted)
                self.selected.selection -= deleted
        created: Optional[bitmap] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += len(created)
            self.selected.selection |= created
            self.process_created(self.by, created)
        updated: Optional[bitmap] = None
        if input_slot.updated.any():
            # currently updates are ignored
            # NB: we assume that the updates do not concern the "grouped by" columns
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += len(updated)
        return self._return_run_step(self.next_state(input_slot), steps)
