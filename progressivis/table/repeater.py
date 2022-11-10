from __future__ import annotations
import numpy as np
import logging
from ..table.module import TableModule, ReturnRunStep
from ..core.slot import SlotDescriptor
from . import Table, TableSelectedView
from ..core.bitmap import bitmap
from typing import Optional, Any, Callable, Tuple

logger = logging.getLogger(__name__)
Shape = Tuple[int, ...]


class Computed:
    def __init__(self, computed=None):
        self.computed = {} if computed is None else computed

    def add_ufunc_column(
        self,
        name: str,
        col: str,
        ufunc: Callable,
        dtype: Optional[np.dtype[Any]] = None,
        xshape: Shape = (),
    ) -> None:
        self.computed[name] = dict(
            category="ufunc", ufunc=ufunc, column=col, dtype=dtype, xshape=xshape
        )


class Repeater(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, computed: Computed, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._computed = computed.computed

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            cols = (
                input_table.columns if self._columns is None else self._columns
            ) + list(self._computed.keys())
            self.result = TableSelectedView(
                input_table, bitmap([]), columns=cols, computed=self._computed
            )
        deleted: Optional[bitmap] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            steps = 1
            if deleted:
                self.selected.selection -= deleted
        created: Optional[bitmap] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += len(created)
            self.selected.selection |= created
        updated: Optional[bitmap] = None
        if input_slot.updated.any():
            # currently updates are ignored
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += len(updated)
        return self._return_run_step(self.next_state(input_slot), steps)
