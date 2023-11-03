from __future__ import annotations
import numpy as np
import logging
from ..core.module import Module, ReturnRunStep, def_input, def_output
from . import PTable, PTableSelectedView
from ..core.pintset import PIntSet
from typing import Optional, Any, Callable, Tuple, Dict

logger = logging.getLogger(__name__)
Shape = Tuple[int, ...]


class Computed:
    def __init__(self, computed: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.computed = {} if computed is None else computed

    def add_ufunc_column(
        self,
        name: str,
        col: str,
        ufunc: Callable[..., Any],
        dtype: Optional[np.dtype[Any]] = None,
        xshape: Shape = (),
    ) -> None:
        self.computed[name] = dict(
            category="ufunc", ufunc=ufunc, column=col, dtype=dtype, xshape=xshape
        )


@def_input("table", PTable)
@def_output("result", PTableSelectedView)
class Repeater(Module):
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
                input_slot.hint or input_table.columns
            ) + list(self._computed.keys())
            self.result = PTableSelectedView(
                input_table, PIntSet([]), columns=cols, computed=self._computed
            )
        deleted: Optional[PIntSet] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            steps = 1
            if deleted:
                self.result.selection -= deleted
        created: Optional[PIntSet] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += len(created)
            self.result.selection |= created
        updated: Optional[PIntSet] = None
        if input_slot.updated.any():
            # currently updates are ignored
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += len(updated)
        return self._return_run_step(self.next_state(input_slot), steps)
