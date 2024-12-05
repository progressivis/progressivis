from __future__ import annotations

import logging

import numpy as np

from ..core.utils import indices_len, fix_loc
from ..core.pintset import PIntSet
from ..core.module import ReturnRunStep, def_input, def_output, document
from ..core.module import Module
from ..table.table import PTable
from ..core.slot import Slot
from ..utils.psdict import PDict
from ..core.decorators import process_slot, run_if_any

from typing import Optional, Dict, Union, Any, Tuple, Sequence

logger = logging.getLogger(__name__)


def _min_func(x: Any, y: Any) -> Any:
    try:  # fixing funny behaviour when min() is called with np.float64
        return np.fmin(x, y)  # np.fmin avoids propagation of Nan
    except Exception:
        return min(x, y)



@document
@def_input("table", PTable, hint_type=Sequence[str], doc="the input table")
@def_output(
    "result",
    PDict,
    doc=("minimum values dictionary where every key represents" " a column"),
)
class Min(Module):
    """
    Computes the minimum of the values for every column
    """

    def __init__(
        self,
        **kwds: Any,
    ) -> None:
        """
        Args:
            columns: columns to be processed. When missing all input columns are processed
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super(Min, self).is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(np.inf)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices = ctx.table.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices)
            op = self.filter_slot_columns(ctx.table, fix_loc(indices)).min(keepdims=False)
            if self.result is None:
                self.result = PDict(op)
            else:
                for k, v in self.result.items():
                    self.result[k] = _min_func(op[k], v)
            return self._return_run_step(self.next_state(ctx.table), steps)


def minimum_val_id(
    candidate_val: float, candidate_id: int, current_val: float, current_id: int
) -> Tuple[float, int, bool]:
    if candidate_val < current_val:
        return candidate_val, candidate_id, True
    return current_val, current_id, False


@def_input("table", PTable)
@def_output("result", PDict)
class ScalarMin(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000
        self._sensitive_ids: Dict[str, int] = {}

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(np.inf)

    def reset_all(self, slot: Slot, run_number: int) -> None:
        slot.reset()
        self.reset()
        slot.update(run_number)

    def are_critical(self, updated_ids: PIntSet, data: PTable) -> bool:
        """
        check if updates invalidate the current min
        """
        assert self.result is not None
        for col, id in self._sensitive_ids.items():
            if id not in updated_ids:
                continue
            if bool(data.loc[id, col] > self.result[col]):
                return True
        return False

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        assert slot is not None
        indices: Optional[Union[None, PIntSet, slice]] = None
        sensitive_ids_bm = PIntSet(self._sensitive_ids.values())
        if slot.deleted.any():
            del_ids = slot.deleted.next(as_slice=False)
            if del_ids & sensitive_ids_bm:
                self.reset_all(slot, run_number)
            # else : deletes are not sensitive, just ignore them
        if slot.updated.any():
            sensitive_update_ids = slot.updated.changes & sensitive_ids_bm
            if sensitive_update_ids and self.are_critical(
                sensitive_update_ids, slot.data()
            ):
                self.reset_all(slot, run_number)
            else:
                # updates are not critical BUT some values
                # might become greater than the current MIN
                # so we will process these updates as creations
                # and we avoid a reset
                indices = slot.updated.next(length=step_size, as_slice=False)
        if indices is None:
            if not slot.created.any():
                return self._return_run_step(self.state_blocked, steps_run=0)
            indices = slot.created.next(length=step_size)  # returns a slice
        steps = indices_len(indices)
        input_df = slot.data()
        idxop = self.filter_slot_columns(slot, fix_loc(indices)).idxmin()
        if not self._sensitive_ids:
            self._sensitive_ids.update(idxop)
        if self.result is None:
            op = {k: input_df.loc[i, k] for (k, i) in idxop.items()}
            self.result = PDict(op)
        else:
            rich_op = {k: (input_df.loc[i, k], i) for (k, i) in idxop.items()}
            for k, v in self.result.items():
                candidate_val, candidate_id = rich_op[k]
                current_val = self.result[k]
                current_id = self._sensitive_ids[k]
                new_val, new_id, tst = minimum_val_id(
                    candidate_val, candidate_id, current_val, current_id
                )
                if tst:
                    self.result[k] = new_val
                    self._sensitive_ids[k] = new_id
        return self._return_run_step(self.next_state(slot), steps_run=steps)
