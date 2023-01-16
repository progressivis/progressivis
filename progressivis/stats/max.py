from __future__ import annotations

import logging

import numpy as np

from ..core.utils import indices_len, fix_loc
from ..core.pintset import PIntSet
from ..core.module import ReturnRunStep
from ..table.module import PTableModule
from ..table.table import PTable
from ..core.slot import SlotDescriptor, Slot
from ..utils.psdict import PDict
from ..core.decorators import process_slot, run_if_any

from typing import Optional, Dict, Union, Any, Tuple

logger = logging.getLogger(__name__)


def _max_func(x: Any, y: Any) -> Any:
    try:  # fixing funny behaviour when max() is called with np.float64
        return np.maximum(x, y)
    except Exception:
        return max(x, y)


class Max(PTableModule):
    inputs = [SlotDescriptor("table", type=PTable, required=True)]

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super(Max, self).is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.psdict.fill(-np.inf)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices = ctx.table.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices)
            input_df = ctx.table.data()
            op = self.filter_columns(input_df, fix_loc(indices)).max(keepdims=False)
            if self.result is None:
                self.result = PDict(op)
            else:
                for k, v in self.psdict.items():
                    self.psdict[k] = _max_func(op[k], v)
            return self._return_run_step(self.next_state(ctx.table), steps)


def maximum_val_id(
    candidate_val: float, candidate_id: int, current_val: float, current_id: int
) -> Tuple[float, int, bool]:
    if candidate_val > current_val:
        return candidate_val, candidate_id, True
    return current_val, current_id, False


class ScalarMax(PTableModule):
    inputs = [SlotDescriptor("table", type=PTable, required=True)]

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
            self.psdict.fill(-np.inf)

    def reset_all(self, slot: Slot, run_number: int) -> None:
        slot.reset()
        self.reset()
        slot.update(run_number)

    def are_critical(self, updated_ids: PIntSet, data: PTable) -> bool:
        """
        check if updates invalidate the current max
        """
        for col, id in self._sensitive_ids.items():
            if id not in updated_ids:
                continue
            if bool(data.loc[id, col] < self.psdict[col]):
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
                # might become greater than the current MAX
                # so we will process these updates as creations
                # and we avoid a reset
                indices = slot.updated.next(length=step_size, as_slice=False)
        if indices is None:
            if not slot.created.any():
                return self._return_run_step(self.state_blocked, steps_run=0)
            indices = slot.created.next(length=step_size)  # returns a slice
        steps = indices_len(indices)
        input_df = slot.data()
        idxop = self.filter_columns(input_df, fix_loc(indices)).idxmax()
        if not self._sensitive_ids:
            self._sensitive_ids.update(idxop)
        if self.result is None:
            op = {k: input_df.loc[i, k] for (k, i) in idxop.items()}
            self.result = PDict(op)
        else:
            rich_op = {k: (input_df.loc[i, k], i) for (k, i) in idxop.items()}
            for k, v in self.psdict.items():
                candidate_val, candidate_id = rich_op[k]
                current_val = self.psdict[k]
                current_id = self._sensitive_ids[k]
                new_val, new_id, tst = maximum_val_id(
                    candidate_val, candidate_id, current_val, current_id
                )
                if tst:
                    self.psdict[k] = new_val
                    self._sensitive_ids[k] = new_id
        return self._return_run_step(self.next_state(slot), steps_run=steps)
