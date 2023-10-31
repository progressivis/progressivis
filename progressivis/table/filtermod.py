from __future__ import annotations

import logging

from . import PTable, PTableSelectedView
from ..core.module import Module, ReturnRunStep, def_input, def_output, def_parameter, document
from ..core.utils import indices_len, fix_loc
from ..core.pintset import PIntSet
import numpy as np

from typing import Any

logger = logging.getLogger(__name__)


@document
@def_parameter("expr", np.dtype(object), "unknown", doc="a `numexpr <https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-operators>`_ alike filtering expression")
# @def_parameter("user_dict", np.dtype(object), None)
@def_input("table", PTable, doc="Data input")
@def_output("result", PTableSelectedView, doc="Returns a filterd view")
class FilterMod(Module):
    """
    Filtering module based on ``numexpr`` library.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)

    def reset(self) -> None:
        if self.result is not None:
            self.result.selection = PIntSet([])

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            self.result = PTableSelectedView(input_table, PIntSet([]))
        steps = 0
        if input_slot.updated.any():
            input_slot.reset()
            input_slot.update(run_number)
            self.reset()
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(length=step_size, as_slice=False)
            self.result.selection -= deleted
            steps += indices_len(deleted)
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            indices = fix_loc(created)
            steps += indices_len(created)
            eval_idx = input_table.eval(
                expr=self.params.expr,
                locs=np.array(indices),
                as_slice=False,
                result_object="index",
            )
            self.result.selection |= PIntSet(eval_idx)
        if not steps:
            return self._return_run_step(self.state_blocked, steps_run=0)
        return self._return_run_step(self.next_state(input_slot), steps)
