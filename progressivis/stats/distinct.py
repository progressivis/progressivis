from __future__ import annotations

import logging

from ..core.module import ReturnRunStep, def_input, def_output
from ..core.utils import indices_len, fix_loc
from ..core.decorators import process_slot, run_if_any
from ..core.module import Module
from ..table.table import PTable
from ..utils.psdict import PDict

from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@def_input("table", PTable)
@def_output("result", PDict)
class Distinct(Module):
    def __init__(
        self, columns: Optional[List[str]] = None, threshold: int = 56, **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self._columns = columns
        self._threshold = threshold
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            for k in self.result.keys():
                self.result[k] = set()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            # import pdb;pdb.set_trace()
            indices = ctx.table.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            input_df = ctx.table.data()
            op = self.filter_columns(input_df, fix_loc(indices))
            if self.result is None:
                self.result = PDict({k: set() for k in op.columns})
            for k, v in self.result.items():
                if v is None:  # too many values already detected
                    continue
                s = set(op[k].tolist())
                if len(s) > self._threshold:
                    self.result[k] = None
                    continue  # shortcut
                self.result[k].update(s)
                if len(self.result[k]) > self._threshold:
                    self.result[k] = None
            if not [
                v for v in self.result.values() if v is not None
            ]:  # no hope to detect categorical columns
                return self._return_run_step(self.state_ready, steps_run=steps)
            return self._return_run_step(self.next_state(ctx.table), steps)
