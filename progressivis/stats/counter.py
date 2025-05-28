from __future__ import annotations

from progressivis.core.module import ReturnRunStep
from ..core.utils import indices_len
from ..core.module import Module, def_input, def_output
from ..table.table import PTable
from ..core.decorators import process_slot, run_if_any
import pandas as pd

from typing import Any

import logging

logger = logging.getLogger(__name__)


@def_input("table", PTable)
@def_output("result", PTable)
class Counter(Module):
    def __init__(self, **kwds: Any):
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.resize(0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(length=step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            data = pd.DataFrame(dict(counter=steps), index=[0])
            if self.result is None:
                self.result = PTable(
                    self.generate_table_name("counter"), data=data, create=True
                )
            elif len(self.result) == 0:
                self.result.append(data)
            else:
                self.result["counter"].loc[0] += steps
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
